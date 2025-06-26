import gc
import glob
import os
import pathlib
from pathlib import Path

import h5py
import numpy as np
import torch
from astropy.io import fits
from scipy.constants import speed_of_light
from scipy.io import loadmat
from scipy.io.matlab import matfile_version
from torch.utils.data import Dataset


def load_data_to_tensor(
    main_data_file: str,
    img_size: tuple,
    data_path: str = None,
    target: str = "3c273",
    super_resolution: float = 1.5,
    freq_num: int = None,
    nfreqs: int = None,
    vis_remove: float = None,
    dl_shift: int = None,
    dm_shift: int = None,
    data_weighting: bool = True,
    weight_type: str = "briggs",
    weight_gridsize: float = 2.0,
    weight_robustness: float = 0.0,
    use_ROP: bool = False,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    data: dict = None,
    **kwargs,
):
    """Read u, v and imweight from specified path.

    Parameters
    ----------
    uv_file_path : str
        Path to the file containing sampling pattern, natural weights and (optional) imaging weights.
    super_resolution : float
        Super resolution factor.
    image_pixel_size : float, optional
        Image pixel size in arcsec, by default None
    data_weighting : bool, optional
        Flag to apply imaging weights, by default True
    load_weight : bool, optional
        Flag to load imaging weights from the file, by default False. If set to False and data_weighting is True, the imaging weights will be generated.
    load_die : bool, optional
        Flag to load DIEs from the file, by default False
    weight_name : str, optional
        Name of the imaging weights in the data file, by default 'nWimag'
    dtype : torch.dtype, optional
        Data type to be used, by default torch.float64
    device : torch.device, optional
        Device to be used, by default torch.device('cpu')
    verbose : bool, optional
        Flag to print information, by default True

    Returns
    -------
    data: dict
        Dictionary containing u, v, w, (optional) y, nW, (optional) nWimag and other information.
    """

    SR_limit_Fourier_domain = False
    if super_resolution < 1.0:
        SR_limit_Fourier_domain = True
        print(
            f"INFO: Super resolution factor (SR) needs to be at least 1.0, but was set to {super_resolution}.",
            flush=True,
        )
        print(f"INFO: The data will be limited to the grid corresponding to SR = 1.0.", flush=True)
    if data is None:
        data = {}

    if target == "3c273":
        super_resolution_target = super_resolution
        super_resolution = 1.87
        img_size = (1024, 1024)
        data_holo = {}
        mat_version, _ = matfile_version(main_data_file)
        if mat_version == 2:
            with h5py.File(main_data_file, "r") as h5File:
                for key, h5obj in h5File.items():
                    if isinstance(h5obj, h5py.Dataset):
                        data_holo[key] = np.array(h5obj)
                        if data_holo[key].dtype.names and "imag" in data_holo[key].dtype.names:
                            data_holo[key] = data_holo[key]["real"] + 1j * data_holo[key]["imag"]
                    elif isinstance(h5obj, h5py.Group):
                        data_holo[key] = {}
                        for key2, h5obj2 in h5obj.items():
                            data_holo[key][key2] = np.array(h5obj2)
                            if (
                                data_holo[key][key2].dtype.names
                                and "imag" in data_holo[key][key2].dtype.names
                            ):
                                data_holo[key][key2] = (
                                    data_holo[key][key2]["real"] + 1j * data_holo[key][key2]["imag"]
                                )
                    else:
                        print("Type not implemented to be read here", h5obj)
        else:
            loadmat(main_data_file, mdict=data_holo)
        if freq_num is not None:
            if nfreqs is None:
                freqs = [data_holo["freqs"].squeeze()[freq_num - 1]]
                print(f"INFO: Using frequency channel {freq_num}: {freqs[0]} Hz", flush=True)
            elif nfreqs > 1:
                freqs = data_holo["freqs"].squeeze()[freq_num - 1 : freq_num - 1 + nfreqs]
                print(f"INFO: Using {nfreqs} frequency channels.", flush=True)
                print(f"INFO: Using frequency channels {freq_num} to {freq_num + nfreqs - 1}: {freqs}", flush=True)
        else:
            freqs = data_holo["freqs"].squeeze()

        if use_ROP:
            data["Q"] = 27
            num_data = 0

            for i_f, f in enumerate(freqs):
                data_tmp = loadmat(
                    os.path.join(data_path, f"273-X08_data_ch_{i_f+1}.mat"), variable_names=["data_I"]
                )
                num_data += data_tmp["data_I"].size

            data["u"] = np.zeros((1, 1, num_data), dtype=np.float64)
            data["v"] = np.zeros((1, 1, num_data), dtype=np.float64)
            data["y"] = np.zeros((1, 1, num_data), dtype=np.complex128)
            data["nW"] = np.zeros((1, 1, num_data), dtype=np.float64)

            data["batches"] = np.zeros((num_data), dtype=int)
            data["ant1"] = np.zeros((num_data), dtype=int)
            data["ant2"] = np.zeros((num_data), dtype=int)
            counter = 0
            for i_f, f in enumerate(freqs):
                data_tmp = loadmat(os.path.join(data_path, f"273-X08_data_ch_{i_f+1}.mat"))
                if i_f == 0:
                    data["B_per_ch"] = len(np.unique(data_tmp["batches_flagged"]))
                new_counter = counter + data_tmp["data_I"].size
                data["u"][0, 0, counter:new_counter] = data_holo["uvw"][:, 0][
                    data_tmp["flag"].squeeze() == 1
                ] / (speed_of_light / f)
                data["v"][0, 0, counter:new_counter] = data_holo["uvw"][:, 1][
                    data_tmp["flag"].squeeze() == 1
                ] / (speed_of_light / f)
                data["y"][0, 0, counter:new_counter] = data_tmp["data_I"].squeeze()
                data["nW"][0, 0, counter:new_counter] = data_tmp["weightsNat"].squeeze()

                data["batches"][counter:new_counter] = (
                    data_tmp["batches_flagged"].squeeze().astype(int) + data["B_per_ch"] * i_f
                )
                data["ant1"][counter:new_counter] = data_tmp["ant1_flagged"].squeeze().astype(int)
                data["ant2"][counter:new_counter] = data_tmp["ant2_flagged"].squeeze().astype(int)

                counter = new_counter

            data["B"] = data["B_per_ch"] * len(freqs)

        else:
            # prepare data for MROP
            num_data = 0
            for i_f, f in enumerate(freqs):
                data_tmp = loadmat(
                    os.path.join(data_path, f"273-X08_data_ch_{i_f+1}.mat"), variable_names=["data_I"]
                )
                num_data += data_tmp["data_I"].size

            data["u"] = np.zeros((1, 1, num_data), dtype=np.float64)
            data["v"] = np.zeros((1, 1, num_data), dtype=np.float64)
            data["y"] = np.zeros((1, 1, num_data), dtype=np.complex128)
            data["nW"] = np.zeros((1, 1, num_data), dtype=np.float64)

            data["batches"] = None
            data["ant1"] = None
            data["ant2"] = None

            counter = 0
            for i_f, f in enumerate(freqs):
                data_ch_i_f = loadmat(os.path.join(data_path, f"273-X08_data_ch_{i_f+1}.mat"))
                new_counter = counter + data_ch_i_f["data_I"].size
                data["u"][0, 0, counter:new_counter] = data_holo["uvw"][:, 0][
                    data_ch_i_f["flag"].squeeze() == 1
                ] / (speed_of_light / f)
                data["v"][0, 0, counter:new_counter] = data_holo["uvw"][:, 1][
                    data_ch_i_f["flag"].squeeze() == 1
                ] / (speed_of_light / f)
                data["y"][0, 0, counter:new_counter] = data_ch_i_f["data_I"].squeeze()
                data["nW"][0, 0, counter:new_counter] = data_ch_i_f["weightsNat"].squeeze()
                counter = new_counter

        del data_holo 
        gc.collect()

    else:
        data_tmp = {}
        mat_version, _ = matfile_version(main_data_file)
        if mat_version == 2:
            with h5py.File(main_data_file, "r") as h5File:
                for key, h5obj in h5File.items():
                    if isinstance(h5obj, h5py.Dataset):
                        data_tmp[key] = np.array(h5obj)
                        if data_tmp[key].dtype.names and "imag" in data_tmp[key].dtype.names:
                            data_tmp[key] = data_tmp[key]["real"] + 1j * data_tmp[key]["imag"]
                    elif isinstance(h5obj, h5py.Group):
                        data_tmp[key] = {}
                        for key2, h5obj2 in h5obj.items():
                            data_tmp[key][key2] = np.array(h5obj2)
                            if data_tmp[key][key2].dtype.names and "imag" in data_tmp[key][key2].dtype.names:
                                data_tmp[key][key2] = (
                                    data_tmp[key][key2]["real"] + 1j * data_tmp[key][key2]["imag"]
                                )
                    else:
                        print("Type not implemented to be read here", h5obj)
        else:
            loadmat(main_data_file, mdict=data_tmp)

        try:
            frequency = np.array([data_tmp["frequency"].item()])
        except:
            frequency = data_tmp["frequency"].squeeze()
        nFreqs = len(frequency)

        data["u"] = data_tmp["u"].squeeze()
        data["v"] = data_tmp["v"].squeeze()

        num_data = data["u"].size

        data["batches"] = data_tmp.get("batches", None)
        data["ant1"] = data_tmp.get("ant1", None)
        data["ant2"] = data_tmp.get("ant2", None)

        for k in ["batches", "ant1", "ant2"]:
            if data[k] is not None:
                data[k] = data[k].squeeze().astype(int)
        data["ant1"][data["ant1"] > 4] -= 1
        data["ant2"][data["ant2"] > 4] -= 1
        data["batches"] -= 1

        if "Q" in data_tmp:
            try:
                data["Q"] = data_tmp["Q"].item()
            except:
                data["Q"] = data_tmp["Q"]

        if "B" in data_tmp:
            try:
                data["B"] = data_tmp["B"].item()
            except:
                data["B"] = data_tmp["B"]

        print("INFO: applying flagging to the sampling pattern", flush=True)

        flag = data_tmp["flag"].astype(bool)  # .squeeze()
        flag_counter = 0
        while len(flag.shape) > 3:
            flag = flag.squeeze(0)
            flag_counter += 1
            if flag_counter > 5:
                raise ValueError("Dimension of flags in the data cannot match dimension of the uv-points.")

        print("INFO: converting uv coordinate unit from meters to wavelength.", flush=True)
        # TODO: modify the Cygnus A .mat file so that False and True are flipped!!!
        data["u"] = np.concatenate(
            [
                data["u"][flag[iFreq, :] == True] / (speed_of_light / frequency[iFreq].item())
                for iFreq in range(nFreqs)
            ]
        )
        data["v"] = np.concatenate(
            [
                data["v"][flag[iFreq, :] == True] / (speed_of_light / frequency[iFreq].item())
                for iFreq in range(nFreqs)
            ]
        )

        # loop through frequencies
        data["y"] = data_tmp["y"].squeeze()[flag[0, :] == True]
        data["nW"] = data_tmp["nW"].squeeze()[flag[0, :] == True]

    max_proj_baseline = np.max(np.sqrt(data["u"] ** 2 + data["v"] ** 2))
    data["max_proj_baseline"] = max_proj_baseline
    spatial_bandwidth = 2 * max_proj_baseline
    image_pixel_size = (180.0 / np.pi) * 3600.0 / (super_resolution * spatial_bandwidth)
    print(
        f"INFO: default pixelsize: {image_pixel_size:.4e} arcsec, that is {super_resolution:.4f} x nominal resolution.",
        flush=True,
    )
    data["super_resolution"] = super_resolution

    data["u"] = torch.tensor(data["u"], dtype=dtype, device=device).view(1, 1, -1)
    data["v"] = -torch.tensor(data["v"], dtype=dtype, device=device).view(1, 1, -1)
    data["y"] = torch.tensor(data["y"], dtype=torch.complex128, device=device).view(1, 1, -1)
    data["nW"] = torch.tensor(data["nW"], dtype=torch.complex128, device=device).view(1, 1, -1)
    halfSpatialBandwidth = (180.0 / np.pi) * 3600.0 / (image_pixel_size) / 2.0

    data["u"] = data["u"] * np.pi / halfSpatialBandwidth
    data["v"] = data["v"] * np.pi / halfSpatialBandwidth

    # print("not doing it")
    if SR_limit_Fourier_domain:
        print("doing it")
        n_meas_SR_1 = data["u"].numel()
        bmax_SR_below_1 = torch.sqrt(
            (data["u"].clone() * super_resolution) ** 2
            + (data["v"].clone() * super_resolution) ** 2
        ).max()
        ant_filter = []
        for ant1 in np.unique(data["ant1"]):
            for ant2 in np.unique(data["ant2"]):
                if ant1 == ant2:
                    continue
                u_ant12 = data["u"][:, :, (data["ant1"] == ant1) & (data["ant2"] == ant2)]
                v_ant12 = data["v"][:, :, (data["ant1"] == ant1) & (data["ant2"] == ant2)]
                if (u_ant12.abs() > bmax_SR_below_1).any() or (v_ant12.abs() > bmax_SR_below_1).any():
                    ant_filter.append((ant1, ant2))

        # filter data["u"] and data["v"] based on ant_filter
        arc_filter = np.zeros(n_meas_SR_1, dtype=bool)
        for ant1, ant2 in ant_filter:
            arc_filter |= (data["ant1"] == ant1) & (data["ant2"] == ant2)
        data["u"] = data["u"].clone()[:, :, arc_filter==False].view(1, 1, -1)
        data["v"] = data["v"].clone()[:, :, arc_filter==False].view(1, 1, -1)
        data["y"] = data["y"].clone()[:, :, arc_filter==False].view(1, 1, -1)
        data["nW"] = data["nW"].clone()[:, :, arc_filter==False].view(1, 1, -1)
        if data["ant1"] is not None:
            data["ant1"] = data["ant1"][arc_filter==False]
        if data["ant2"] is not None:
            data["ant2"] = data["ant2"][arc_filter==False]
        if data["batches"] is not None:
            data["batches"] = data["batches"][arc_filter==False]

        print(f"INFO: Limiting the visibilities to the grid correpsonding to SR = 1.0", flush=True)
        print(
            f"INFO: That is {data['u'].numel()} / {n_meas_SR_1} ({data['u'].numel()/n_meas_SR_1 * 100 :.2f}%) visibilities.",
            flush=True,
        )

    if data_weighting:
        from ..ri_measurement_operator.pysrc.utils.gen_imaging_weights import gen_imaging_weights

        print(f"INFO: computing {weight_type} imaging weights...", flush=True)
        if weight_type == "briggs":
            if "weight_robustness" in data:
                weight_robustness = data["weight_robustness"].item()
                print(f"INFO: load weight_robustness from data file {weight_robustness}", flush=True)
            else:
                print(f"INFO: weight_robustness {weight_robustness}", flush=True)
        else:
            weight_robustness = 0.0
        data["nWimag"] = gen_imaging_weights(
            data["u"].clone(),
            data["v"].clone(),
            data["nW"],
            img_size,
            weight_type=weight_type,
            weight_gridsize=weight_gridsize,
            weight_robustness=weight_robustness,
        ).numpy(force=True)
    else:
        print("INFO: imaging weights will not be applied.", flush=True)
        data["nWimag"] = [
            1.0,
        ]
    data["nWimag"] = torch.tensor(data["nWimag"], dtype=dtype, device=device).view(1, 1, -1)

    if vis_remove is not None:
        data["y"] = data["y"] - vis_remove
        print(f"INFO: Removing {vis_remove} from the data.")

    if dl_shift is not None and dm_shift is not None:
        print(
            f"INFO: Applying phase shift to the data corresponding to shifting the centre of the original image by ({dl_shift}, {dm_shift}) pixels."
        )
        dl = 157 * image_pixel_size * np.pi
        dm = -143 * image_pixel_size * np.pi
        phase = torch.exp(1j * 2 * np.pi * (data["u"] * dl + data["v"] * dm))
        data["y"] = data["y"] * phase
        data["u"] = data["u"].clone() * (super_resolution / super_resolution_target)
        data["v"] = data["v"].clone() * (super_resolution / super_resolution_target)

    return data
