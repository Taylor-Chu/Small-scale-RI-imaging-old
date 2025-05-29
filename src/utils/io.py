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

from ..ri_measurement_operator.pysrc.utils.gen_imaging_weights import gen_imaging_weights


def load_data_to_tensor(
    main_data_file: str,
    img_size: tuple,
    data_path: str = None,
    # dirty_file_path: str,
    # data_size: int,
    super_resolution: float = 1.5,
    # dirac_peak: float = None,
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

    if data is None:
        data = {}
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
                        if data_holo[key][key2].dtype.names and "imag" in data_holo[key][key2].dtype.names:
                            data_holo[key][key2] = (
                                data_holo[key][key2]["real"] + 1j * data_holo[key][key2]["imag"]
                            )
                else:
                    print("Type not implemented to be read here", h5obj)
    else:
        loadmat(main_data_file, mdict=data_holo)
        

    if use_ROP:
        data["Q"] = 27
        num_data = 0
        freqs = [data_holo["freqs"].squeeze()[0]]
        # freqs = data_holo["freqs"].squeeze()
        
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
                B = len(np.unique(data_tmp["batches_flagged"]))
            new_counter = counter + data_tmp["data_I"].size
            data["u"][0, 0, counter:new_counter] = data_holo["uvw"][:, 0][data_tmp["flag"].squeeze() == 1] / (
                speed_of_light / f
            )
            data["v"][0, 0, counter:new_counter] = data_holo["uvw"][:, 1][data_tmp["flag"].squeeze() == 1] / (
                speed_of_light / f
            )
            data["y"][0, 0, counter:new_counter] = data_tmp["data_I"].squeeze()
            data["nW"][0, 0, counter:new_counter] = data_tmp["weightsNat"].squeeze()
            
            data["batches"][counter:new_counter] = data_tmp["batches_flagged"].squeeze().astype(int) + B * i_f
            data["ant1"][counter:new_counter] = data_tmp["ant1_flagged"].squeeze().astype(int)
            data["ant2"][counter:new_counter] = data_tmp["ant2_flagged"].squeeze().astype(int)
            
            counter = new_counter
            
        data["B"] = B * len(freqs)
        
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
        
    else:
        # prepare data for MROP
        num_data = 0
        for i_f, f in enumerate(data_holo["freqs"].squeeze()):
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
        for i_f, f in enumerate(data_holo["freqs"].squeeze()):
            data_ch_i_f = loadmat(os.path.join(data_path, f"273-X08_data_ch_{i_f+1}.mat"))
            new_counter = counter + data_ch_i_f["data_I"].size
            data["u"][0, 0, counter:new_counter] = data_holo["uvw"][:, 0][data_ch_i_f["flag"].squeeze() == 1] / (
                speed_of_light / f
            )
            data["v"][0, 0, counter:new_counter] = data_holo["uvw"][:, 1][data_ch_i_f["flag"].squeeze() == 1] / (
                speed_of_light / f
            )
            data["y"][0, 0, counter:new_counter] = data_ch_i_f["data_I"].squeeze()
            data["nW"][0, 0, counter:new_counter] = data_ch_i_f["weightsNat"].squeeze()
            counter = new_counter

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

    del data_holo  # , tmp, uniques, counts
    gc.collect()
    
    if data_weighting:
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
        
    # data["y"] -= 17.7

    return data
