"""
Prepare proper measurement operator, prior and algorithm for imaging task
"""

from typing import Dict
import torch
import numpy as np
from astropy.io import fits
from scipy.io import loadmat
from scipy.constants import speed_of_light
import os
import gc

from .prox_operator import ProxOpAIRI, ProxOpElipse, ProxOpSARAPos
from .optimiser import FBAIRI, PDAIRI, FBSARA
from .utils import gen_imaging_weight

# from .utils.io_3c273 import load_data_to_tensor
from .ri_measurement_operator.pysrc.utils.io_new import load_data_to_tensor
from .utils.bda_averaging import (
    apply_bda,
    # compute_baseline_lengths,
    # load_visibilities,
    # load_uv_coordinates,
    # load_natural_weights,
    average_visibilities,
    average_natural_weights,
)

from .ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import (
    MeasOpPytorchFinufft,
)

from .ri_measurement_operator.pysrc.utils.gen_imaging_weights import gen_imaging_weights


def imager(param_optimiser: Dict, param_measop: Dict, param_proxop: Dict) -> None:
    """
    Imager for small scale RI imaging task.

    This function prepares the measurement operator, prior, and algorithm for the imaging task.
    It supports different algorithms such as 'airi', 'usara', and 'cairi'. The function also
    handles the imaging process if the 'flag_imaging' is set in the 'param_optimiser'.

    Args:
        param_optimiser (dict): A dictionary containing the parameters for the optimiser.
            It includes parameters like 'algorithm', 'im_min_itr', 'im_max_itr', 'im_var_tol',
            'im_peak_est', 'heu_noise_scale', 'dnn_adaptive_peak', 'dnn_adaptive_peak_tol_min',
            'dnn_adaptive_peak_tol_max', 'dnn_adaptive_peak_tol_step', 'result_path', 'itr_save',
            'verbose', and 'flag_imaging'.
        param_measop (dict): A dictionary containing the parameters for the measurement operator.
            It includes parameters like 'superresolution', 'im_pixel_size', 'flag_data_weighting',
            'weight_load', 'img_size', 'weight_type', 'weight_gridsize', 'weight_robustness',
            'dtype', 'device', 'nufft_grid_size', 'nufft_kb_kernel_dim', and 'nufft_mode'.
        param_proxop (dict): A dictionary containing the parameters for the proximal operator.
            It includes parameters like 'dnn_shelf_path', 'dnn_apply_transform', 'device', 'dtype',
            and 'verbose'.
    """
    # initialisation

    # data_path = os.path.dirname(param_optimiser["data_file"])
    data = loadmat(param_optimiser["data_file"])

    data_classical = load_data_to_tensor(
        param_optimiser["data_file"],
        super_resolution=param_measop["superresolution"],
        # image_pixel_size=param_measop["im_pixel_size"],
        data_weighting=False,
        # load_weight=param_measop["weight_load"],
        img_size=param_measop["img_size"],
        # uv_unit="radians",
        # weight_type=param_measop["weight_type"],
        # weight_gridsize=param_measop["weight_gridsize"],
        # weight_robustness=param_measop["weight_robustness"],
        dtype=param_measop["dtype"],
        device=param_measop["device"],
        verbose=param_optimiser["verbose"],
    )

    super_resolution = data_classical["super_resolution"].item()
    max_proj_bl_wav = data_classical["max_proj_baseline"].item()

    Npix = param_measop["img_size"][0]
    spatial_bandwidth = 2 * max_proj_bl_wav
    pixel_size_rad = 1.0 / (super_resolution * spatial_bandwidth)
    img_half_width_rad = (Npix / 2) * pixel_size_rad
    pixel_size_arcsec = np.degrees(pixel_size_rad) * 3600
    halfSpatialBandwidth = (180.0 / np.pi) * 3600.0 / pixel_size_arcsec / 2.0

    print(f"\nImage parameters:")
    print(f"  Image size: {param_measop["img_size"][0]}×{param_measop["img_size"][1]} pixels")
    print(f"  Pixel size: {pixel_size_arcsec:.4f} arcsec")
    print(f"  Field of view: {np.degrees(2*img_half_width_rad)*3600:.2f} arcsec")
    print(f"  img_half_width_rad: {img_half_width_rad:.8f} rad")
    print("\n" + "=" * 70)
    print("COMPUTING BDA")
    print("=" * 70)

    result_bda = apply_bda(
        data,
        flag=data_classical["flag"],
        max_avg_time=param_measop["max_avg_time"],
        max_avg_freq=param_measop["max_avg_freq"],
        img_half_width_rad=img_half_width_rad,
        smearing_limit=param_measop["smearing_limit"],
        verbose=True,
    )

    print(f"\nBDA Results:")
    print(f"  Input visibilities:  {result_bda['total_input_unflagged']:,}")
    print(f"  Output samples:      {result_bda['total_output_samples']:,}")
    print(f"  Compression:         {result_bda['compression_actual']:.2f}×")
    print(
        f"  Data reduction:      {(1 - result_bda['total_output_samples']/result_bda['total_input_unflagged'])*100:.1f}%"
    )

    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    # Load classical (full) visibilities and coordinates
    print("\nLoading classical data:")
    nWimag = gen_imaging_weights(
        data_classical["u"].clone(),
        data_classical["v"].clone(),
        data_classical["nW"],
        param_measop["img_size"],
        weight_type="briggs",
        weight_robustness=data["weight_robustness"].item(),
    ).to(device=param_measop["device"])

    meas_op = MeasOpPytorchFinufft(
        u=data_classical["u"],
        v=data_classical["v"],
        natural_weight=data_classical["nW"],
        image_weight=nWimag,
        img_size=param_measop["img_size"],
        dtype=torch.float64,
        device=param_measop["device"],
    )

    y = data["y"].ravel()
    nW_classical = data_classical["nW"].numpy(force=True).ravel()

    # del data_classical
    # gc.collect()

    # flag_use = result_bda["flag_use"]
    # Convert flat natural weights to (F, B, V) grid for averaging
    # (average_natural_weights expects this format)
    F = result_bda["F"]
    B = result_bda["B"]
    V = result_bda["V"]
    not_flagged = result_bda["not_flagged"]  # (F, B, V) bool

    natural_weights_sqrt = np.zeros((F, B, V), dtype=np.float64)
    natural_weights_sqrt[not_flagged] = nW_classical  # same freq-major C-order
    weights_full = nW_classical**2
    vis_classical_weighted = y * weights_full

    # Weight visibilities by their natural weights BEFORE averaging
    # For natural weighting, we need weighted average: sum(w_i*v_i)/sum(w_i)
    # not unweighted sum: sum(v_i)
    print("\nWeighting visibilities before BDA … ", end="", flush=True)
    # Square the sqrt(w) to get full weights w, then multiply visibilities
    weights_full = nW_classical**2
    vis_classical_weighted = y * weights_full
    print("done", flush=True)

    # Average weighted visibilities with BDA (this gives sum of w_i*v_i per bin)
    print("Averaging weighted visibilities with BDA … ", end="", flush=True)
    vis_bda_weighted_sum, weights_bda = average_visibilities(vis_classical_weighted, None, result_bda)

    # Compute sum of natural weights per BDA bin
    print("Summing natural weights with BDA … ", end="", flush=True)
    # average_natural_weights takes sqrt(w) and returns sqrt(sum(w))
    nat_weights_bda_sqrt = average_natural_weights(natural_weights_sqrt, result_bda)
    nat_weights_sum = nat_weights_bda_sqrt**2  # Square to get sum(w)

    # Compute weighted average: sum(w_i*v_i) / sum(w_i)
    vis_bda_avg_weighted = vis_bda_weighted_sum / nat_weights_sum

    print(f"\nData summary:")
    print(f"  Classical: {len(vis_classical_weighted):,} visibilities")
    print(f"  BDA:       {len(vis_bda_avg_weighted):,} samples (weighted averages)")
    print(f"  BDA bin sizes: min={weights_bda.min()}, max={weights_bda.max()}, mean={weights_bda.mean():.2f}")
    print(f"  Natural weights (classical): mean sqrt(w)={nW_classical.mean():.3f}")
    print(f"  Natural weights (BDA):       mean sqrt(w)={nat_weights_bda_sqrt.mean():.3f}")

    print("\n" + "=" * 70)
    print("CREATING MEASUREMENT OPERATORS")
    print("=" * 70)

    u_bda = result_bda["u_avg"]
    v_bda = result_bda["v_avg"]

    max_proj_baseline = np.max(np.sqrt(u_bda**2 + v_bda**2))
    data["max_proj_baseline"] = max_proj_baseline
    if super_resolution is None:
        super_resolution = data["super_resolution"].item()
    spatial_bandwidth = 2 * max_proj_baseline
    image_pixel_size = (180.0 / np.pi) * 3600.0 / (super_resolution * spatial_bandwidth)

    data["image_pixel_size"] = image_pixel_size
    halfSpatialBandwidth = (180.0 / np.pi) * 3600.0 / (image_pixel_size) / 2.0

    u_bda = u_bda * np.pi / halfSpatialBandwidth
    v_bda = v_bda * np.pi / halfSpatialBandwidth

    u_bda = torch.tensor(u_bda, dtype=torch.float64).view(1, 1, -1)
    v_bda = -torch.tensor(v_bda, dtype=torch.float64).view(1, 1, -1)

    # y = meas_op.forward_op(gdth)
    dirty = (
        meas_op.adjoint_op(data_classical["y"] * data_classical["nW"] * nWimag).squeeze().numpy(force=True)
    )
    y = data_classical["y"].numpy(force=True).ravel()
    nW_classical = data_classical["nW"].numpy(force=True).ravel()

    F = result_bda["F"]
    B = result_bda["B"]
    V = result_bda["V"]
    not_flagged = result_bda["not_flagged"]  # (F, B, V) bool

    natural_weights_sqrt = np.zeros((F, B, V), dtype=np.float64)
    natural_weights_sqrt[not_flagged] = nW_classical  # same freq-major C-order
    weights_full = nW_classical**2
    vis_classical_weighted = y * weights_full

    vis_bda_weighted_sum, weights_bda = average_visibilities(vis_classical_weighted, data, result_bda)
    nat_weights_bda_sqrt = average_natural_weights(natural_weights_sqrt, result_bda)
    nat_weights_sum = nat_weights_bda_sqrt**2  # Square to get sum(w)

    nW_bda_torch = torch.tensor(
        nat_weights_bda_sqrt, dtype=torch.float64, device=param_measop["device"]
    ).view(1, 1, -1)

    nWimag_bda = gen_imaging_weights(
        u_bda.clone(),
        v_bda.clone(),
        nW_bda_torch,
        param_measop["img_size"],
        weight_type="briggs",
        weight_robustness=data["weight_robustness"].item(),
    ).to(device=param_measop["device"])

    meas_op_bda = MeasOpPytorchFinufft(
        u=u_bda,
        v=v_bda,
        natural_weight=nW_bda_torch,
        image_weight=nWimag_bda,
        img_size=param_measop["img_size"],
        dtype=torch.float64,
        device=param_measop["device"],
    )

    # Compute weighted average: sum(w_i*v_i) / sum(w_i)
    y_bda = vis_bda_weighted_sum / nat_weights_sum
    y_bda_torch = torch.tensor(y_bda, dtype=torch.complex128, device=param_measop["device"]).view(1, 1, -1)

    dirty = (
        meas_op.adjoint_op(data_classical["y"] * data_classical["nW"] * nWimag).squeeze().numpy(force=True)
    )
    dirty_bda = meas_op_bda.adjoint_op(y_bda_torch * nW_bda_torch * nWimag_bda).squeeze().numpy(force=True)
    dirty_rel_diff = np.linalg.norm(dirty.ravel() - dirty_bda.ravel()) / np.linalg.norm(dirty.ravel())
    assert (
        dirty_rel_diff < 0.02
    ), f"Dirty images from classical and BDA operators differ by more than 2%: {dirty_rel_diff:.4f}"
    
    psf = meas_op.get_psf().numpy(force=True).squeeze()
    psf_bda = meas_op_bda.get_psf().numpy(force=True).squeeze()
    psf_rel_diff = np.linalg.norm(psf.ravel() - psf_bda.ravel()) / np.linalg.norm(psf.ravel())
    assert (
        psf_rel_diff < 0.01
    ), f"PSFs from classical and BDA operators differ by more than 1%: {psf_rel_diff:.4f}"
    #####

    meas_op_approx = None

    optimiser = None
    if param_optimiser["algorithm"] == "airi":
        prox_op_airi = ProxOpAIRI(
            param_proxop["dnn_shelf_path"],
            rand_trans=param_proxop["dnn_apply_transform"],
            device=param_proxop["device"],
            dtype=param_proxop["dtype"],
            verbose=param_proxop["verbose"],
        )

        optimiser = FBAIRI(
            y_bda_torch,
            meas_op_bda,
            prox_op_airi,
            meas_op_approx=meas_op_approx,
            im_min_itr=param_optimiser["im_min_itr"],
            im_max_itr=param_optimiser["im_max_itr"],
            im_var_tol=param_optimiser["im_var_tol"],
            im_peak_est=param_optimiser["im_peak_est"],
            heu_noise_scale=param_optimiser["heu_noise_scale"],
            new_heu=param_optimiser["new_heu"],
            adapt_net_select=param_optimiser["dnn_adaptive_peak"],
            peak_tol_min=param_optimiser["dnn_adaptive_peak_tol_min"],
            peak_tol_max=param_optimiser["dnn_adaptive_peak_tol_max"],
            peak_tol_step=param_optimiser["dnn_adaptive_peak_tol_step"],
            save_pth=param_optimiser["result_path"],
            file_prefix=param_optimiser["file_prefix"],
            iter_save=param_optimiser["itr_save"],
            verbose=param_optimiser["verbose"],
        )

    elif param_optimiser["algorithm"] == "cairi":
        prox_op_airi = ProxOpAIRI(
            param_proxop["dnn_shelf_path"],
            rand_trans=param_proxop["dnn_apply_transform"],
            device=param_proxop["device"],
            dtype=param_proxop["dtype"],
            verbose=param_proxop["verbose"],
        )

        # preconditioning weight
        if param_optimiser["precond_flag"]:
            precond_weight = (
                torch.from_numpy(
                    gen_imaging_weight(
                        data["u"].cpu().numpy(),
                        data["v"].cpu().numpy(),
                        param_measop["img_size"],
                        weight_type="uniform",
                        grid_size=2,
                    ).reshape(1, 1, -1)
                )
                ** 2
            )
        else:
            precond_weight = torch.ones(1, 1)

        # Theoretical l2 error bound, assume chi-square distribution, tau=1
        l2_bound = np.sqrt(torch.numel(y) + 2.0 * np.sqrt(torch.numel(y)))
        if param_optimiser["verbose"]:
            print(
                "INFO: The theoretical l2 error bound is",
                f"{l2_bound}",
            )

        prox_op_dual_data = ProxOpElipse(
            center=y_bda_torch,
            precond_weight=precond_weight,
            radius=l2_bound,
            device=meas_op.get_device(),
            dtype=meas_op.get_data_type_meas(),
        )

        optimiser = PDAIRI(
            y_bda_torch,
            meas_op_bda,
            prox_op_airi,
            prox_op_dual_data,
            im_min_itr=param_optimiser["im_min_itr"],
            im_max_itr=param_optimiser["im_max_itr"],
            im_var_tol=param_optimiser["im_var_tol"],
            im_peak_est=param_optimiser["im_peak_est"],
            heu_noise_scale=param_optimiser["heu_noise_scale"],
            adapt_net_select=param_optimiser["dnn_adaptive_peak"],
            peak_tol_min=param_optimiser["dnn_adaptive_peak_tol_min"],
            peak_tol_max=param_optimiser["dnn_adaptive_peak_tol_max"],
            peak_tol_step=param_optimiser["dnn_adaptive_peak_tol_step"],
            save_pth=param_optimiser["result_path"],
            file_prefix=param_optimiser["file_prefix"],
            iter_save=param_optimiser["itr_save"],
            verbose=param_optimiser["verbose"],
        )

    elif param_optimiser["algorithm"] == "usara":
        prox_op_sara = ProxOpSARAPos(
            param_measop["img_size"],
            device=param_proxop["device"],
            dtype=param_proxop["dtype"],
            verbose=param_proxop["verbose"],
        )

        optimiser = FBSARA(
            y_bda_torch,
            meas_op_bda,
            prox_op_sara,
            use_ROP=param_measop["use_ROP"],
            meas_op_approx=meas_op_approx,
            im_min_itr=param_optimiser["im_min_itr"],
            im_max_itr=param_optimiser["im_max_itr"],
            im_var_tol=param_optimiser["im_var_tol"],
            heu_reg_scale=param_optimiser["heu_reg_param_scale"],
            new_heu=param_optimiser["new_heu"],
            im_max_itr_outer=param_optimiser["im_max_outer_itr"],
            im_var_tol_outer=param_optimiser["im_var_outer_tol"],
            save_pth=param_optimiser["result_path"],
            file_prefix=param_optimiser["file_prefix"],
            reweight_save=param_optimiser["reweighting_save"],
            verbose=param_optimiser["verbose"],
        )

    # imaging
    if param_optimiser["flag_imaging"]:
        # initialisation
        optimiser.initialisation()
        # run imaging loop
        optimiser.run()
        # finalisation
        optimiser.finalisation()

        # calculate final metrics
        if param_optimiser["verbose"]:
            img_model = optimiser.get_model_image()
            img_residual = optimiser.get_residual_image()
            img_dirty = optimiser.get_dirty_image()
            psf = optimiser.get_psf()

            img_residual_std = np.std(img_residual).item()
            img_residual_std_noramalised = img_residual_std / psf.max().item()
            img_residual_ratio = np.linalg.norm(img_residual.flatten()) / np.linalg.norm(img_dirty.flatten())
            print(
                "INFO: The standard deviation of the final",
                f"residual dirty image is {img_residual_std}",
            )
            print(
                "INFO: The standard deviation of the normalised",
                f"final residual dirty image is {img_residual_std_noramalised}",
            )
            print(
                "INFO: The ratio between the norm of the residual",
                f"and the dirty image: ||residual|| / || dirty || = {img_residual_ratio}",
            )

            if param_optimiser["groundtruth"]:
                img_gdth = fits.getdata(param_optimiser["groundtruth"]).astype(np.double)
                rsnr = 20 * np.log10(
                    np.linalg.norm(img_gdth.flatten())
                    / np.linalg.norm(img_gdth.flatten() - img_model.flatten())
                )
                print(
                    "INFO: The signal-to-noise ratio of the final",
                    f"reconstructed image is {rsnr} dB",
                )
