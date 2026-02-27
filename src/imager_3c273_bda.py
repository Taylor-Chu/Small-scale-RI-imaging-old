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

from .prox_operator import ProxOpAIRI, ProxOpElipse, ProxOpSARAPos
from .optimiser import FBAIRI, PDAIRI, FBSARA
from .utils import gen_imaging_weight

# from .utils.io_3c273 import load_data_to_tensor
from .utils.bda_averaging_3c273 import (
    apply_bda,
    compute_baseline_lengths,
    load_visibilities,
    load_uv_coordinates,
    load_natural_weights,
    average_visibilities,
    average_natural_weights,
)

from .ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import (
    MeasOpPytorchFinufft,
)


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

    # Load and inspect data
    print("=" * 70)
    print("DATA INSPECTION")
    print("=" * 70)

    data_path = os.path.dirname(param_optimiser["data_file"])
    raw = loadmat(param_optimiser["data_file"])

    vis_remove =17.7
    freqs = raw["freqs"].squeeze()
    ant1_all = raw["ant1"].squeeze().astype(int)
    ant2_all = raw["ant2"].squeeze().astype(int)
    bk_all = raw["batches"].squeeze().astype(int)
    u_all = raw["uvw"][:, 0]
    v_all = raw["uvw"][:, 1]

    Q = 27
    V = Q * (Q - 1) // 2  # 351 baselines
    F = len(freqs)
    B = int(np.unique(bk_all).size)
    N = len(u_all)

    print(f"\nArray configuration:")
    print(f"  Antennas (Q):        {Q}")
    print(f"  Baselines (V):       {V}")
    print(f"  Snapshots (B):       {B}")
    print(f"  Frequencies (F):     {F}")
    print(f"  Total entries (N):   {N} (expected: {B*V}, missing: {B*V-N})")

    print(f"\nFrequency coverage:")
    print(f"  Range: {freqs.min()/1e9:.4f} – {freqs.max()/1e9:.4f} GHz")
    print(f"  Channel spacing: {np.mean(np.diff(freqs))/1e6:.2f} MHz")
    print(f"  Total bandwidth: {(freqs.max()-freqs.min())/1e6:.0f} MHz")

    # Build (B, V) grid for baseline analysis
    unique_batches = np.unique(bk_all)
    batch_to_row = np.empty(int(unique_batches.max()) + 1, dtype=int)
    batch_to_row[unique_batches] = np.arange(B)
    row_idx = batch_to_row[bk_all]

    p_arr, q_arr = np.triu_indices(Q, k=1)
    bl_map = np.full((Q, Q), -1, dtype=int)
    bl_map[p_arr, q_arr] = np.arange(V)
    col_idx = bl_map[ant1_all, ant2_all]

    u_2d = np.zeros((B, V))
    v_2d = np.zeros((B, V))
    u_2d[row_idx, col_idx] = u_all
    v_2d[row_idx, col_idx] = v_all

    bl_lengths = compute_baseline_lengths(u_2d.ravel(), v_2d.ravel(), Q, B)
    print(f"\nBaseline lengths:")
    print(f"  Range: {bl_lengths.min()/1e3:.3f} – {bl_lengths.max()/1e3:.3f} km")
    print(f"  b_max/b_min: {bl_lengths.max()/bl_lengths.min():.1f}")

    # Compute image geometry parameters
    Npix = param_measop["img_size"][0]
    u_lam = u_2d * freqs.mean() / speed_of_light
    v_lam = v_2d * freqs.mean() / speed_of_light
    max_proj_bl_wav = np.sqrt(u_lam**2 + v_lam**2).max()
    spatial_bandwidth = 2 * max_proj_bl_wav
    pixel_size_rad = 1.0 / (param_measop["superresolution"] * spatial_bandwidth)
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
        param_optimiser["data_file"],
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
    vis_classical = load_visibilities(param_optimiser["data_file"], result_bda, verbose=True) - vis_remove
    u_classical, v_classical = load_uv_coordinates(param_optimiser["data_file"], result_bda, verbose=True)
    weights_classical = load_natural_weights(param_optimiser["data_file"], result_bda, verbose=True)

    # Convert flat natural weights to (F, B, V) grid for averaging
    # (average_natural_weights expects this format)
    F = result_bda["F"]
    B = result_bda["B"]
    V = result_bda["V"]
    flag_use = result_bda["flag_use"]

    weights_grid = np.zeros((F, B, V), dtype=np.float32)
    idx = 0
    for f in range(F):
        mask_f = flag_use[f] == 0  # (B, V) bool
        n_f = int(np.sum(mask_f))
        weights_grid[f][mask_f] = weights_classical[idx : idx + n_f]
        idx += n_f

    # Weight visibilities by their natural weights BEFORE averaging
    # For natural weighting, we need weighted average: sum(w_i*v_i)/sum(w_i)
    # not unweighted sum: sum(v_i)
    print("\nWeighting visibilities before BDA … ", end="", flush=True)
    # Square the sqrt(w) to get full weights w, then multiply visibilities
    weights_full = weights_classical**2
    vis_classical_weighted = vis_classical * weights_full
    print("done")

    # Average weighted visibilities with BDA (this gives sum of w_i*v_i per bin)
    print("Averaging weighted visibilities with BDA … ", end="", flush=True)
    vis_bda_weighted_sum, weights_bda = average_visibilities(vis_classical_weighted, None, result_bda)

    # Compute sum of natural weights per BDA bin
    print("Summing natural weights with BDA … ", end="", flush=True)
    # average_natural_weights takes sqrt(w) and returns sqrt(sum(w))
    nat_weights_bda_sqrt = average_natural_weights(weights_grid, result_bda)
    nat_weights_sum = nat_weights_bda_sqrt**2  # Square to get sum(w)

    # Compute weighted average: sum(w_i*v_i) / sum(w_i)
    vis_bda_avg_weighted = vis_bda_weighted_sum / nat_weights_sum

    print(f"\nData summary:")
    print(f"  Classical: {len(vis_classical):,} visibilities")
    print(f"  BDA:       {len(vis_bda_avg_weighted):,} samples (weighted averages)")
    print(f"  BDA bin sizes: min={weights_bda.min()}, max={weights_bda.max()}, mean={weights_bda.mean():.2f}")
    print(f"  Natural weights (classical): mean sqrt(w)={weights_classical.mean():.3f}")
    print(f"  Natural weights (BDA):       mean sqrt(w)={nat_weights_bda_sqrt.mean():.3f}")
    
    print("\n" + "=" * 70)
    print("CREATING MEASUREMENT OPERATORS")
    print("=" * 70)

    # Normalize classical coordinates
    u_classical_norm = u_classical * np.pi / halfSpatialBandwidth
    v_classical_norm = v_classical * np.pi / halfSpatialBandwidth
    # u_classical_torch = torch.tensor(u_classical_norm, dtype=torch.float64, device=param_measop["device"]).view(1, 1, -1)
    # v_classical_torch = -torch.tensor(v_classical_norm, dtype=torch.float64, device=param_measop["device"]).view(1, 1, -1)

    # Convert natural weights to torch tensors
    # nW_classical = torch.tensor(weights_classical, dtype=torch.float64).view(1, 1, -1)

    # Normalize BDA coordinates
    u_bda_norm = result_bda["u_avg"] * np.pi / halfSpatialBandwidth
    v_bda_norm = result_bda["v_avg"] * np.pi / halfSpatialBandwidth
    u_bda_torch = torch.tensor(u_bda_norm, dtype=torch.float64, device=param_measop["device"]).view(1, 1, -1)
    v_bda_torch = -torch.tensor(v_bda_norm, dtype=torch.float64, device=param_measop["device"]).view(1, 1, -1)

    # Convert BDA natural weights to torch tensors
    nW_bda = torch.tensor(nat_weights_bda_sqrt, dtype=torch.float64, device=param_measop["device"]).view(1, 1, -1)

    # print(f"\nClassical operator:")
    # print(f"  u range: [{u_classical_torch.min():.2f}, {u_classical_torch.max():.2f}]")
    # print(f"  v range: [{v_classical_torch.min():.2f}, {v_classical_torch.max():.2f}]")
    # print(f"  Samples: {u_classical_torch.numel():,}")
    # print(f"  Natural weights: mean={nW_classical.mean():.3f}")

    print(f"\nBDA operator:")
    print(f"  u range: [{u_bda_torch.min():.2f}, {u_bda_torch.max():.2f}]")
    print(f"  v range: [{v_bda_torch.min():.2f}, {v_bda_torch.max():.2f}]")
    print(f"  Samples: {u_bda_torch.numel():,}")
    # print(f"  Natural weights: mean={nW_bda.mean():.3f}")

    # Create measurement operators with natural weighting
    print("\nInstantiating NUFFT operators with natural weighting … ", end="", flush=True)
    
    y_bda_torch = torch.tensor(vis_bda_avg_weighted, dtype=torch.complex128, device=param_measop["device"]).view(1, 1, -1)
    y = y_bda_torch * nW_bda

    # meas_op_classical = MeasOpPytorchFinufft(
    #     u=u_classical_torch,
    #     v=v_classical_torch,
    #     img_size=param_measop["img_size"],
    #     natural_weight=nW_classical,
    #     dtype=torch.float64,
    # )

    meas_op = MeasOpPytorchFinufft(
        u=u_bda_torch,
        v=v_bda_torch,
        img_size=param_measop["img_size"],
        natural_weight=nW_bda,
        dtype=torch.float64,
        device=param_measop["device"]
    )
    
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
            y,
            meas_op,
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
            center=y,
            precond_weight=precond_weight,
            radius=l2_bound,
            device=meas_op.get_device(),
            dtype=meas_op.get_data_type_meas(),
        )

        optimiser = PDAIRI(
            y,
            meas_op,
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
            y,
            meas_op,
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
