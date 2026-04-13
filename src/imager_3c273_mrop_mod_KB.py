"""
Prepare proper measurement operator, prior and algorithm for imaging task
"""

from typing import Dict
import torch
import numpy as np
from astropy.io import fits

from .prox_operator import ProxOpAIRI, ProxOpElipse, ProxOpSARAPos
from .optimiser import FBAIRI, PDAIRI, FBSARA
from .utils import gen_imaging_weight
from .utils.io_3c273 import load_data_to_tensor
from .mrop_ri_measurement_operator.src.utils.solve_epsilon import solve_epsilon_diff_ab
from .ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft
from .mrop_ri_measurement_operator.src.mrop_vmap_mf_mod_KB import create_meas_op_ROP_vmap_mod_KB as create_meas_op_ROP
from .mrop_ri_measurement_operator import weighting_correction


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

    data = load_data_to_tensor(
        main_data_file=param_optimiser["data_file"],
        data_path="/".join(param_optimiser["data_file"].split("/")[:-1]),
        super_resolution=param_measop["superresolution"],
        data_weighting=param_measop["flag_data_weighting"],
        load_weight=param_measop["weight_load"],
        img_size=param_measop["img_size"],
        weight_type=param_measop["weight_type"],
        weight_robustness=param_measop["weight_robustness"],
        nfreqs=None,
        freq_num=None,
        use_ROP=param_measop["use_ROP"],
        vis_remove=17.7,
        dl_shift=param_measop["dl_shift"], #128 
        dm_shift=param_measop["dm_shift"], #-128
        dtype=param_measop["dtype"],
        device=param_measop["device"],
        # verbose=param_optimiser["verbose"],
    )

    if data["nFreqs"] == 1:
        data["flag"] = data["flag"][:, 0, :].unsqueeze(1)
        
        

    if param_measop["ROP_param"]["Q"] is None:
        assert "Q" in data, "number of anntennas Q is not in data and not provided"
        param_measop["ROP_param"]["Q"] = int(data["Q"])

    N = int(np.prod(param_measop["img_size"]))
    K = int(data["nFreqs"])
    V = int(param_measop["ROP_param"]["Q"] * (param_measop["ROP_param"]["Q"] - 1) // 2)
    B = data["B_per_ch"]
    Q = int(param_measop["ROP_param"]["Q"])

    print(f"INFO: Original dimensions: N = {N}, V = {V}, K = {K}, B = {B}, N_ratio = {param_measop["ROP_param"]["N_ratio"]}.")
    epsilon, P, M_B, M_K = solve_epsilon_diff_ab(N, Q, B, K, param_measop["ROP_param"]["N_ratio"])

    print(
        f"INFO: Calculated epsilon for MROP modulation dimensions: {epsilon:.4f} (epsilon = (N / VBK)^(1/4))."
    )
    param_measop["ROP_param"]["M_K"] = M_K
    param_measop["ROP_param"]["M_B"] = M_B
    param_measop["ROP_param"]["P"] = P
    param_measop["ROP_param"]["M"] = M_K * M_B
    
    print(
        f"INFO: MROP set with P = {param_measop["ROP_param"]["P"]}, M_K = {param_measop["ROP_param"]["M_K"]}, M_B = {param_measop["ROP_param"]["M_B"]}, M = {param_measop["ROP_param"]["M"]}."
    )
    print(
        f"INFO: PM / N = {param_measop["ROP_param"]["P"] * param_measop["ROP_param"]["M"] / N:.4f}",
        flush=True,
    )

    if param_measop["ROP_param"]["B"] is None:
        if "flag" in data and data["flag"] is not None and "B" not in data:
            data["B"] = data["flag"].shape[-1] / V * K
        assert "B" in data, "number of snapshots B is not in data and not provided"
        param_measop["ROP_param"]["B"] = int(data["B"])
    data, weight_corr = weighting_correction(data, param_measop["ROP_param"])
    print(
        f"INFO: Correction has been applied to the weighting for {param_measop['ROP_param']['ROP_type']}",
        flush=True,
    )

    meas_op = None

    nufft_op = create_meas_op_ROP(MeasOpPytorchFinufft)

    meas_op = nufft_op(
        ROP_param=param_measop["ROP_param"],
        u=data["u"],
        v=data["v"],
        num_chs=data["nFreqs"],
        # flag=data["flag"].numpy(force=True),
        ant1=data["ant1"],
        ant2=data["ant2"],
        batches=data["batches"],
        img_size=param_measop["img_size"],
        natural_weight=data["nW"],
        image_weight=data["nWimag"],
        device=param_measop["device"],
        dtype=param_measop["dtype"],
    )

    if param_measop["use_ROP"]:
        print(
            f"INFO: data size before {param_measop['ROP_param']['ROP_type']} is {data['y'].numel()}",
            flush=True,
        )
        if param_measop["ROP_param"]["ROP_type"] == "MROP":
            data["y"] = meas_op.MD(data["y"] * weight_corr)
        elif param_measop["ROP_param"]["ROP_type"] == "CROP":
            data["y"] = meas_op.D(data["y"] * weight_corr)
        print(
            f"INFO: data size after {param_measop['ROP_param']['ROP_type']} is {data['y'].numel()}",
            flush=True,
        )

    meas_op_approx = None
    if param_optimiser["approx_meas_op"]:
        from .ri_measurement_operator.pysrc.measOperator.meas_op_PSF import MeasOpPSF

        meas_op_approx = MeasOpPSF(
            data["u"],
            data["v"],
            param_measop["img_size"],
            natural_weight=data["nW"],
            image_weight=data["nWimag"],
            real_flag=True,
            normalise_psf=False,
            device=param_measop["device"],
            dtype=param_measop["dtype"],
        )

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
            data["y"],
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
        l2_bound = np.sqrt(torch.numel(data["y"]) + 2.0 * np.sqrt(torch.numel(data["y"])))
        if param_optimiser["verbose"]:
            print(
                "INFO: The theoretical l2 error bound is",
                f"{l2_bound}",
            )

        prox_op_dual_data = ProxOpElipse(
            center=data["y"],
            precond_weight=precond_weight,
            radius=l2_bound,
            device=meas_op.get_device(),
            dtype=meas_op.get_data_type_meas(),
        )

        optimiser = PDAIRI(
            data["y"],
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
            data["y"],
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
