"""
Prepare proper measurement operator, prior and algorithm for imaging task
"""

from typing import Dict
import torch
import numpy as np
from astropy.io import fits
import gc

from .prox_operator import ProxOpAIRI, ProxOpElipse, ProxOpSARAPos
from .optimiser import FBAIRI, PDAIRI, FBSARA
from .utils import gen_imaging_weight

# from .utils.io_3c273 import load_data_to_tensor
from .ri_measurement_operator.pysrc.utils.io_new import load_data_to_tensor
from .mrop_ri_measurement_operator.src.utils.solve_epsilon import solve_epsilon_diff_ab
from .ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft
from .mrop_ri_measurement_operator.src.mrop_vmap_mf_mod_KB import (
    create_meas_op_ROP_vmap_mod_KB as create_meas_op_ROP,
)
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
        param_optimiser["data_file"],
        super_resolution=param_measop["superresolution"],
        image_pixel_size=param_measop["im_pixel_size"],
        data_weighting=param_measop["flag_data_weighting"],
        load_weight=param_measop["weight_load"],
        img_size=param_measop["img_size"],
        uv_unit="radians",
        weight_type=param_measop["weight_type"],
        # weight_gridsize=param_measop["weight_gridsize"],
        weight_robustness=param_measop["weight_robustness"],
        dtype=param_measop["dtype"],
        device=param_measop["device"],
        verbose=param_optimiser["verbose"],
    )

    if param_measop["ROP_param"]["Q"] is None:
        assert "Q" in data, "number of anntennas Q is not in data and not provided"
        param_measop["ROP_param"]["Q"] = int(data["Q"])

    N = int(np.prod(param_measop["img_size"]))
    K = int(data["nFreqs"])
    V = int(param_measop["ROP_param"]["Q"] * (param_measop["ROP_param"]["Q"] - 1) // 2)
    B = int(data["B_per_ch"])
    Q = int(param_measop["ROP_param"]["Q"])

    print(
        f"INFO: Original dimensions: N = {N}, V = {V}, K = {K}, B = {B}, N_ratio = {param_measop["ROP_param"]["N_ratio"]}."
    )
    epsilon, P, M_B, M_K = solve_epsilon_diff_ab(N, Q, B, K, param_measop["ROP_param"]["N_ratio"])

    print(f"INFO: Calculated epsilon for MROP modulation dimensions: {epsilon:.4f}")
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

    param_measop["ROP_param"]["B"] = int(data["B_per_ch"] * K)
    data, weight_corr = weighting_correction(data, param_measop["ROP_param"])
    # adjusting correction factor by 1/sqrt(K) if same_ab_all is True
    if param_measop["ROP_param"]["same_ab_all"]:
        weight_corr /= np.sqrt(K)
        print(
            f"INFO: Adjusting weighting correction by 1/sqrt(K) for same_ab_all=True, new weight_corr={weight_corr:.4f}",
            flush=True,
        )
        data["nWimag"] /= np.sqrt(K)
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
        flag=data["flag"],
        # ant1=data["ant1"],
        # ant2=data["ant2"],
        # batches=data["batches"],
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

    y = data["y"].clone()
    del data
    gc.collect()
    torch.cuda.empty_cache()

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


# """
# Prepare proper measurement operator, prior and algorithm for imaging task
# """

# from typing import Dict
# import torch
# import numpy as np
# from astropy.io import fits

# from .prox_operator import ProxOpAIRI, ProxOpElipse, ProxOpSARAPos
# from .optimiser import FBAIRI, PDAIRI, FBSARA
# from .utils import gen_imaging_weight
# from .ri_measurement_operator.pysrc.utils.io import load_data_to_tensor
# from .mrop_ri_measurement_operator.src.utils.solve_epsilon import solve_epsilon_diff_ab


# def imager(param_optimiser: Dict, param_measop: Dict, param_proxop: Dict) -> None:
#     """
#     Imager for small scale RI imaging task.

#     This function prepares the measurement operator, prior, and algorithm for the imaging task.
#     It supports different algorithms such as 'airi', 'usara', and 'cairi'. The function also
#     handles the imaging process if the 'flag_imaging' is set in the 'param_optimiser'.

#     Args:
#         param_optimiser (dict): A dictionary containing the parameters for the optimiser.
#             It includes parameters like 'algorithm', 'im_min_itr', 'im_max_itr', 'im_var_tol',
#             'im_peak_est', 'heu_noise_scale', 'dnn_adaptive_peak', 'dnn_adaptive_peak_tol_min',
#             'dnn_adaptive_peak_tol_max', 'dnn_adaptive_peak_tol_step', 'result_path', 'itr_save',
#             'verbose', and 'flag_imaging'.
#         param_measop (dict): A dictionary containing the parameters for the measurement operator.
#             It includes parameters like 'superresolution', 'im_pixel_size', 'flag_data_weighting',
#             'weight_load', 'img_size', 'weight_type', 'weight_gridsize', 'weight_robustness',
#             'dtype', 'device', 'nufft_grid_size', 'nufft_kb_kernel_dim', and 'nufft_mode'.
#         param_proxop (dict): A dictionary containing the parameters for the proximal operator.
#             It includes parameters like 'dnn_shelf_path', 'dnn_apply_transform', 'device', 'dtype',
#             and 'verbose'.
#     """
#     # initialisation

#     data = load_data_to_tensor(
#         param_optimiser["data_file"],
#         super_resolution=param_measop["superresolution"],
#         image_pixel_size=param_measop["im_pixel_size"],
#         data_weighting=param_measop["flag_data_weighting"],
#         load_weight=param_measop["weight_load"],
#         img_size=param_measop["img_size"],
#         uv_unit="radians",
#         weight_type=param_measop["weight_type"],
#         weight_gridsize=param_measop["weight_gridsize"],
#         weight_robustness=param_measop["weight_robustness"],
#         dtype=param_measop["dtype"],
#         device=param_measop["device"],
#         verbose=param_optimiser["verbose"],
#     )

#     if data["nFreqs"].item() == 1:
#         data["flag"] = data["flag"][:, 0, :].unsqueeze(1)

#     if param_measop["use_ROP"]:
#         assert param_measop["use_BDA"] is False, "BDA cannot be used with ROP"
#         from .mrop_ri_measurement_operator import weighting_correction

#         if param_measop["ROP_param"]["Q"] is None:
#             assert "Q" in data, "number of anntennas Q is not in data and not provided"
#             param_measop["ROP_param"]["Q"] = int(data["Q"])

#         N = int(np.prod(param_measop["img_size"]))
#         K = int(data["nFreqs"].item())
#         V = int(param_measop["ROP_param"]["Q"] * (param_measop["ROP_param"]["Q"] - 1) // 2)
#         B = data["flag"].shape[-1] / V

#         epsilon, P, M_B, M_K = solve_epsilon_diff_ab(N, param_measop["ROP_param"]["Q"], B, K)

#         # epsilon = (N * param_measop["ROP_param"]["N_ratio"] / (V * K * B)) ** (1/3)
#         print(
#             f"INFO: Calculated epsilon for MROP modulation dimensions: {epsilon:.4f} (epsilon = (N / BVK)^(1/3))."
#         )
#         print(f"INFO: Original dimensions: N = {N}, V = {V}, K = {K}, B = {B}.")
#         param_measop["ROP_param"]["M_K"] = M_K
#         param_measop["ROP_param"]["M_B"] = M_B
#         # param_measop["ROP_param"]["P"] = int(np.floor(epsilon * V))
#         param_measop["ROP_param"]["P"] = P
#         param_measop["ROP_param"]["M"] = M_K * M_B
#         print(
#             f"INFO: MROP set with P = {param_measop["ROP_param"]["P"]}, M_K = {param_measop["ROP_param"]["M_K"]}, M_B = {param_measop["ROP_param"]["M_B"]}, M = {param_measop["ROP_param"]["M"]}."
#         )
#         print(
#             f"INFO: PM / N = {param_measop["ROP_param"]["P"] * param_measop["ROP_param"]["M"] / N:.4f}",
#             flush=True,
#         )
#         if param_measop["ROP_param"]["B"] is None:
#             if "flag" in data and data["flag"] is not None and "B" not in data:
#                 data["B"] = data["flag"].shape[-1] / V * K
#             assert "B" in data, "number of snapshots B is not in data and not provided"
#             param_measop["ROP_param"]["B"] = int(data["B"])
#         data, weight_corr = weighting_correction(data, param_measop["ROP_param"])
#         print(
#             f"INFO: Correction has been applied to the weighting for {param_measop['ROP_param']['ROP_type']}",
#             flush=True,
#         )
#     elif param_measop["use_BDA"]:
#         if "flag" in data and data["flag"] is not None and "B" not in data:
#             data["B"] = int(
#                 data["flag"].shape[-1]
#                 / (param_measop["ROP_param"]["Q"] * (param_measop["ROP_param"]["Q"] - 1))
#                 * 2
#                 * data["nFreqs"].item()
#             )
#         if param_measop["ROP_param"]["Q"] is not None and "Q" not in data:
#             data["Q"] = param_measop["ROP_param"]["Q"]
#         assert "B" in data, "number of baselines B is not in data and not provided"
#         assert "Q" in data, "number of anntennas Q is not in data and not provided"

#         from .utils.bda import gen_BDA, symmetrisation

#         # if param_optimiser["nfreqs"] is not None:
#         #     assert param_optimiser["nfreqs"] == 1, "BDA is only implemented for single frequency data"
#         # print("SR=", data["super_resolution"])
#         # W_bda, u_s, v_s, I_s_i, I_s_j, B, Q = gen_BDA(
#         W_bda, u_s, v_s, I_s, B, Q, y, nW = gen_BDA(
#             data_file=param_optimiser["data_file"],
#             # data_path=param_optimiser["data_path"],
#             # freq_num= 1 if param_optimiser["freq_num"] is None else param_optimiser["freq_num"],
#             # vla=True,
#             # data=data,
#             B=data["B"],
#             Q=data["Q"],
#             uv_pt="avg",
#             start_zone=1,
#             start_num=1,
#             # sr_factor=data["super_resolution"],
#             # return_B=False
#         )

#         # I_s = torch.sparse_coo_tensor(
#         #     indices=torch.stack([I_s_i.ravel(), I_s_j.ravel()]),
#         #     values=torch.ones(len(I_s_i.ravel())),
#         #     size=(u_s.numel(), B * Q**2),
#         #     dtype=torch.complex128,
#         #     is_coalesced=True,
#         # )

#         flag = np.concatenate(
#             [
#                 data["flag"].numpy(force=True)[0, iFreq, :]
#                 for iFreq in range(data["flag"].numpy(force=True).shape[1])
#             ]
#         )

#         y = symmetrisation(
#             y,
#             data["B"],
#             data["Q"],
#             flag,
#         )
#         nW = symmetrisation(
#             nW,
#             data["B"],
#             data["Q"],
#             flag,
#         )

#         data["y"] = (I_s @ y.to(I_s.device).ravel()) / W_bda
#         print(f"INFO: BDA averaged size: {W_bda.numel()}", flush=True)

#         data["y"] = data["y"].view(1, 1, -1)
#         data["nW"] = (I_s @ nW.to(device=I_s.device, dtype=I_s.dtype).ravel()) / W_bda
#         # data["nW"] = (data["nW"].to(data["y"].device) * torch.ones_like(data["y"])) * torch.sqrt(W_bda)
#         data["u"] = u_s.to(device=I_s.device).view(1, 1, -1)
#         data["v"] = v_s.to(device=I_s.device).view(1, 1, -1)
#         data["y"] *= data["nW"].view(1, 1, -1)
#         # data["nW"] = data["nW"].view(1, 1, -1)

#         if param_measop["flag_data_weighting"]:
#             from .ri_measurement_operator.pysrc.utils.gen_imaging_weights import gen_imaging_weights

#             data["nWimag"] = gen_imaging_weights(
#                 data["u"].clone(),
#                 data["v"].clone(),
#                 data["nW"].clone(),
#                 param_measop["img_size"],
#                 weight_type=param_measop["weight_type"],
#                 weight_gridsize=param_measop["weight_gridsize"],
#                 weight_robustness=param_measop["weight_robustness"],
#             ).view(1, 1, -1)
#         else:
#             data["nWimag"] = torch.tensor(
#                 [1.0], dtype=param_measop["dtype"], device=param_measop["device"]
#             ).view(1, 1, -1)

#         data["y"] *= data["nWimag"]

#         for k in ["y", "u", "v", "nW", "nWimag"]:
#             data[k] = data[k].to(param_measop["device"]).view(1, 1, -1)
#         print("INFO: BDA applied in measurement operator and data", flush=True)

#     meas_op = None

#     if param_measop["nufft_package"] == "pynufft":
#         from .ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pynufft import (
#             MeasOpPynufft,
#         )

#         if not param_measop["use_ROP"]:
#             nufft_op = MeasOpPynufft
#         else:
#             from .mrop_ri_measurement_operator import create_meas_op_ROP

#             nufft_op = create_meas_op_ROP(MeasOpPynufft)

#         meas_op = nufft_op(
#             ROP_param=param_measop["ROP_param"],
#             u=data["u"],
#             v=data["v"],
#             flag=data["flag"].numpy(force=True),
#             img_size=param_measop["img_size"],
#             natural_weight=data["nW"],
#             image_weight=data["nWimag"],
#             grid_size=param_measop["nufft_grid_size"],
#             kernel_dim=param_measop["nufft_kb_kernel_dim"],
#             device=param_measop["device"],
#             dtype=param_measop["dtype"],
#         )

#     elif param_measop["nufft_package"] == "tkbnufft":
#         from .ri_measurement_operator.pysrc.measOperator.meas_op_nufft_tkbn import (
#             MeasOpTkbNUFFT,
#         )

#         if not param_measop["use_ROP"]:
#             nufft_op = MeasOpTkbNUFFT
#         else:
#             from .mrop_ri_measurement_operator import create_meas_op_ROP

#             nufft_op = create_meas_op_ROP(MeasOpTkbNUFFT)

#         meas_op = nufft_op(
#             ROP_param=param_measop["ROP_param"],
#             u=data["u"],
#             v=data["v"],
#             flag=data["flag"].numpy(force=True),
#             img_size=param_measop["img_size"],
#             natural_weight=data["nW"],
#             image_weight=data["nWimag"],
#             grid_size=param_measop["nufft_grid_size"],
#             kernel_dim=param_measop["nufft_kb_kernel_dim"],
#             mode=param_measop["nufft_mode"],
#             device=param_measop["device"],
#             dtype=param_measop["dtype"],
#         )

#     else:
#         from .ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import (
#             MeasOpPytorchFinufft,
#         )

#         if not param_measop["use_ROP"]:
#             nufft_op = MeasOpPytorchFinufft
#         else:
#             # if param_measop["ROP_param"]["ROP_batchwise"]:
#             #     if param_optimiser.get("nfreqs", data["nFreqs"]) in [None, 1]:
#             #         from .mrop_ri_measurement_operator import create_meas_op_ROP_batchwise as create_meas_op_ROP
#             #     else:
#             #         from .mrop_ri_measurement_operator import create_meas_op_ROP_batchwise_mf as create_meas_op_ROP
#             # elif param_measop["ROP_param"]["ROP_vmap"]:
#             #     if param_optimiser.get("nfreqs", data["nFreqs"]) in [None, 1]:
#             #     # if param_optimiser["nfreqs"] is None or data["nfreqs"] == 1:
#             #     # if param_optimiser["nfreqs"] is None or param_optimiser["nfreqs"] == 1:
#             #         from .mrop_ri_measurement_operator import create_meas_op_ROP_vmap as create_meas_op_ROP

#             #         print("INFO: Using vmap ROP for single frequency data", flush=True)
#             #     else:
#             #         if param_measop["ROP_param"]["freq_mod"]:
#             #             from .mrop_ri_measurement_operator import create_meas_op_ROP_vmap_mf_bf_mod as create_meas_op_ROP
#             #             print("INFO: Using vmap ROP for multi-frequency data, treating frequency dimension as batches", flush=True)
#             #         else:
#             #             from .mrop_ri_measurement_operator import create_meas_op_ROP_vmap_mf as create_meas_op_ROP
#             #             print("INFO: Using vmap ROP for multi-frequency data", flush=True)
#             # else:
#             #     from .mrop_ri_measurement_operator import create_meas_op_ROP
#             from .mrop_ri_measurement_operator.src.mrop_vmap_mf_mod_KB import (
#                 create_meas_op_ROP_vmap_mod_KB as create_meas_op_ROP,
#             )

#             nufft_op = create_meas_op_ROP(MeasOpPytorchFinufft)

#         meas_op = nufft_op(
#             ROP_param=param_measop["ROP_param"],
#             u=data["u"],
#             v=data["v"],
#             num_chs=data["nFreqs"].item(),
#             flag=data["flag"].numpy(force=True),
#             img_size=param_measop["img_size"],
#             natural_weight=data["nW"],
#             image_weight=data["nWimag"],
#             device=param_measop["device"],
#             dtype=param_measop["dtype"],
#         )

#     if param_measop["use_ROP"]:
#         print(
#             f"INFO: data size before {param_measop['ROP_param']['ROP_type']} is {data['y'].numel()}",
#             flush=True,
#         )
#         if param_measop["ROP_param"]["ROP_type"] == "MROP":
#             data["y"] = meas_op.MD(data["y"] * weight_corr)
#         elif param_measop["ROP_param"]["ROP_type"] == "CROP":
#             data["y"] = meas_op.D(data["y"] * weight_corr)
#         print(
#             f"INFO: data size after {param_measop['ROP_param']['ROP_type']} is {data['y'].numel()}",
#             flush=True,
#         )

#     meas_op_approx = None
#     if param_optimiser["approx_meas_op"]:
#         from .ri_measurement_operator.pysrc.measOperator.meas_op_PSF import MeasOpPSF

#         meas_op_approx = MeasOpPSF(
#             data["u"],
#             data["v"],
#             param_measop["img_size"],
#             natural_weight=data["nW"],
#             image_weight=data["nWimag"],
#             real_flag=True,
#             normalise_psf=False,
#             device=param_measop["device"],
#             dtype=param_measop["dtype"],
#         )

#     optimiser = None
#     if param_optimiser["algorithm"] == "airi":
#         prox_op_airi = ProxOpAIRI(
#             param_proxop["dnn_shelf_path"],
#             rand_trans=param_proxop["dnn_apply_transform"],
#             device=param_proxop["device"],
#             dtype=param_proxop["dtype"],
#             verbose=param_proxop["verbose"],
#         )

#         optimiser = FBAIRI(
#             data["y"],
#             meas_op,
#             prox_op_airi,
#             meas_op_approx=meas_op_approx,
#             im_min_itr=param_optimiser["im_min_itr"],
#             im_max_itr=param_optimiser["im_max_itr"],
#             im_var_tol=param_optimiser["im_var_tol"],
#             im_peak_est=param_optimiser["im_peak_est"],
#             heu_noise_scale=param_optimiser["heu_noise_scale"],
#             new_heu=param_optimiser["new_heu"],
#             adapt_net_select=param_optimiser["dnn_adaptive_peak"],
#             peak_tol_min=param_optimiser["dnn_adaptive_peak_tol_min"],
#             peak_tol_max=param_optimiser["dnn_adaptive_peak_tol_max"],
#             peak_tol_step=param_optimiser["dnn_adaptive_peak_tol_step"],
#             save_pth=param_optimiser["result_path"],
#             file_prefix=param_optimiser["file_prefix"],
#             iter_save=param_optimiser["itr_save"],
#             verbose=param_optimiser["verbose"],
#         )

#     elif param_optimiser["algorithm"] == "cairi":
#         prox_op_airi = ProxOpAIRI(
#             param_proxop["dnn_shelf_path"],
#             rand_trans=param_proxop["dnn_apply_transform"],
#             device=param_proxop["device"],
#             dtype=param_proxop["dtype"],
#             verbose=param_proxop["verbose"],
#         )

#         # preconditioning weight
#         if param_optimiser["precond_flag"]:
#             precond_weight = (
#                 torch.from_numpy(
#                     gen_imaging_weight(
#                         data["u"].cpu().numpy(),
#                         data["v"].cpu().numpy(),
#                         param_measop["img_size"],
#                         weight_type="uniform",
#                         grid_size=2,
#                     ).reshape(1, 1, -1)
#                 )
#                 ** 2
#             )
#         else:
#             precond_weight = torch.ones(1, 1)

#         # Theoretical l2 error bound, assume chi-square distribution, tau=1
#         l2_bound = np.sqrt(torch.numel(data["y"]) + 2.0 * np.sqrt(torch.numel(data["y"])))
#         if param_optimiser["verbose"]:
#             print(
#                 "INFO: The theoretical l2 error bound is",
#                 f"{l2_bound}",
#             )

#         prox_op_dual_data = ProxOpElipse(
#             center=data["y"],
#             precond_weight=precond_weight,
#             radius=l2_bound,
#             device=meas_op.get_device(),
#             dtype=meas_op.get_data_type_meas(),
#         )

#         optimiser = PDAIRI(
#             data["y"],
#             meas_op,
#             prox_op_airi,
#             prox_op_dual_data,
#             im_min_itr=param_optimiser["im_min_itr"],
#             im_max_itr=param_optimiser["im_max_itr"],
#             im_var_tol=param_optimiser["im_var_tol"],
#             im_peak_est=param_optimiser["im_peak_est"],
#             heu_noise_scale=param_optimiser["heu_noise_scale"],
#             adapt_net_select=param_optimiser["dnn_adaptive_peak"],
#             peak_tol_min=param_optimiser["dnn_adaptive_peak_tol_min"],
#             peak_tol_max=param_optimiser["dnn_adaptive_peak_tol_max"],
#             peak_tol_step=param_optimiser["dnn_adaptive_peak_tol_step"],
#             save_pth=param_optimiser["result_path"],
#             file_prefix=param_optimiser["file_prefix"],
#             iter_save=param_optimiser["itr_save"],
#             verbose=param_optimiser["verbose"],
#         )

#     elif param_optimiser["algorithm"] == "usara":
#         prox_op_sara = ProxOpSARAPos(
#             param_measop["img_size"],
#             device=param_proxop["device"],
#             dtype=param_proxop["dtype"],
#             verbose=param_proxop["verbose"],
#         )

#         optimiser = FBSARA(
#             data["y"],
#             meas_op,
#             prox_op_sara,
#             use_ROP=param_measop["use_ROP"],
#             meas_op_approx=meas_op_approx,
#             im_min_itr=param_optimiser["im_min_itr"],
#             im_max_itr=param_optimiser["im_max_itr"],
#             im_var_tol=param_optimiser["im_var_tol"],
#             heu_reg_scale=param_optimiser["heu_reg_param_scale"],
#             new_heu=param_optimiser["new_heu"],
#             im_max_itr_outer=param_optimiser["im_max_outer_itr"],
#             im_var_tol_outer=param_optimiser["im_var_outer_tol"],
#             save_pth=param_optimiser["result_path"],
#             file_prefix=param_optimiser["file_prefix"],
#             reweight_save=param_optimiser["reweighting_save"],
#             verbose=param_optimiser["verbose"],
#         )

#     # imaging
#     if param_optimiser["flag_imaging"]:
#         # initialisation
#         optimiser.initialisation()
#         # run imaging loop
#         optimiser.run()
#         # finalisation
#         optimiser.finalisation()

#         # calculate final metrics
#         if param_optimiser["verbose"]:
#             img_model = optimiser.get_model_image()
#             img_residual = optimiser.get_residual_image()
#             img_dirty = optimiser.get_dirty_image()
#             psf = optimiser.get_psf()

#             img_residual_std = np.std(img_residual).item()
#             img_residual_std_noramalised = img_residual_std / psf.max().item()
#             img_residual_ratio = np.linalg.norm(img_residual.flatten()) / np.linalg.norm(img_dirty.flatten())
#             print(
#                 "INFO: The standard deviation of the final",
#                 f"residual dirty image is {img_residual_std}",
#             )
#             print(
#                 "INFO: The standard deviation of the normalised",
#                 f"final residual dirty image is {img_residual_std_noramalised}",
#             )
#             print(
#                 "INFO: The ratio between the norm of the residual",
#                 f"and the dirty image: ||residual|| / || dirty || = {img_residual_ratio}",
#             )

#             if param_optimiser["groundtruth"]:
#                 img_gdth = fits.getdata(param_optimiser["groundtruth"]).astype(np.double)
#                 rsnr = 20 * np.log10(
#                     np.linalg.norm(img_gdth.flatten())
#                     / np.linalg.norm(img_gdth.flatten() - img_model.flatten())
#                 )
#                 print(
#                     "INFO: The signal-to-noise ratio of the final",
#                     f"reconstructed image is {rsnr} dB",
#                 )
