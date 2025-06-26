import gc
import os
from typing import Dict

import numpy as np
import torch
from astropy.io import fits
from scipy.io import loadmat, savemat

from .utils import gen_noise, gen_noise


# from .utils.io import load_data_to_tensor

from .ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft
from .ri_measurement_operator.pysrc.utils.gen_imaging_weights import gen_imaging_weights
from .ri_measurement_operator.pysrc.utils.io import load_data_to_tensor


def data_sim(
    param_optimiser: Dict, param_measop: Dict, sigma0: float, sigma_range_min: float, sigma_range_max: float
) -> None:
    # data = load_mat_data_file_2_tensor_ri_rop(
    #     file_path=param_optimiser["data_file"],
    #     gdth_path=param_optimiser["groundtruth"],
    #     ROP_type=None,
    #     use_BDA=False,
    #     subsampling=False,
    #     sim=True,
    #     sr_factor=param_measop["superresolution"],
    #     im_pixel_size=param_measop["im_pixel_size"],
    #     img_size=param_measop["img_size"],
    #     dtype=param_measop["dtype"],
    #     device=param_measop["device"],
    #     verbose=param_optimiser["verbose"],
    # )
    data = load_data_to_tensor(
        param_optimiser["data_file"],
        super_resolution=param_measop["superresolution"],
        image_pixel_size=param_measop["im_pixel_size"],
        data_weighting=False,
        load_weight=False,
        img_size=param_measop["img_size"],
        uv_unit="radians",
        weight_type=param_measop["weight_type"],
        weight_gridsize=param_measop["weight_gridsize"],
        weight_robustness=param_measop["weight_robustness"],
        dtype=param_measop["dtype"],
        device=param_measop["device"],
        verbose=param_optimiser["verbose"],
    )

    meas_op_raw = MeasOpPytorchFinufft(
        u=data["u"],
        v=data["v"],
        img_size=param_measop["img_size"],
        natural_weight=data["nW"],
        image_weight=data["nWimag"],
        device=param_measop["device"],
        dtype=param_measop["dtype"],
    )
    seed = 2103
    # seed = int(param_optimiser["data_file"].split("/")[-1].split("_")[-1].split(".mat")[0])
    # seed = int(param_optimiser["data_file"].split("/")[-1].split("_id_")[-1].split("_")[0])
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    data["uv_id"] = seed
    data["gdth"] = torch.tensor(
        fits.getdata(param_optimiser["groundtruth"]).astype(np.float32),
        dtype=param_measop["dtype"],
        device=param_measop["device"],
    ).view(1, 1, *param_measop["img_size"])
    data, tau_0, sigma = gen_noise(
        meas_op_raw,
        data,
        sigma0,
        sigma_range_min,
        sigma_range_max,
        device=param_measop["device"],
        verbose=param_optimiser["verbose"],
    )

    print(f"Target dynamic range: {1/sigma.item():.4e}")

    fits.writeto(
        os.path.join(
            param_optimiser["result_path"],
            "gdth_expo.fits",
        ),
        data["gdth"].numpy(force=True),
        overwrite=True,
    )

    data["nW"] = torch.tensor([1 / tau_0], dtype=param_measop["dtype"], device=param_measop["device"]).view(
        1, 1, -1
    )

    if param_measop["weight_type"] not in ["natural"]:
        data["nWimag"] = gen_imaging_weights(
            data["u"].clone(),
            data["v"].clone(),
            im_size=param_measop["img_size"],
            nW=data["nW"],
            weight_type=param_measop["weight_type"],
            weight_gridsize=param_measop["weight_gridsize"],
            weight_robustness=param_measop["weight_robustness"],
        )

        meas_op_0 = MeasOpPytorchFinufft(
            u=data["u"],
            v=data["v"],
            natural_weight=data["nW"],
            image_weight=data["nWimag"],
            img_size=param_measop["img_size"],
            device=param_measop["device"],
            dtype=param_measop["dtype"],
        )

        measop_norm_1 = meas_op_0.get_op_norm()
        measop_norm_2 = meas_op_0.get_op_norm_prime()
        eta_correction = np.sqrt(measop_norm_2 / measop_norm_1)
        tau = tau_0 * np.sqrt(2 * measop_norm_1) * sigma / eta_correction

        del meas_op_0

    else:

        data["nWimag"] = torch.ones_like(data["nW"])
        tau = tau_0

    print(f"tau: {tau:.4e}")

    data["nW"] = torch.tensor([1 / tau], dtype=param_measop["dtype"], device=param_measop["device"]).view(
        1, 1, -1
    )

    data["y"] = meas_op_raw.forward_op(data["gdth"])

    del meas_op_raw
    gc.collect()

    noise = tau * (torch.randn_like(data["y"]) + 1j * torch.randn_like(data["y"])) / np.sqrt(2.0)

    data["y"] += noise.to(data["y"].device)

    data_save = loadmat(param_optimiser["data_file"])
    data_save.update(
        {"y": data["y"].cpu().numpy(), "nW": data["nW"].cpu().numpy(), "tau": tau, "sigma": sigma}
    )

    dr = 1 / sigma

    save_mat_path = os.path.join(
        param_optimiser["result_path"],
        f"data_sim_id_{seed}_DR_{dr:.2e}.mat",
    )

    savemat(save_mat_path, data_save)
