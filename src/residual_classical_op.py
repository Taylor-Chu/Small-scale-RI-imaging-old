import torch
from typing import Dict
import numpy as np
from astropy.io import fits
from .utils.io import load_data_to_tensor
from .ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import (
    MeasOpPytorchFinufft,
)


def residual_classical_op(
    param_optimiser: Dict, param_measop: Dict, param_proxop: Dict, rec_file: str
) -> None:
    data = load_data_to_tensor(
        param_optimiser["data_file"],
        img_size=param_measop["img_size"],
        data_path=param_optimiser["data_path"],
        super_resolution=param_measop["superresolution"],
        freq_num=param_optimiser["freq_num"],
        vis_remove=param_optimiser["vis_remove"],
        data_weighting=param_measop["flag_data_weighting"],
        image_pixel_size=param_measop["im_pixel_size"],
        load_weight=param_measop["weight_load"],
        uv_unit="radians",
        weight_type=param_measop["weight_type"],
        weight_gridsize=param_measop["weight_gridsize"],
        weight_robustness=param_measop["weight_robustness"],
        use_ROP=param_measop["use_ROP"],
        dtype=param_measop["dtype"],
        device=param_measop["device"],
        verbose=param_optimiser["verbose"],
    )

    rec = torch.tensor(
        fits.getdata(rec_file).astype(np.float64), dtype=param_measop["dtype"], device=param_measop["device"]
    )

    nufft_op = MeasOpPytorchFinufft

    meas_op = nufft_op(
        ROP_param=param_measop["ROP_param"],
        u=data["u"],
        v=data["v"],
        ant1=data["ant1"],
        ant2=data["ant2"],
        batches=data["batches"],
        img_size=param_measop["img_size"],
        natural_weight=data["nW"],
        image_weight=data["nWimag"],
        device=param_measop["device"],
        dtype=param_measop["dtype"],
    )

    data["y"] = data["y"] * data["nW"] * data["nWimag"]

    dirty = meas_op.adjoint_op(data["y"])
    psf = meas_op.get_psf()
    psf_peak = psf.max().item()
    rec = rec.view(*dirty.shape)
    res = (dirty - meas_op.adjoint_op(meas_op.forward_op(rec))) / psf_peak

    fname = rec_file.split("/")[-1].replace(".fits", "_classical_op.fits")
    # print(fname)
    # print(param_optimiser['file_prefix'])
    fits.writeto(
        f"{param_optimiser['result_path']}/{fname}",
        res.cpu().numpy(),
        overwrite=True,
    )
