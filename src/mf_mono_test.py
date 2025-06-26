from mrop_ri_measurement_operator.mrop_mf_mono import create_meas_op_ROP_mf_mono
from ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft
from utils.io import load_data_to_tensor
import torch
import argparse
from astropy.io import fits
import numpy as np

def test_adjoint_op(meas_op, img_size, M, seed=1377):
    np.random.seed(seed)
    x = torch.randn(1, 1, *img_size, dtype=torch.float64)
    y = (
        torch.randn(1, 1, M, dtype=torch.complex128) + 1j * torch.randn(1, 1, M, dtype=torch.complex128)
    ) / np.sqrt(2.0)
    
    if "finufft" in str(meas_op.__class__.__name__).lower():
        p1 = torch.sum(y.conj() * meas_op.forward_op(x))
        p2 = torch.sum((meas_op.adjoint_op(y) * x))
    else:
        p1 = torch.sum(y.conj() * meas_op.forward_op(x))
        p2 = torch.sum((meas_op.adjoint_op(y).conj() * x))

    rel_diff = abs(p1 - p2) / abs(p1)
    if rel_diff < 1e-10:
        print("Adjoint operator test passed")
    else:
        print("Adjoint operator test failed")
        print(f"Relative difference: {rel_diff}")

if __name__ == "__main__":
    args = argparse.Namespace()
    args.data_path = "/scratch/space1/ec110/cc2040/273-X08/"
    args.data_file = "/scratch/space1/ec110/cc2040/273-X08/msSpecs.mat"
    args.super_resolution = 1.87

    data = load_data_to_tensor(
        main_data_file = args.data_file,
        img_size = (1024, 1024),
        data_path = args.data_path,
        super_resolution = args.super_resolution,
        data_weighting = False,
        weight_type = "natural",
        weight_gridsize = 2.0,
        use_ROP = True,
        dtype = torch.float64,
    )

    P = 10
    M = 100
    img_size = (1024, 1024)
    grid_size = tuple(i * 2 for i in img_size)

    ROP_param = {
        "Q": data["Q"],
        "P": P,
        "M": M,
        "B": data["B"],
        "rv_type": "unitary",
        "ROP_seed": 1337,
        "ROP_type": "MROP",
        "ROP_batchwise": False,
        "ROP_batch_step": 5000
    }
    
    op_tmp = create_meas_op_ROP_mf_mono(MeasOpPytorchFinufft)
    mrop_meas_op = op_tmp(
        ROP_param=ROP_param,
        u=data["u"],
        v=data["v"],
        num_chs=64,
        ant1=data["ant1"],
        ant2=data["ant2"],
        batches=data["batches"],
        img_size=(1024, 1024),
        natural_weight=data["nW"],
        grid_size=(2048, 2048),
        dtype=torch.float64,
    )
    
    # y_full = mrop_meas_op.symmetrisation(data["y"])
    # y_mrop = torch.sum(mrop_meas_op.MROP_vmap(y_full, mrop_meas_op.alpha, mrop_meas_op.beta, mrop_meas_op.gamma), dim=0)
    
    # y_mropt = mrop_meas_op.MROPt(y_mrop, mrop_meas_op.alpha, mrop_meas_op.beta, mrop_meas_op.gamma)
    # y_mropt2 = mrop_meas_op.upper_triangularisation(y_mropt)
    # print(y_mropt2.shape)
    # dirty = mrop_meas_op._AtGt(y_mropt2).squeeze().real
    # dirty_full = mrop_meas_op._AtGt(data["y"]).squeeze().real
    # print(dirty_full.shape)
    # print(dirty.shape)
    # fits.writeto("/work/ec110/ec110/cc2040/jobs/python/mrop/mf_mono/dirty_full.fits", dirty_full.cpu().numpy(), overwrite=True)
    # fits.writeto("/work/ec110/ec110/cc2040/jobs/python/mrop/mf_mono/dirty.fits", dirty.cpu().numpy(), overwrite=True)
    test_adjoint_op(mrop_meas_op, (1024, 1024), int(P * M))
    print("TADA!")
    
    
    # print(y_mropt.shape)
    # print(y_mropt.shape[0] * y_mropt.shape[1], data["B"])
    
    # y_mropt2 = y_mropt.reshape(data["B"], data["Q"], data["Q"])
    # print(torch.allclose(y_mropt2[0], y_mropt[0, 0]))
    # print(torch.allclose(y_mropt2[1], y_mropt[0, 1]))
    # print(torch.allclose(y_mropt2[1], y_mropt[1, 0]))