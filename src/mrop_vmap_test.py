from mrop_ri_measurement_operator.mrop import create_meas_op_ROP
from mrop_ri_measurement_operator.mrop_vmap import create_meas_op_ROP_vmap
from ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft
from utils.io import load_data_to_tensor
import torch
import argparse
from astropy.io import fits
import numpy as np
import timeit

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
        freq_num = 1,
        vis_remove = 17.7,
        use_ROP = True,
        dtype = torch.float64,
    )

    P = 10
    M = 100
    B = data["B"]
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
        "ROP_batch_step": data["B"]
    }
    
    op_tmp = create_meas_op_ROP_vmap(MeasOpPytorchFinufft)
    mrop_meas_op = op_tmp(
        ROP_param=ROP_param,
        u=data["u"],
        v=data["v"],
        ant1=data["ant1"],
        ant2=data["ant2"],
        batches=data["batches"],
        img_size=(1024, 1024),
        natural_weight=data["nW"],
        grid_size=(2048, 2048),
        dtype=torch.float64,
    )
    
    op_tmp2 = create_meas_op_ROP(MeasOpPytorchFinufft)
    mrop_meas_op2 = op_tmp2(
        ROP_param=ROP_param,
        u=data["u"],
        v=data["v"],
        ant1=data["ant1"],
        ant2=data["ant2"],
        batches=data["batches"],
        img_size=(1024, 1024),
        natural_weight=data["nW"],
        grid_size=(2048, 2048),
        dtype=torch.float64,
    )
    print(
        "MD:", 
        torch.allclose(
            mrop_meas_op2.MD(data["y"] * data["nW"]),
            mrop_meas_op.MD(data["y"] * data["nW"])
        )
    )
    
    dirty2 = mrop_meas_op2.adjoint_op(mrop_meas_op2.MD(data["y"] * data["nW"]))
    dirty = mrop_meas_op.adjoint_op(mrop_meas_op.MD(data["y"] * data["nW"]))
    
    # y_full = mrop_meas_op.symmetrisation(data["y"] * data["nW"])

    # beta = mrop_meas_op.beta
    # alpha = mrop_meas_op.alpha
    # gamma = mrop_meas_op.gamma
    # beta2 = mrop_meas_op2.beta
    # alpha2 = mrop_meas_op2.alpha
    # gamma2 = mrop_meas_op2.gamma
    
    # # ROP
    # y_rop = []
    # for i in range(y_full.shape[-1]):
    #     y_rop.append(mrop_meas_op.ROP(y_full[..., i], alpha2[..., i], beta2[..., i]))
    # y_rop = torch.stack(y_rop, dim=0)
    
    # y_rop2 = []
    # for i in range(y_full.shape[-1]):
    #     y_rop2.append(mrop_meas_op2.ROP(y_full[..., i], alpha2[..., i], beta2[..., i]))
    # y_rop2 = torch.stack(y_rop2, dim=0)
    
    # print(f"ROP: {torch.allclose(y_rop, y_rop2)}")
    
    # # CROP
    # y_crop = mrop_meas_op.CROP_vmap(y_full, alpha, beta)
    # y_crop2 = mrop_meas_op2.CROP(y_full).T
    
    # print(f"CROP: {torch.allclose(y_crop, y_crop2)}")
    
    # # Modulation
    
    # print(y_crop.shape, y_crop2.shape)
    # y_mrop = torch.mm(gamma, y_crop.view(B, P)) / np.sqrt(M)
    # y_mrop_2 = mrop_meas_op.MROP(y_full)
    
    # print(y_mrop.shape, y_mrop_2.shape)
    # print(f"MROP vmap: {torch.allclose(y_mrop.ravel(), y_mrop_2)}")
    
    # y_mrop2 = torch.mm(gamma2, y_crop2.T.reshape(P, B).T) / np.sqrt(M)
    # y_mrop2_2 = mrop_meas_op2.MROP(y_full)
    # print(f"MROP: {torch.allclose(y_mrop2.ravel(), y_mrop2_2.ravel())}")
    
    # print(f"MROP: {torch.allclose(y_mrop_2.ravel(), y_mrop2_2.T.ravel())}")

    # dirty = mrop_meas_op.adjoint_op(y_mrop)
    # dirty2 = mrop_meas_op2.adjoint_op(y_mrop2)
    
    print(dirty.shape, dirty2.shape)
    print(torch.allclose(dirty, dirty2))
    
    fits.writeto(
        "/work/ec110/ec110/cc2040/jobs/python/mrop/vmap/dirty2_17.7.fits",
        dirty2.squeeze().cpu().numpy().astype(np.float32),
        overwrite=True
    )
    fits.writeto(
        "/work/ec110/ec110/cc2040/jobs/python/mrop/vmap/dirty_17.7.fits",
        dirty.squeeze().cpu().numpy().astype(np.float32),
        overwrite=True
    )
    
    # dirac = torch.zeros((1, 1, *img_size), dtype=torch.float64)
    # dirac[0, 0, img_size[0] // 2, img_size[1] // 2] = 1.0
    
    # ts = np.zeros(100)
    # for i in range(100):
    #     start = timeit.default_timer()
    #     dirty = mrop_meas_op.adjoint_op(mrop_meas_op.forward_op(dirac))
    #     ts[i] = timeit.default_timer() - start
        
    # print(np.mean(ts), np.std(ts))
    
    # vmap_ts = np.zeros(100)
    # for i in range(100):
    #     start = timeit.default_timer()
    #     dirty2 = mrop_meas_op2.adjoint_op(mrop_meas_op2.forward_op(dirac))
    #     vmap_ts[i] = timeit.default_timer() - start
    
    # print(np.mean(vmap_ts), np.std(vmap_ts))   
    
    
    print("TADA!")
