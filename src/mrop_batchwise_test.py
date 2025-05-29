from mrop_ri_measurement_operator.meas_op_nufft_mrop_batchwise import create_meas_op_ROP_batchwise
from mrop_ri_measurement_operator.meas_op_nufft_mrop import create_meas_op_ROP as create_meas_op_ROP
from ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft
from ri_measurement_operator.pysrc.utils.gen_imaging_weights import gen_imaging_weights

import argparse
from scipy.constants import speed_of_light
from scipy.io import loadmat
from scipy.io.matlab import matfile_version
import h5py
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import gc

if __name__ == "__main__":
    args = argparse.Namespace()
    args.data_path = "/users/cc2040/sharedscratch/3c273/273-X08/"
    args.data_file = "/users/cc2040/sharedscratch/3c273/273-X08/msSpecs.mat"
    args.super_resolution = 1.87
    
    Q = 27
    i_f = 0
    data_tmp = loadmat(
        os.path.join(args.data_path, f"273-X08_data_ch_{i_f+1}.mat"), 
        variable_names=["data_I", "flag", "batches_flagged", "ant1_flagged", "ant2_flagged", "weightsNat"]
    )
    uv_data = loadmat(args.data_file)
    y = torch.tensor(data_tmp["data_I"].squeeze(), dtype=torch.complex128).view(1, 1, -1)
    ant1 = data_tmp["ant1_flagged"].squeeze().astype(int)
    ant2 = data_tmp["ant2_flagged"].squeeze().astype(int)
    batches = data_tmp["batches_flagged"].squeeze().astype(int) # + B * i_f
    B = np.unique(data_tmp["batches_flagged"]).size
    P = 10
    M = 100
    img_size = (1024, 1024)
    grid_size = tuple(i * 2 for i in img_size)

    ROP_param = {
        "Q": Q,
        "P": P,
        "M": M,
        "B": B,
        "rv_type": "unitary",
        "ROP_seed": 1337,
        "ROP_type": "MROP",
        "ROP_batch_step": 250
    }

    u = uv_data["uvw"][:, 0][data_tmp["flag"].squeeze() == 1] / (speed_of_light / uv_data["freqs"].squeeze()[i_f])
    v = uv_data["uvw"][:, 1][data_tmp["flag"].squeeze() == 1] / (speed_of_light / uv_data["freqs"].squeeze()[i_f])
    nW = data_tmp["weightsNat"].squeeze()

    max_proj_baseline = np.max(np.sqrt(u ** 2 + v ** 2))
    spatial_bandwidth = 2 * max_proj_baseline
    image_pixel_size = (180.0 / np.pi) * 3600.0 / (args.super_resolution * spatial_bandwidth)
    u = torch.tensor(u, dtype=torch.float64).view(1, 1, -1)
    v = -torch.tensor(v, dtype=torch.float64).view(1, 1, -1)
    nW = torch.tensor(nW, dtype=torch.complex128).view(1, 1, -1)
    halfSpatialBandwidth = (180.0 / np.pi) * 3600.0 / (image_pixel_size) / 2.0

    u = u * np.pi / halfSpatialBandwidth
    v = v * np.pi / halfSpatialBandwidth


    op1 = create_meas_op_ROP(MeasOpPytorchFinufft)
    mrop_op1 = op1(
        ROP_param=ROP_param,
        u=u,
        v=v,
        ant1=ant1,
        ant2=ant2,
        batches=batches,
        img_size=img_size,
        real_flag=True,
        natural_weight=nW,
        grid_size=grid_size,
        dtype=torch.float64
    )
    
    op2 = create_meas_op_ROP_batchwise(MeasOpPytorchFinufft)
    mrop_op2 = op2(
        ROP_param=ROP_param,
        u=u,
        v=v,
        ant1=ant1,
        ant2=ant2,
        batches=batches,
        img_size=img_size,
        real_flat=True,
        natural_weight=nW,
        grid_size=grid_size,
        dtype=torch.float64
    )
    
    out_mrop = mrop_op1.MD(y)
    out_mrop2 = mrop_op2.MD(y)
    print(torch.allclose(out_mrop, out_mrop2))
    
    out_mropt = mrop_op1.MDt(out_mrop)
    out_mropt2 = mrop_op2.MDt(out_mrop2)
    print(torch.allclose(out_mropt, out_mropt2))