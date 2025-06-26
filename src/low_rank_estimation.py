import glob
import os
from scipy.io import loadmat, savemat
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
# from kneed import KneeLocator

from ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft

from mrop_ri_measurement_operator.mrop import create_meas_op_ROP

def get_singular_values(mat_file_path):
    data = loadmat(mat_file_path)
    data["y"] = torch.tensor(data["y"], dtype=torch.complex128)
    ROP_param = {
        "P": 80,
        "M": 800,
        "Q": data["na"],
        "B": data["nTimeSamples"],
        "rv_type": "unitary",
        "ROP_type": "MROP",
        "ROP_seed": 1337,
        "ROP_batch_step": data["nTimeSamples"]
    }
    B = data["nTimeSamples"].item()
    Q = data["na"].item()
    y_triu = data["y"].view(B, Q * (Q - 1) // 2)
    y = torch.zeros(B, Q, Q, dtype=y_triu.dtype, device=y_triu.device)
    triu_r, triu_c = torch.triu_indices(Q, Q, offset=1)

    y[:, triu_r, triu_c] = y_triu
    y[:, triu_c, triu_r] = y_triu.conj()
    y = y.view(B, Q * Q)
    _, S, _ = torch.linalg.svd(y)
    return S.cpu().numpy()

if __name__ == "__main__":
    data_paths = glob.glob("/work/ec110/ec110/cc2040/jobs/python/airi_usara/mrop/natural_weighting/data/uniform_meerkat/*")
    Ss = []
    for data_path in tqdm(data_paths):
        if data_path.endswith(".log"):
            continue
        mat_file_path = glob.glob(os.path.join(data_path, "*.mat"))[0]
        Ss.append(get_singular_values(mat_file_path))
        # break

    rank = []
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for i, S in enumerate(Ss):
        ax.plot(S, label=f"Run {i+1}")
        energy_threshold = np.linalg.norm(S) * 0.01
        ax.axhline(energy_threshold, linestyle='--', alpha=0.5)
        rank.append(len(S[S > energy_threshold]))
        # delta = np.diff(S)
        # threshold = -0.1
        # elbow_index = np.argmax(delta > threshold) + 1
        # print(f"Run {i+1} elbow index: {elbow_index}")
        # ax.axvline(elbow_index, linestyle='--', color='gray', alpha=0.5)
    print(f"Rank: {rank}")
    print(f"Average rank: {np.mean(rank)}")
    print(f"Standard deviation of rank: {np.std(rank)}")
    print(f"Median rank: {np.median(rank)}")
    ax.set_yscale("log")
    ax.set_xscale("log")

    fig.tight_layout()
    fig.savefig("/work/ec110/ec110/cc2040/research/mrop/paper2/low_rankness/results/singular_values.png", dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for i, S in enumerate(Ss):
        S /= S.max()
        ax.plot(S, label=f"Run {i+1}")
    ax.set_yscale("log")
    ax.set_xscale("log")

    fig.tight_layout()
    fig.savefig("/work/ec110/ec110/cc2040/research/mrop/paper2/low_rankness/results/singular_values_normalised.png", dpi=300, bbox_inches='tight')

