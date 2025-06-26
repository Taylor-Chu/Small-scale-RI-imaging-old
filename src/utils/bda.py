from typing import Literal, Optional, Tuple

import torch
from scipy.constants import speed_of_light
from scipy.io import loadmat
from scipy.io.matlab import matfile_version
from tqdm import tqdm
import numpy as np
import h5py
import os


def symmetrisation(y_triu, real_data, B, Q, batches, ant1, ant2):
    """
    For a given vector y_triu (of size BQ(Q-1)/2) containing concatenation of the upper triangular part of the covariance matrix at each batch,
    reconstruct the covariance matrix (of size BQ^2) at each batch.

    :param y_triu: upper triangular part of the covariance matrix
    :type y_triu: torch.Tensor
    :return: symmetrised covariance matrix
    :rtype: torch.Tensor
    """
    if not real_data:
        y_triu = y_triu.view(B, Q * (Q - 1) // 2)
        y = torch.zeros(B, Q, Q, dtype=y_triu.dtype, device=y_triu.device)
        triu_r, triu_c = torch.triu_indices(Q, Q, offset=1)

        y[:, triu_r, triu_c] = y_triu
        y[:, triu_c, triu_r] = y_triu.conj()
        return y
    else:
        y = torch.zeros(B, Q, Q, dtype=y_triu.dtype, device=y_triu.device)
        y[batches, ant1, ant2] = y_triu
        y[batches, ant2, ant1] = y_triu.conj()
        return y


def gen_BDA(
    data_file,
    data_path,
    freq_num: int = None,
    vla=False,
    uv_pt="midpoint",
    start_zone=1,
    start_num=1,
    sr_factor=1.5,
    real_data=True,
    device=torch.device("cpu"),
    return_B=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate selection matrix and average vector for BDA

    :return: _description_
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    assert uv_pt in ["midpoint", "avg"], "uv_pt must be either 'midpoint' or 'avg'"
    data = {}
    mat_version, _ = matfile_version(data_file)
    keys = ["uvw", "freqs"]
    if mat_version == 2:
        with h5py.File(data_file, "r") as h5File:
            for key, h5obj in h5File.items():
                if isinstance(h5obj, h5py.Dataset):
                    data[key] = np.array(h5obj)
                    if data[key].dtype.names and "imag" in data[key].dtype.names:
                        data[key] = data[key]["real"] + 1j * data[key]["imag"]
                elif isinstance(h5obj, h5py.Group):
                    data[key] = {}
                    for key2, h5obj2 in h5obj.items():
                        data[key][key2] = np.array(h5obj2)
                        if data[key][key2].dtype.names and "imag" in data[key][key2].dtype.names:
                            data[key][key2] = data[key][key2]["real"] + 1j * data[key][key2]["imag"]
                else:
                    print("Type not implemented to be read here", h5obj)
    else:
        loadmat(data_file, mdict=data, variable_names=keys)

    freqs = [data["freqs"].squeeze()[freq_num - 1]]
    Q = 27

    i_f = freq_num - 1
    data_tmp = loadmat(
        os.path.join(data_path, f"273-X08_data_ch_{i_f+1}.mat"),
        variable_names=["batches_flagged", "ant1_flagged", "ant2_flagged", "flag"],
    )
    B = len(np.unique(data_tmp["batches_flagged"]))
    u = data["uvw"][:, 0][data_tmp["flag"].squeeze() == 1] / (speed_of_light / freqs[i_f])
    u = torch.tensor(u, dtype=torch.float64, device=device)
    v = data["uvw"][:, 1][data_tmp["flag"].squeeze() == 1] / (speed_of_light / freqs[i_f])
    v = torch.tensor(-v, dtype=torch.float64, device=device)
    ant1 = data_tmp["ant1_flagged"].squeeze().astype(int)
    ant2 = data_tmp["ant1_flagged"].squeeze().astype(int)
    batches = data_tmp["batches_flagged"].squeeze().astype(int) # + data["B_per_ch"] * i_f

    u = symmetrisation(
        u, real_data, B, Q, batches, ant1, ant2
    )
    v = symmetrisation(
        v, real_data, B, Q, batches, ant1, ant2
    )
    
    # V = 0.5 * B * Q * (Q - 1)
    V = (data_tmp["flag"][data_tmp["flag"].squeeze() == 1]).size
    
    baselines = torch.sqrt(u.ravel() ** 2 + v.ravel() ** 2)
    baseline_min = torch.min(baselines[baselines > 0])
    baseline_max = torch.max(baselines[baselines > 0])

    # Compute the 'baseline zones' for BDA, starting from max, decreasing by factor of 2
    baseline_edges = torch.tensor([baseline_max])
    num_pt_avg = torch.tensor([start_num], dtype=torch.int32)

    # baseline_div = 1.5 if vla else 2
    baseline_div = 2
    baseline_edge_new = baseline_edges[-1] / baseline_div
    c = 0
    while baseline_edge_new > baseline_min:
        baseline_edges = torch.cat([baseline_edges, baseline_edge_new.unsqueeze(0)])
        # average over +1 point to get odd number of points
        # so that the mid-point can be used as the averaged uv point
        if c + 1 >= start_zone:
            print(c)
            if uv_pt == "midpoint":
                if c + 1 == start_zone and start_num == 1:
                    new_num_pt_avg = torch.tensor([3], dtype=torch.int32)
                else:
                    new_num_pt_avg = num_pt_avg[-1].unsqueeze(0) * 2 - 1
            elif uv_pt == "avg":
                new_num_pt_avg = num_pt_avg[-1].unsqueeze(0) * 2
        else:
            new_num_pt_avg = torch.tensor([start_num], dtype=torch.int32)
        num_pt_avg = torch.cat([num_pt_avg, new_num_pt_avg])
        baseline_edge_new = baseline_edges[-1] / baseline_div
        if num_pt_avg[-1] * 2 > 512:
            break
        c += 1
    for i in range(len(num_pt_avg)):
        print(f"Zone {i}: {num_pt_avg[i].item()} points")

    baseline_edges = torch.cat([baseline_edges, torch.tensor([0.0])])
    baseline_edges = torch.flip(baseline_edges, dims=[0])  # Flip for increasing order
    baseline_edges[-1] = baseline_max * 1.01
    num_pt_avg = torch.flip(num_pt_avg, dims=[0]).view(-1, 1)

    # return num_pt_avg, baseline_edges

    blz_avg_pt = {b: 0 for b in range(len(num_pt_avg))}

    # Reshape u and v into shape (nTimeSamples, na**2) to
    # get the batches/ arc for each baseline/ antenna pair

    W_bda, u_s, v_s = (
        torch.zeros(B * Q * Q, dtype=torch.int),
        torch.zeros(B * Q * Q, dtype=torch.float64),
        torch.zeros(B * Q * Q, dtype=torch.float64),
    )

    I_s_i, I_s_j = (
        torch.zeros(B, Q, Q, dtype=torch.float64),
        torch.zeros(B, Q, Q, dtype=torch.float64),
    )
    if return_B:
        Bs = torch.zeros(B * Q * Q, dtype=torch.int)

    w_counter = 0
    for q1 in tqdm(range(Q)):
        for q2 in range(Q):
            I_s_counter = 0
            baseline_q = torch.sqrt(u[:, q1, q2] ** 2 + v[:, q1, q2] ** 2)
            bins_q = torch.bucketize(baseline_q, baseline_edges, right=True) - 1
            # check at which indicies bins_q is different from the one before
            diff = bins_q[1:] - bins_q[:-1]
            # blz = BaseLine Zone
            # this is to avoid discontinuity in averaging points in the same baseline zone
            # i.e. no averaging of points in the same baseline zone separated by other baseline zones
            blz_start_idxs = torch.cat([torch.tensor([0]), torch.where(diff != 0)[0] + 1])
            blz_end_idx = torch.cat([blz_start_idxs[1:], torch.tensor([B])])

            for i, blz_start_idx in enumerate(blz_start_idxs):
                blz = bins_q[blz_start_idx]
                # number of points in current baseline zone
                len_blz = blz_end_idx[i] - blz_start_idx
                # number of averaged points in the baseline zone
                num_avg_points_in_blz = torch.ceil(len_blz / num_pt_avg[blz]).to(torch.int).item()

                for k in range(num_avg_points_in_blz):
                    k_start = I_s_counter
                    k_end = min(
                        k_start + num_pt_avg[blz].item(), blz_end_idx[i].item()
                    )  # end index of the averaged points
                    k_length = k_end - k_start

                    I_s_i[k_start:k_end, q1, q2] = w_counter
                    I_s_j[k_start:k_end, q1, q2] = torch.tensor(
                        [k1 * Q**2 + (q2 + q1 * Q) for k1 in range(k_start, k_end)]
                    )

                    if uv_pt == "midpoint":
                        u_s[w_counter] = u[(k_start + k_end - 1) // 2, q1, q2]
                        v_s[w_counter] = v[(k_start + k_end - 1) // 2, q1, q2]
                    elif uv_pt == "avg":
                        # find the central two points in k_start:k_end
                        # k_mid_start = (k_start + k_end) // 2
                        # k_mid_end = k_mid_start + 2
                        # averaging the central two points
                        # u_s[w_counter] = torch.mean(u[k_mid_start:k_mid_end, q1, q2])
                        # v_s[w_counter] = torch.mean(v[k_mid_start:k_mid_end, q1, q2])
                        u_s[w_counter] = torch.mean(u[k_start:k_end, q1, q2])
                        v_s[w_counter] = torch.mean(v[k_start:k_end, q1, q2])
                    if return_B:
                        Bs[w_counter] = (k_start + k_end - 1) // 2

                    W_bda[w_counter] = k_length
                    I_s_counter += k_length
                    blz_avg_pt[blz.item()] += 1
                    # check BDA performance
                    # largest dt such that if averaging start at 2 rather than 1 BDA performance degrades
                    # dt such that if dt is increased , BDA performance degrades

                    w_counter += 1

    W_bda = W_bda[:w_counter]
    u_s = u_s[:w_counter]
    v_s = v_s[:w_counter]

    u_s /= (speed_of_light / freqs[i_f])
    v_s /= (speed_of_light / freqs[i_f])
    maxProjBaseline = torch.sqrt(torch.max(u_s**2 + v_s**2))
    spatial_bandwidth = 2 * maxProjBaseline.item()
    im_pixel_size = (180.0 / torch.pi) * 3600.0 / (sr_factor * spatial_bandwidth)
    half_spatial_bandwidth = (180.0 / torch.pi) * 3600.0 / (im_pixel_size) / 2.0
    u_s = u_s * torch.pi / half_spatial_bandwidth
    v_s = v_s * torch.pi / half_spatial_bandwidth

    print(f"{u_s.numel()}/{V} ({u_s.numel()/V * 100:.2f}%) points averaged")

    if return_B:
        return W_bda, u_s, v_s, I_s_i, I_s_j, Bs  # , blz_avg_pt, baseline_edges  # , u_bda, v_bda  # , y2
    else:
        return W_bda, u_s, v_s, I_s_i, I_s_j, B, Q, batches, ant1, ant2  # , blz_avg_pt, baseline_edges  # , u_bda, v_bda  # , y2


def BDA_vis(y, idx):
    if "torch" not in str(type(idx)):
        idx = torch.tensor(idx, dtype=torch.int)
    M = int(idx.max().item()) + 1
    V_flat = y.ravel()
    S_flat = idx.ravel().to(dtype=torch.int64, device=y.device)
    sum_values = torch.zeros(M, dtype=torch.complex128, device=y.device)
    count_values = torch.zeros(M, dtype=torch.int64, device=y.device)

    # Sum values by index
    sum_values.scatter_add_(0, S_flat, V_flat)

    # Count occurrences of each index
    count_values.scatter_add_(0, S_flat, torch.ones_like(V_flat, dtype=torch.long))

    # Avoid division by zero; if any index doesn't occur, we set its count to 1.
    count_values = count_values.clamp(min=1)

    # Compute the average values
    avg_values = sum_values / count_values

    return avg_values
