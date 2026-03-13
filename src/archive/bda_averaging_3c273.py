#!/usr/bin/env python3
"""
Baseline-Dependent Averaging (BDA) for Radio Interferometric Data
=================================================================

Implements BDA in both time and frequency following:
  [1] Wijnholds, Willis & Salvini (2018), "Baseline Dependent Averaging
      in Radio Interferometry", MNRAS (arXiv:1802.09321)
  [2] Atemkeng, Smirnov, Tasse et al. (2018), "Baseline-dependent sampling
      and windowing for radio interferometry", MNRAS

Averaging scheme (Scheme 2 from [1], Sec. 4.5, extended to frequency per [2]):
    n_t(pq) = min(max_avg_t, floor(b_max / b_pq))
    n_f(pq) = min(max_avg_f, floor(b_max / b_pq))

Output format
-------------
Flat vectors of averaged (u, v) coordinates and a function to average
visibilities in the same order.  The ordering is:

    for iFreq in range(F_out):        # output frequency bins (per baseline)
        for each unflagged BDA bin at this frequency:
            append averaged u, v

This mirrors the original concatenation convention:
    [freq0_baselines, freq1_baselines, ..., freqF_baselines]

Within each frequency, the BDA bins are ordered by (baseline, time_bin).

Key Functions
-------------
apply_bda() : Compute BDA bins and average u,v coordinates
average_visibilities() : Average visibilities using BDA bins
average_natural_weights() : Average natural weights using BDA bins
load_visibilities() : Load visibilities in correct order for average_visibilities()
load_uv_coordinates() : Load u,v coordinates matching visibility order
load_natural_weights() : Load natural weights matching visibility order

Data Loading
------------
IMPORTANT: The average_visibilities() function expects visibilities in a specific
order. Use load_visibilities() to load data in the correct format, which guarantees
consistency with the BDA bin construction. Do NOT use other loading methods unless
you understand the exact ordering convention.
"""

import argparse
import time as timer
from pathlib import Path

import numpy as np
import scipy.io as sio

# ──────────────────────────────────────────────────────────────────────
C_LIGHT = 299792458.0  # m/s


# ──────────────────────────────────────────────────────────────────────
def baseline_indices(Q):
    """Return (p, q) arrays for the Q(Q-1)/2 upper-triangle baselines."""
    return np.triu_indices(Q, k=1)


# ──────────────────────────────────────────────────────────────────────
def compute_baseline_lengths(u_m, v_m, Q, B):
    """
    Mean physical baseline length (meters) for each of the V baselines,
    averaged over all B snapshots.
    """
    V = Q * (Q - 1) // 2
    u_2d = u_m.reshape(B, V)
    v_2d = v_m.reshape(B, V)
    lengths = np.sqrt(u_2d**2 + v_2d**2)  # (B, V)
    return np.mean(lengths, axis=0)  # (V,)


# ──────────────────────────────────────────────────────────────────────
def compute_bda_factors(
    bl_lengths, max_avg_time, max_avg_freq, freqs=None, img_half_width_rad=None, smearing_limit=1.0
):
    """
    Per-baseline averaging factors (Scheme 2 from [1]):

        n_pq = min(max_avg, floor(b_max / b_pq))

    For frequency averaging an additional smearing check is applied when
    ``freqs`` and ``img_half_width_rad`` are provided.  The BDA formula
    ``floor(b_max / b_pq)`` is derived under the implicit assumption that
    one channel at the longest baseline is already at the smearing limit.
    When the channels are *wide* (large δν / ν) this formula can assign
    n_freq > 1 even though a single wide channel already exceeds the
    smearing criterion.  The correction caps n_freq at

        n_freq_smear = floor( smearing_limit * c / (δν * b_pq * θ_FOV) )

    where δν is the mean channel spacing and θ_FOV is the image half-width.

    Parameters
    ----------
    bl_lengths          : (V,) baseline lengths in meters
    max_avg_time        : int, maximum time-averaging factor
    max_avg_freq        : int, maximum frequency-averaging factor
    freqs               : (F,) Hz, channel centre frequencies.  If given
                          together with img_half_width_rad the smearing cap
                          is applied.
    img_half_width_rad  : float, image half-width in radians (θ_FOV).
    smearing_limit      : float, maximum allowed phase smear in radians
                          (default 1.0 = lenient; use π/4 ≈ 0.79 for <10%
                          amplitude loss).

    Returns
    -------
    n_time, n_freq : (V,) int arrays
    """
    b_max = np.max(bl_lengths)
    ratio = b_max / bl_lengths  # >= 1

    n_time = np.clip(np.floor(ratio).astype(int), 1, max_avg_time)
    n_freq = np.clip(np.floor(ratio).astype(int), 1, max_avg_freq)

    # ── Smearing correction for frequency averaging ──
    if freqs is not None and img_half_width_rad is not None and max_avg_freq > 1:
        freqs = np.asarray(freqs)
        if len(freqs) > 1:
            delta_nu = np.mean(np.diff(freqs))  # mean channel spacing [Hz]
        else:
            delta_nu = freqs[0]  # single channel – no averaging possible

        # Maximum allowed n_freq from smearing criterion per baseline
        if delta_nu > 0 and img_half_width_rad > 0:
            n_freq_smear = np.floor(
                smearing_limit * C_LIGHT / (delta_nu * bl_lengths * img_half_width_rad)
            ).astype(int)
            n_freq = np.minimum(n_freq, np.maximum(n_freq_smear, 1))

    return n_time, n_freq


# ──────────────────────────────────────────────────────────────────────
def build_bda_bins(B, F, V, n_time, n_freq, flag_use):
    """
    Construct the list of BDA bins that contain at least one unflagged
    sample.

    Parameters
    ----------
    B, F, V   : int
    n_time    : (V,) per-baseline time-averaging factor
    n_freq    : (V,) per-baseline frequency-averaging factor
    flag_use  : (F, B, V) bool/uint8, True/1 = flagged

    Returns
    -------
    bins : list of dicts, one per valid (non-empty) BDA output sample.
           Each dict: 'bl', 't_start', 't_end', 'f_start', 'f_end',
                      'unflagged_mask' (nf_chunk, nt_chunk) bool,
                      'n_unflagged' int
    """
    bins = []
    for bl in range(V):
        nt = int(n_time[bl])
        nf = int(n_freq[bl])
        fl_bl = flag_use[:, :, bl]  # (F, B)

        n_t = (B + nt - 1) // nt  # number of time blocks (ceil)
        n_f = (F + nf - 1) // nf  # number of freq blocks (ceil)

        # Pad to exact multiples so reshape works; padding = 1 (flagged)
        F_pad = n_f * nf
        B_pad = n_t * nt
        if F_pad != F or B_pad != B:
            fl_pad = np.ones((F_pad, B_pad), dtype=np.uint8)
            fl_pad[:F, :B] = fl_bl
        else:
            fl_pad = fl_bl

        # Count unflagged per (freq_block, time_block) in one numpy call
        unfl_count = np.sum(fl_pad.reshape(n_f, nf, n_t, nt) == 0, axis=(1, 3))  # (n_f, n_t)

        # Iterate only over non-empty blocks (typically a small fraction)
        fi_arr, ti_arr = np.where(unfl_count > 0)
        for fi, ti in zip(fi_arr.tolist(), ti_arr.tolist()):
            f = fi * nf
            t = ti * nt
            f_end = min(f + nf, F)
            t_end = min(t + nt, B)
            chunk = fl_bl[f:f_end, t:t_end]
            unflagged = chunk == 0
            bins.append(
                {
                    "bl": bl,
                    "t_start": t,
                    "t_end": t_end,
                    "f_start": f,
                    "f_end": f_end,
                    "unflagged_mask": unflagged,
                    "n_unflagged": int(np.sum(unflagged)),
                }
            )

    return bins


# ──────────────────────────────────────────────────────────────────────
def apply_bda(
    data_file, max_avg_time=32, max_avg_freq=4, img_half_width_rad=None, smearing_limit=1.0, verbose=True
):
    """
    Compute BDA bins and average (u, v) coordinates.

    The output u_avg, v_avg are flat vectors (N_out,) in units of
    wavelengths. Flagged samples are excluded from the average (never
    contribute). Each output element corresponds to exactly one BDA bin.

    Parameters
    ----------
    data               : dict from scipy.io.loadmat
    max_avg_time       : int, max time bins to average (for shortest baselines)
    max_avg_freq       : int, max frequency channels to average
    img_half_width_rad : float or None.  Image half-width in radians used to
                         apply the channel-smearing cap on n_freq (see
                         ``compute_bda_factors``).  When None the cap is
                         *not* applied and the raw BDA formula is used –
                         this is fine when all channels are already narrow
                         (δν/ν ≪ 1/(b_max * θ_FOV)), but will over-average
                         when channels are wide.
    smearing_limit     : float, max allowable smearing in radians passed to
                         ``compute_bda_factors``.  Default 1.0 (lenient).
    verbose            : bool

    Returns
    -------
    result : dict
    """
    t0 = timer.time()

    # ── Unpack main file ──
    import os

    from scipy.io import loadmat

    data = loadmat(data_file)
    uvw_all = data["uvw"]  # (N, 3)  N = #(batch,bl) pairs present
    u_all = uvw_all[:, 0]  # (N,) meters
    v_all = uvw_all[:, 1]
    freqs = data["freqs"].squeeze()  # (F,)
    ant1_all = data["ant1"].squeeze().astype(int)  # (N,) 0-indexed
    ant2_all = data["ant2"].squeeze().astype(int)  # (N,) 0-indexed, ant1 < ant2
    bk_all = data["batches"].squeeze().astype(int)  # (N,) snapshot index

    Q = 27
    V = Q * (Q - 1) // 2  # 351
    F = len(freqs)  # 64
    data_path = os.path.dirname(data_file)

    # ── Build index maps ──
    # Map each batch label → sequential row index 0..B-1
    unique_batches = np.unique(bk_all)
    B = len(unique_batches)
    batch_to_row = np.empty(int(unique_batches.max()) + 1, dtype=int)
    batch_to_row[unique_batches] = np.arange(B)
    row_idx = batch_to_row[bk_all]  # (N,) row in (B, V) grid

    # Map (ant1, ant2) → baseline column index 0..V-1 (upper-triangle order)
    p_arr, q_arr = np.triu_indices(Q, k=1)
    bl_map = np.full((Q, Q), -1, dtype=int)
    bl_map[p_arr, q_arr] = np.arange(V)
    col_idx = bl_map[ant1_all, ant2_all]  # (N,) column in (B, V) grid

    if verbose:
        print(f"Data: Q={Q}, V={V}, B={B}, F={F}, N_entries={len(u_all)}")
        print(f"Frequencies: {freqs[0]/1e9:.4f} – {freqs[-1]/1e9:.4f} GHz")

    # ── Scatter uvw into (B, V) grid ──
    # Missing (batch, bl) pairs stay at zero (they are flagged anyway).
    u_2d = np.zeros((B, V))
    v_2d = np.zeros((B, V))
    u_2d[row_idx, col_idx] = u_all
    v_2d[row_idx, col_idx] = v_all

    # ── Build flag_use (F, B, V): 0 = unflagged, 1 = flagged ──
    # Per-channel files have flag (N,) with 1 = unflagged, 0 = flagged.
    # Absent (batch, bl) pairs default to flagged (1).
    flag_use = np.ones((F, B, V), dtype=np.uint8)
    for i_f in range(F):
        data_ch = loadmat(
            os.path.join(data_path, f"273-X08_data_ch_{i_f+1}.mat"),
            variable_names=["flag"],
        )
        ch_flag = data_ch["flag"].squeeze()  # (N,) — 1=unflagged
        flag_use[i_f, row_idx, col_idx] = (1 - ch_flag).astype(np.uint8)

    # ── Baseline lengths & averaging factors ──
    # compute_baseline_lengths expects a flat (B*V,) vector; pass the flattened grid
    bl_lengths = compute_baseline_lengths(u_2d.ravel(), v_2d.ravel(), Q, B)

    if verbose:
        print(f"Baseline range: {bl_lengths.min():.1f} – {bl_lengths.max():.1f} m")

    n_time, n_freq = compute_bda_factors(
        bl_lengths,
        max_avg_time,
        max_avg_freq,
        freqs=freqs,
        img_half_width_rad=img_half_width_rad,
        smearing_limit=smearing_limit,
    )

    if verbose:
        print(
            f"BDA factors (Scheme 2): time [{n_time.min()}, {n_time.max()}], "
            f"freq [{n_freq.min()}, {n_freq.max()}]"
        )

    # ── Vectorized BDA: one numpy pass per baseline ──────────────────
    # For each baseline bl with averaging factors (nt, nf):
    #   1. Pad flag_use[:, :, bl] to multiples of (nf, nt)
    #   2. Reshape to (n_f, nf, n_t, nt) and sum axis (1,3) → valid bin mask
    #   3. Compute u,v wavelength grid and perform masked block-average
    # All per-bin loops are avoided; only 351 Python iterations remain.

    total_input_unflagged = int(np.sum(flag_use == 0))

    # Accumulators (one entry per BDA output bin, appended per baseline)
    u_avg_list, v_avg_list = [], []
    bl_out_list = []
    t0_out, t1_out, f0_out, f1_out = [], [], [], []
    nu_out = []

    for bl in range(V):
        nt = int(n_time[bl])
        nf = int(n_freq[bl])
        fl_bl = flag_use[:, :, bl]  # (F, B)

        n_t = (B + nt - 1) // nt
        n_f = (F + nf - 1) // nf
        B_pad = n_t * nt
        F_pad = n_f * nf
        need_pad = (B_pad != B) or (F_pad != F)

        if need_pad:
            fl_pad = np.ones((F_pad, B_pad), dtype=np.uint8)
            fl_pad[:F, :B] = fl_bl
        else:
            fl_pad = fl_bl

        unfl_4d = fl_pad.reshape(n_f, nf, n_t, nt) == 0  # (n_f, nf, n_t, nt)
        n_unfl_2d = unfl_4d.sum(axis=(1, 3))  # (n_f, n_t)

        fi_v, ti_v = np.where(n_unfl_2d > 0)
        n_bins = len(fi_v)
        if n_bins == 0:
            continue

        # ── u, v in wavelengths for this baseline ──
        u_bl = u_2d[:, bl]  # (B,) metres
        v_bl = v_2d[:, bl]
        u_lam = np.outer(freqs, u_bl) / C_LIGHT  # (F, B)
        v_lam = np.outer(freqs, v_bl) / C_LIGHT

        if need_pad:
            u_pad = np.zeros((F_pad, B_pad))
            v_pad = np.zeros((F_pad, B_pad))
            u_pad[:F, :B] = u_lam
            v_pad[:F, :B] = v_lam
        else:
            u_pad = u_lam
            v_pad = v_lam

        # Block-sum weighed by unflagged mask → mean per bin
        u_sum = np.where(unfl_4d, u_pad.reshape(n_f, nf, n_t, nt), 0.0).sum(axis=(1, 3))
        v_sum = np.where(unfl_4d, v_pad.reshape(n_f, nf, n_t, nt), 0.0).sum(axis=(1, 3))
        n_v = n_unfl_2d[fi_v, ti_v]  # (n_bins,) int

        u_avg_list.append(u_sum[fi_v, ti_v] / n_v)
        v_avg_list.append(v_sum[fi_v, ti_v] / n_v)
        bl_out_list.append(np.full(n_bins, bl, dtype=np.int32))
        f0_out.append(fi_v * nf)
        f1_out.append(np.minimum((fi_v + 1) * nf, F))
        t0_out.append(ti_v * nt)
        t1_out.append(np.minimum((ti_v + 1) * nt, B))
        nu_out.append(n_v)

    # ── Concatenate per-baseline results ──
    u_avg = np.concatenate(u_avg_list)
    v_avg = np.concatenate(v_avg_list)
    bl_arr = np.concatenate(bl_out_list)
    f0_arr = np.concatenate(f0_out).astype(np.int32)
    f1_arr = np.concatenate(f1_out).astype(np.int32)
    t0_arr = np.concatenate(t0_out).astype(np.int32)
    t1_arr = np.concatenate(t1_out).astype(np.int32)
    nu_arr = np.concatenate(nu_out).astype(np.int32)

    N_out = len(u_avg)
    actual_compression = total_input_unflagged / N_out

    # Wrap bin metadata in a compact dict so average_visibilities can use it
    bins = {
        "bl": bl_arr,
        "f_start": f0_arr,
        "f_end": f1_arr,
        "t_start": t0_arr,
        "t_end": t1_arr,
        "n_unflagged": nu_arr,
    }

    elapsed = timer.time() - t0
    if verbose:
        print(f"BDA output: {N_out:,} bins " f"(from {total_input_unflagged:,} unflagged inputs)")
        print(
            f"Compression: {actual_compression:.2f}×, "
            f"reduction: {(1 - N_out / total_input_unflagged) * 100:.1f}%"
        )
        print(f"Done in {elapsed:.1f}s.")

    return {
        "u_avg": u_avg,  # (N_out,) wavelengths
        "v_avg": v_avg,  # (N_out,) wavelengths
        "bins": bins,  # dict of arrays for average_visibilities()
        "n_time": n_time,  # (V,)
        "n_freq": n_freq,  # (V,)
        "bl_lengths": bl_lengths,  # (V,) meters
        "frequencies": freqs,  # (F,) Hz
        "flag_use": flag_use,  # (F, B, V) – reference
        "u_2d": u_2d,  # (B, V) meters – reference
        "v_2d": v_2d,  # (B, V) meters – reference
        "Q": Q,
        "V": V,
        "B": B,
        "F": F,
        "max_avg_time": max_avg_time,
        "max_avg_freq": max_avg_freq,
        "compression_actual": actual_compression,
        "total_input_unflagged": total_input_unflagged,
        "total_output_samples": N_out,
    }


# ──────────────────────────────────────────────────────────────────────
def average_visibilities(vis_flat, data, result):
    """
    Average a flat visibility vector using the BDA bins from `apply_bda`.

    The input is the raw output of the measurement operator: a flat
    vector of *only* the unflagged visibilities, ordered by
    snapshot (fastest) then baseline within each frequency block:

        vis_flat = [freq0_unflagged, freq1_unflagged, ..., freqF_unflagged]

    where within each frequency the unflagged entries follow C-order
    over (snapshot, baseline), i.e. the same order as:

        np.concatenate([
            data['u'][data['flag'][0, f, :] == False] / (c / freq[f])
            for f in range(F)
        ])

    Flagged entries (zeros in the (F,B,V) grid) never contribute to
    the average. The output is **weighted** (vis_avg * weight) so it
    can be passed directly to the adjoint NUFFT to produce a correct
    dirty image.

    Parameters
    ----------
    vis_flat : (N_unflagged,) complex vector of unflagged visibilities.
    data     : dict from scipy.io.loadmat (needs 'flag' and 'nFreqs').
    result   : dict returned by `apply_bda`.

    Returns
    -------
    vis_avg_weighted : (N_out,) complex, averaged visibilities multiplied
                       by the bin weight (number of samples per bin).
                       Same ordering as result['u_avg'] / result['v_avg'].
    weights          : (N_out,) int, number of unflagged samples per bin.
    """
    F = result["F"]
    B = result["B"]
    V = result["V"]
    flag_use = result["flag_use"]  # (F, B, V), 1=flagged, 0=unflagged
    n_time = result["n_time"]  # (V,)
    n_freq = result["n_freq"]  # (V,)

    # ── Scatter the flat unflagged vector into the (F, B, V) grid ──
    vis_dtype = vis_flat.dtype if np.iscomplexobj(vis_flat) else np.complex128
    vis_grid = np.zeros((F, B, V), dtype=vis_dtype)
    idx = 0
    for f in range(F):
        mask_f = flag_use[f] == 0  # (B, V) bool
        n_f = int(np.sum(mask_f))
        vis_grid[f][mask_f] = vis_flat[idx : idx + n_f]
        idx += n_f

    # ── Vectorized bin-average, one baseline at a time ──
    bins = result["bins"]  # dict of arrays
    bl_arr = bins["bl"]
    N_out = len(bl_arr)

    vis_avg_weighted = np.empty(N_out, dtype=vis_dtype)
    weights = np.empty(N_out, dtype=np.int32)

    # Keep a write cursor per baseline so we can map bl → output slice
    # Build per-baseline index into the output array
    bl_start = np.searchsorted(bl_arr, np.arange(V))  # first bin of each bl(sorted)
    bl_count = np.diff(np.append(bl_start, N_out))  # bins per baseline

    for bl in range(V):
        n_bins_bl = int(bl_count[bl])
        if n_bins_bl == 0:
            continue
        start = int(bl_start[bl])

        nt = int(n_time[bl])
        nf = int(n_freq[bl])
        fl_bl = flag_use[:, :, bl]  # (F, B)
        vis_bl = vis_grid[:, :, bl]  # (F, B)

        n_t = (B + nt - 1) // nt
        n_f = (F + nf - 1) // nf
        B_pad = n_t * nt
        F_pad = n_f * nf
        need_pad = (B_pad != B) or (F_pad != F)

        if need_pad:
            fl_pad = np.ones((F_pad, B_pad), dtype=np.uint8)
            vis_pad = np.zeros((F_pad, B_pad), dtype=vis_dtype)
            fl_pad[:F, :B] = fl_bl
            vis_pad[:F, :B] = vis_bl
        else:
            fl_pad = fl_bl
            vis_pad = vis_bl

        unfl_4d = fl_pad.reshape(n_f, nf, n_t, nt) == 0
        n_unfl_2d = unfl_4d.sum(axis=(1, 3))  # (n_f, n_t)

        vis_sum_2d = np.where(unfl_4d, vis_pad.reshape(n_f, nf, n_t, nt), 0).sum(axis=(1, 3))  # (n_f, n_t)

        # Retrieve the (fi, ti) indices stored in the bins dict for THIS baseline
        f0 = bins["f_start"][start : start + n_bins_bl]
        t0 = bins["t_start"][start : start + n_bins_bl]
        fi = (f0 // nf).astype(int)
        ti = (t0 // nt).astype(int)
        nu = n_unfl_2d[fi, ti]

        vis_avg_weighted[start : start + n_bins_bl] = vis_sum_2d[fi, ti]
        weights[start : start + n_bins_bl] = nu

    return vis_avg_weighted, weights


# ──────────────────────────────────────────────────────────────────────
def average_natural_weights(natural_weights_sqrt, result):
    """
    Combine natural weights within each BDA bin.
    
    If the input contains sqrt(w) (as is conventional in the data files),
    this function computes sqrt(sum(w)) for each BDA bin:
      1. Square the input: w = sqrt(w)^2
      2. Sum within bins: sum(w)
      3. Take square root: sqrt(sum(w))
    
    This ensures proper weighting when both the measurement operator and
    visibility are multiplied by the result.

    Parameters
    ----------
    natural_weights_sqrt : (F, B, V) float array, containing sqrt(w) values
    result               : dict returned by `apply_bda`

    Returns
    -------
    w_avg_sqrt : (N_out,) sqrt of summed weights per BDA bin
    """
    F = result["F"]
    B = result["B"]
    V = result["V"]
    n_time = result["n_time"]
    n_freq = result["n_freq"]
    bins = result["bins"]
    bl_arr = bins["bl"]
    N_out = len(bl_arr)
    w_avg_sqrt = np.empty(N_out)

    bl_start = np.searchsorted(bl_arr, np.arange(V))
    bl_count = np.diff(np.append(bl_start, N_out))

    for bl in range(V):
        n_bins_bl = int(bl_count[bl])
        if n_bins_bl == 0:
            continue
        start = int(bl_start[bl])
        nt = int(n_time[bl])
        nf = int(n_freq[bl])
        nw_sqrt_bl = natural_weights_sqrt[:, :, bl]  # (F, B), sqrt(w) values

        n_t = (B + nt - 1) // nt
        n_f = (F + nf - 1) // nf
        B_pad = n_t * nt
        F_pad = n_f * nf
        if B_pad != B or F_pad != F:
            nw_sqrt_pad = np.zeros((F_pad, B_pad))
            nw_sqrt_pad[:F, :B] = nw_sqrt_bl
        else:
            nw_sqrt_pad = nw_sqrt_bl

        # Square, sum, then sqrt: sqrt(sum(w)) = sqrt(sum(sqrt(w)^2))
        nw_squared = nw_sqrt_pad ** 2
        nw_sum = nw_squared.reshape(n_f, nf, n_t, nt).sum(axis=(1, 3))  # (n_f, n_t)
        nw_sum_sqrt = np.sqrt(nw_sum)
        
        f0 = bins["f_start"][start : start + n_bins_bl]
        t0 = bins["t_start"][start : start + n_bins_bl]
        fi = (f0 // nf).astype(int)
        ti = (t0 // nt).astype(int)
        w_avg_sqrt[start : start + n_bins_bl] = nw_sum_sqrt[fi, ti]

    return w_avg_sqrt


# ──────────────────────────────────────────────────────────────────────
def load_visibilities(data_file, result, verbose=True):
    """
    Load visibilities from per-channel .mat files in the correct order to
    match the expected format for average_visibilities().

    The visibilities are loaded into an (F, B, V) grid and then extracted
    as a flat array in the order: [freq0_unflagged, freq1_unflagged, ...]
    where within each frequency, unflagged entries follow C-order over (B, V).

    This guarantees consistency with the BDA bin construction in apply_bda().

    Parameters
    ----------
    data_file : str
        Path to the main msSpecs.mat file
    result : dict
        Result dictionary from apply_bda() containing F, B, V, flag_use
    verbose : bool
        Print progress messages

    Returns
    -------
    vis_flat : (N_unflagged,) complex array
        Visibilities in the correct order for average_visibilities()
    """
    import os

    from scipy.io import loadmat

    data_path = os.path.dirname(data_file)
    F = result["F"]
    B = result["B"]
    V = result["V"]
    flag_use = result["flag_use"]  # (F, B, V), 0=unflagged, 1=flagged

    # Load main file for index mapping
    raw = loadmat(data_file)
    ant1_all = raw["ant1"].squeeze().astype(int)
    ant2_all = raw["ant2"].squeeze().astype(int)
    bk_all = raw["batches"].squeeze().astype(int)
    Q = 27

    # Build index maps (same as in apply_bda)
    unique_batches = np.unique(bk_all)
    batch_to_row = np.empty(int(unique_batches.max()) + 1, dtype=int)
    batch_to_row[unique_batches] = np.arange(B)

    p_arr, q_arr = np.triu_indices(Q, k=1)
    bl_map = np.full((Q, Q), -1, dtype=int)
    bl_map[p_arr, q_arr] = np.arange(V)

    if verbose:
        print(f"Loading visibilities from {F} channel files … ", end="", flush=True)

    t0 = timer.time()
    vis_grid = np.zeros((F, B, V), dtype=np.complex128)

    # Load per-channel visibility files
    for i_f in range(F):
        data_ch = loadmat(
            os.path.join(data_path, f"273-X08_data_ch_{i_f+1}.mat"),
            variable_names=["data_I", "ant1_flagged", "ant2_flagged", "batches_flagged"],
        )

        # Per-channel files contain only unflagged entries
        vis_ch = data_ch["data_I"].squeeze()  # (N_unflagged_ch,)
        ant1_ch = data_ch["ant1_flagged"].squeeze().astype(int)
        ant2_ch = data_ch["ant2_flagged"].squeeze().astype(int)
        bk_ch = data_ch["batches_flagged"].squeeze().astype(int)

        # Map to grid indices
        row_idx_ch = batch_to_row[bk_ch]
        col_idx_ch = bl_map[ant1_ch, ant2_ch]

        # Scatter into grid
        vis_grid[i_f, row_idx_ch, col_idx_ch] = vis_ch

    if verbose:
        print(f"done in {timer.time()-t0:.1f}s")

    # Extract unflagged visibilities in the correct order
    vis_list = []
    for f in range(F):
        mask_f = flag_use[f] == 0  # (B, V) bool, True = unflagged
        vis_list.append(vis_grid[f][mask_f])

    vis_flat = np.concatenate(vis_list)

    if verbose:
        n_expected = int(np.sum(flag_use == 0))
        print(f"Extracted {len(vis_flat):,} unflagged visibilities " f"(expected: {n_expected:,})")
        if len(vis_flat) != n_expected:
            print(f"⚠️  WARNING: Count mismatch!")

    return vis_flat


# ──────────────────────────────────────────────────────────────────────
def load_uv_coordinates(data_file, result, verbose=True):
    """
    Load u,v coordinates in wavelengths for unflagged visibilities, matching
    the order expected by the measurement operator.

    Returns u,v coordinates in the same order as load_visibilities():
    [freq0_unflagged, freq1_unflagged, ...] where within each frequency,
    unflagged entries follow C-order over (B, V).

    Parameters
    ----------
    data_file : str
        Path to the main msSpecs.mat file
    result : dict
        Result dictionary from apply_bda() containing frequencies, u_2d, v_2d,
        flag_use
    verbose : bool
        Print progress messages

    Returns
    -------
    u_wav, v_wav : (N_unflagged,) float arrays
        u,v coordinates in wavelengths
    """
    from scipy.constants import speed_of_light as c

    freqs = result["frequencies"]  # (F,) Hz
    u_2d = result["u_2d"]  # (B, V) meters
    v_2d = result["v_2d"]  # (B, V) meters
    flag_use = result["flag_use"]  # (F, B, V)
    F = result["F"]

    if verbose:
        print("Extracting u,v coordinates for unflagged visibilities … ", end="", flush=True)

    t0 = timer.time()
    u_list = []
    v_list = []

    for f in range(F):
        freq = freqs[f]
        # Convert to wavelengths for this frequency
        u_wav_f = u_2d * freq / c  # (B, V)
        v_wav_f = v_2d * freq / c  # (B, V)

        # Extract unflagged positions
        mask_f = flag_use[f] == 0  # (B, V) bool
        u_list.append(u_wav_f[mask_f])
        v_list.append(v_wav_f[mask_f])

    u_wav = np.concatenate(u_list)
    v_wav = np.concatenate(v_list)

    if verbose:
        print(f"done in {timer.time()-t0:.1f}s")
        print(f"Extracted {len(u_wav):,} u,v pairs")

    return u_wav, v_wav


# ──────────────────────────────────────────────────────────────────────
def load_natural_weights(data_file, result, verbose=True):
    """
    Load natural weights from per-channel .mat files in the correct order to
    match the expected format for average_natural_weights().

    The natural weights are loaded into an (F, B, V) grid and then extracted
    as a flat array in the order: [freq0_unflagged, freq1_unflagged, ...]
    where within each frequency, unflagged entries follow C-order over (B, V).

    This guarantees consistency with the BDA bin construction in apply_bda().

    Parameters
    ----------
    data_file : str
        Path to the main msSpecs.mat file
    result : dict
        Result dictionary from apply_bda() containing F, B, V, flag_use
    verbose : bool
        Print progress messages

    Returns
    -------
    weights_flat : (N_unflagged,) float array
        Natural weights in the correct order for average_natural_weights()
    """
    import os

    from scipy.io import loadmat

    data_path = os.path.dirname(data_file)
    F = result["F"]
    B = result["B"]
    V = result["V"]
    flag_use = result["flag_use"]  # (F, B, V), 0=unflagged, 1=flagged

    # Load main file for index mapping
    raw = loadmat(data_file)
    ant1_all = raw["ant1"].squeeze().astype(int)
    ant2_all = raw["ant2"].squeeze().astype(int)
    bk_all = raw["batches"].squeeze().astype(int)
    Q = 27

    # Build index maps (same as in apply_bda)
    unique_batches = np.unique(bk_all)
    batch_to_row = np.empty(int(unique_batches.max()) + 1, dtype=int)
    batch_to_row[unique_batches] = np.arange(B)

    p_arr, q_arr = np.triu_indices(Q, k=1)
    bl_map = np.full((Q, Q), -1, dtype=int)
    bl_map[p_arr, q_arr] = np.arange(V)

    if verbose:
        print(f"Loading natural weights from {F} channel files … ", end="", flush=True)

    t0 = timer.time()
    weights_grid = np.zeros((F, B, V), dtype=np.float32)

    # Load per-channel weight files
    for i_f in range(F):
        data_ch = loadmat(
            os.path.join(data_path, f"273-X08_data_ch_{i_f+1}.mat"),
            variable_names=["weightsNat", "ant1_flagged", "ant2_flagged", "batches_flagged"],
        )

        # Per-channel files contain only unflagged entries
        weights_ch = data_ch["weightsNat"].squeeze()  # (N_unflagged_ch,)
        ant1_ch = data_ch["ant1_flagged"].squeeze().astype(int)
        ant2_ch = data_ch["ant2_flagged"].squeeze().astype(int)
        bk_ch = data_ch["batches_flagged"].squeeze().astype(int)

        # Map to grid indices
        row_idx_ch = batch_to_row[bk_ch]
        col_idx_ch = bl_map[ant1_ch, ant2_ch]

        # Scatter into grid
        weights_grid[i_f, row_idx_ch, col_idx_ch] = weights_ch

    if verbose:
        print(f"done in {timer.time()-t0:.1f}s")

    # Extract unflagged weights in the correct order
    weights_list = []
    for f in range(F):
        mask_f = flag_use[f] == 0  # (B, V) bool, True = unflagged
        weights_list.append(weights_grid[f][mask_f])

    weights_flat = np.concatenate(weights_list)

    if verbose:
        n_expected = int(np.sum(flag_use == 0))
        print(f"Extracted {len(weights_flat):,} natural weights " f"(expected: {n_expected:,})")
        print(
            f"  Range: [{weights_flat.min():.3f}, {weights_flat.max():.3f}],"
            f" mean: {weights_flat.mean():.3f}"
        )
        if len(weights_flat) != n_expected:
            print(f"⚠️  WARNING: Count mismatch!")

    return weights_flat


# ──────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────
def plot_bda_diagnostics(result, save_path=None):
    """Plot BDA compression factors and uv-coverage."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    bl_km = result["bl_lengths"] / 1e3
    n_time = result["n_time"]
    n_freq = result["n_freq"]

    # (a) Averaging factors vs baseline length
    ax = axes[0, 0]
    ax.scatter(bl_km, n_time, s=5, alpha=0.5, label="Time", color="C0")
    ax.scatter(bl_km, n_freq, s=5, alpha=0.5, label="Freq", color="C1")
    ax.set_xlabel("Baseline length [km]")
    ax.set_ylabel("Averaging factor")
    ax.set_title("BDA Averaging Factors (Scheme 2)")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # (b) Histogram
    ax = axes[0, 1]
    ax.hist(n_time, bins=50, alpha=0.6, label="Time", color="C0")
    ax.hist(n_freq, bins=50, alpha=0.6, label="Freq", color="C1")
    ax.set_xlabel("Averaging factor")
    ax.set_ylabel("Number of baselines")
    ax.set_title("Distribution of Averaging Factors")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) Combined compression per baseline
    ax = axes[1, 0]
    combined = n_time * n_freq
    ax.scatter(bl_km, combined, s=5, alpha=0.5, color="C2")
    ax.set_xlabel("Baseline length [km]")
    ax.set_ylabel("Combined compression factor")
    ax.set_title("Per-Baseline Compression (time × freq)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # (d) UV coverage after BDA
    ax = axes[1, 1]
    u_kl = result["u_avg"] / 1e3
    v_kl = result["v_avg"] / 1e3
    ax.scatter(u_kl, v_kl, s=0.1, alpha=0.3, color="C1", rasterized=True)
    ax.set_xlabel(r"$u$ [k$\lambda$]")
    ax.set_ylabel(r"$v$ [k$\lambda$]")
    ax.set_title(f"UV Coverage after BDA ({len(u_kl)} points)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    compr = result["compression_actual"]
    reduc = (1 - result["total_output_samples"] / result["total_input_unflagged"]) * 100
    fig.suptitle(
        f'BDA | max_t={result["max_avg_time"]}, max_f={result["max_avg_freq"]} | '
        f"{compr:.1f}× compression ({reduc:.1f}% reduction)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="BDA for radio interferometric data")
    parser.add_argument("input_file", type=str, help="Input .mat file")
    parser.add_argument("--max-avg-time", type=int, default=32)
    parser.add_argument("--max-avg-freq", type=int, default=4)
    parser.add_argument(
        "--img-half-width-rad",
        type=float,
        default=None,
        help="Image half-width in radians for frequency smearing cap.",
    )
    parser.add_argument(
        "--smearing-limit",
        type=float,
        default=1.0,
        help="Max phase smearing in radians (default 1.0).  " "Use ~0.1 for narrow-channel accuracy.",
    )
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    data = sio.loadmat(args.input_file)
    result = apply_bda(
        data,
        args.max_avg_time,
        args.max_avg_freq,
        img_half_width_rad=args.img_half_width_rad,
        smearing_limit=args.smearing_limit,
    )

    if args.plot:
        plot_bda_diagnostics(result, save_path=args.plot)

    if args.output:
        np.savez_compressed(
            args.output,
            u_avg=result["u_avg"],
            v_avg=result["v_avg"],
            n_time=result["n_time"],
            n_freq=result["n_freq"],
            bl_lengths=result["bl_lengths"],
            frequencies=result["frequencies"],
        )
        print(f"Saved: {args.output}")

    return result


# ──────────────────────────────────────────────────────────────────────
# Usage Example
# ──────────────────────────────────────────────────────────────────────
"""
Example workflow for BDA with dirty image comparison:

    # 1. Compute BDA bins and average u,v coordinates
    result = apply_bda(
        data_file,
        max_avg_time=32,
        max_avg_freq=16,
        img_half_width_rad=img_half_width_rad,
        smearing_limit=0.5
    )
    
    # 2. Load visibilities in correct order
    vis_classical = load_visibilities(data_file, result)
    
    # 3. Load u,v coordinates in correct order
    u_classical, v_classical = load_uv_coordinates(data_file, result)
    
    # 4. Average visibilities with BDA
    vis_bda_sum, weights_bda = average_visibilities(vis_classical, None, result)
    
    # 5. Use in dirty image computation
    # For classical: use vis_classical with u_classical, v_classical
    # For BDA: use vis_bda_sum (weighted sums) with result['u_avg'], result['v_avg']
    
    # The weighted sums preserve the correct dirty image scale:
    # adjoint(sum(vis_i at u_i)) ≈ adjoint(vis_bda_sum at u_avg)
"""


if __name__ == "__main__":
    main()
