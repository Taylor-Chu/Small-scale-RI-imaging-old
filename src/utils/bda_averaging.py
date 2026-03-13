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

Performance
-----------
All hot paths are fully vectorised – no Python-level loops over bins.  The
two supported backends are:

  * **NumPy** (default, CPU) – uses ``np.bincount`` for scatter-add.
  * **PyTorch** – pass ``device=torch.device("cuda")`` to use GPU.
    Heavy arrays are built on-GPU; small auxiliary work stays on CPU.

Output format
-------------
Flat vectors of averaged (u, v) coordinates and a function to average
visibilities in the same order.  The ordering follows the lexicographic
sort of the global bin key (bl, f_bin, t_bin) – consistent across all
functions that consume ``result``.
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
import time as timer
import argparse


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
def build_bda_mapping(B, F, V, n_time, n_freq, flag_use):
    """
    Vectorised BDA mapping – **no Python loops over bins**.

    For every unflagged cell (f, t, bl) in the (F, B, V) grid, assigns
    a unique output-bin index.  Output bins are sorted lexicographically
    on the global key  ``bl * max_local + f_bin * n_t_bins[bl] + t_bin``.

    Parameters
    ----------
    B, F, V   : int
    n_time    : (V,) int, per-baseline time-averaging factor
    n_freq    : (V,) int, per-baseline frequency-averaging factor
    flag_use  : (F, B, V) uint8/bool, 1/True = flagged

    Returns
    -------
    mapping : dict with keys
        'bin_inv'      (N_unflagged,) int32 – maps each unflagged entry
                       (in C-order over flag_use) to its output bin index.
        'bin_counts'   (N_out,) int32 – unflagged samples per output bin.
        'bin_bl'       (N_out,) int32
        'bin_t_start'  (N_out,) int32
        'bin_t_end'    (N_out,) int32
        'bin_f_start'  (N_out,) int32
        'bin_f_end'    (N_out,) int32
        'not_flagged'  (F, B, V) bool  – convenience mask
        'N_out'        int
    """
    n_time = np.asarray(n_time, dtype=np.int32)
    n_freq = np.asarray(n_freq, dtype=np.int32)

    # Per-baseline counts of time/freq bins
    n_t_bins = (B + n_time - 1) // n_time  # (V,)
    n_f_bins = (F + n_freq - 1) // n_freq  # (V,)

    # ── Build flattened local bin id for every (f, t, bl) cell ──────
    # t_bin[t, bl] = t // n_time[bl]            shape (B, V)
    # f_bin[f, bl] = f // n_freq[bl]            shape (F, V)
    t_idx = np.arange(B, dtype=np.int32)
    f_idx = np.arange(F, dtype=np.int32)

    t_bin_bv = t_idx[:, None] // n_time[None, :]  # (B, V)
    f_bin_fv = f_idx[:, None] // n_freq[None, :]  # (F, V)

    # local_bin = f_bin * n_t_bins[bl] + t_bin   shape (F, B, V)
    local_bin = (
        f_bin_fv[:, None, :] * n_t_bins[None, None, :] + t_bin_bv[None, :, :]  # (F, 1, V)  # (1, B, V)
    )  # (F, B, V)  int32

    # global_bin = bl * max_local + local_bin   (unique per (f_bin,t_bin,bl))
    max_local = int(np.max(n_f_bins * n_t_bins)) + 1
    bl_idx = np.arange(V, dtype=np.int64)[None, None, :]  # (1,1,V)
    global_bin = bl_idx * max_local + local_bin.astype(np.int64)  # (F,B,V)

    # ── Unflagged mask & bin assignment ─────────────────────────────
    not_flagged = flag_use == 0  # (F, B, V) bool

    bin_vals = global_bin[not_flagged]  # (N_unflagged,) int64

    unique_bins, bin_inv, bin_counts = np.unique(bin_vals, return_inverse=True, return_counts=True)
    bin_inv = bin_inv.astype(np.int32)
    bin_counts = bin_counts.astype(np.int32)
    N_out = len(unique_bins)

    # ── Decode unique_bins → (bl, f_bin, t_bin) ─────────────────────
    bl_out = (unique_bins // max_local).astype(np.int32)  # (N_out,)
    local_out = (unique_bins % max_local).astype(np.int32)  # (N_out,)

    f_bin_out = local_out // n_t_bins[bl_out]  # (N_out,)
    t_bin_out = local_out % n_t_bins[bl_out]  # (N_out,)

    t_start = t_bin_out * n_time[bl_out]
    f_start = f_bin_out * n_freq[bl_out]
    t_end = np.minimum(t_start + n_time[bl_out], B)
    f_end = np.minimum(f_start + n_freq[bl_out], F)

    return {
        "bin_inv": bin_inv,  # (N_unflagged,)
        "bin_counts": bin_counts,  # (N_out,)
        "bin_bl": bl_out,  # (N_out,)
        "bin_t_start": t_start,  # (N_out,)
        "bin_t_end": t_end,  # (N_out,)
        "bin_f_start": f_start,  # (N_out,)
        "bin_f_end": f_end,  # (N_out,)
        "not_flagged": not_flagged,  # (F, B, V)
        "N_out": N_out,
    }


# ──────────────────────────────────────────────────────────────────────
# Internal helpers: scatter-add on CPU (numpy) and GPU (torch)
# ──────────────────────────────────────────────────────────────────────


def _scatter_sum_np(bin_inv, vals, N_out):
    """np.bincount-based scatter-add for real 1-D arrays."""
    return np.bincount(bin_inv, weights=vals.astype(np.float64), minlength=N_out)


def _scatter_sum_np_complex(bin_inv, vals, N_out):
    """Scatter-add for complex 1-D arrays (CPU)."""
    real = np.bincount(bin_inv, weights=vals.real.astype(np.float64), minlength=N_out)
    imag = np.bincount(bin_inv, weights=vals.imag.astype(np.float64), minlength=N_out)
    return real + 1j * imag


def _scatter_sum_torch(bin_inv_np, vals_np, N_out, device, dtype=None):
    """torch.scatter_add-based scatter-add (supports CUDA)."""
    import torch

    if dtype is None:
        dtype = torch.float64
    bin_t = torch.as_tensor(bin_inv_np, dtype=torch.long, device=device)
    vals_t = torch.as_tensor(vals_np, dtype=dtype, device=device)
    out = torch.zeros(N_out, dtype=dtype, device=device)
    out.scatter_add_(0, bin_t, vals_t)
    return out.cpu().numpy()


def _scatter_sum_torch_complex(bin_inv_np, vals_np, N_out, device):
    """torch.scatter_add for complex arrays via real/imag split (CUDA safe)."""
    real = _scatter_sum_torch(bin_inv_np, vals_np.real, N_out, device)
    imag = _scatter_sum_torch(bin_inv_np, vals_np.imag, N_out, device)
    return real + 1j * imag


# ──────────────────────────────────────────────────────────────────────
def apply_bda(
    data,
    flag=None,
    max_avg_time=32,
    max_avg_freq=4,
    img_half_width_rad=None,
    smearing_limit=1.0,
    verbose=True,
    device=None,
):
    """
    Compute BDA bins and average (u, v) coordinates.

    The output u_avg, v_avg are flat vectors (N_out,) in units of
    wavelengths. Flagged samples are excluded from the average (never
    contribute). Each output element corresponds to exactly one BDA bin.

    All heavy computation is vectorised (no Python loops over bins).
    Pass ``device=torch.device("cuda")`` to offload scatter-add to GPU.

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
    device             : torch.device or None.  When not None, scatter-add
                         operations are executed on this device (e.g.
                         ``torch.device("cuda")``).  Requires PyTorch.

    Returns
    -------
    result : dict
    """
    t0 = timer.time()

    # ── Unpack ──
    u_m = data["u"].squeeze()  # (V*B,) meters
    v_m = data["v"].squeeze()
    if "flag" not in data:
        assert flag is not None
    else:
        flag = data["flag"]  # (2, F, V*B)  or similar
    freqs = data["frequency"].squeeze()  # (F,)
    Q = 27
    V = Q * (Q - 1) // 2  # 351
    F = data["frequency"].size  # e.g. 4 or 64
    B = int(flag.shape[-1] / V)  # number of snapshots

    if verbose:
        print(f"Data: Q={Q}, V={V}, B={B}, F={F}")
        print(f"Frequencies: {freqs / 1e9} GHz")

    # ── Reshape ──
    u_2d = u_m.reshape(B, V)  # (B, V) meters
    v_2d = v_m.reshape(B, V)
    flag_4d = flag.reshape(1, F, B, V)
    flag_use = flag_4d[0]  # (F, B, V)

    # ── Baseline lengths & averaging factors ──
    bl_lengths = compute_baseline_lengths(u_m, v_m, Q, B)

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

    # ── Build vectorised mapping (no per-bin Python loop) ────────────
    mapping = build_bda_mapping(B, F, V, n_time, n_freq, flag_use)
    N_out = mapping["N_out"]
    bin_inv = mapping["bin_inv"]  # (N_unflagged,)
    bin_counts = mapping["bin_counts"]  # (N_out,)
    not_flagged = mapping["not_flagged"]  # (F, B, V) bool

    total_input_unflagged = int(np.sum(not_flagged))
    actual_compression = total_input_unflagged / N_out

    if verbose:
        print(f"BDA output: {N_out} bins (from {total_input_unflagged} unflagged inputs)")
        print(
            f"Compression: {actual_compression:.2f}×, "
            f"reduction: {(1 - N_out / total_input_unflagged) * 100:.1f}%"
        )

    # ── Vectorised average of (u, v) in wavelengths ──────────────────
    # u_3d[f, t, bl] = u_2d[t, bl] * freqs[f] / C       (wavelengths)
    u_3d = u_2d[None, :, :] * (freqs[:, None, None] / C_LIGHT)  # (F,B,V)
    v_3d = v_2d[None, :, :] * (freqs[:, None, None] / C_LIGHT)

    u_vals = u_3d[not_flagged]  # (N_unflagged,)
    v_vals = v_3d[not_flagged]

    if device is not None:
        u_sum = _scatter_sum_torch(bin_inv, u_vals, N_out, device)
        v_sum = _scatter_sum_torch(bin_inv, v_vals, N_out, device)
    else:
        u_sum = _scatter_sum_np(bin_inv, u_vals, N_out)
        v_sum = _scatter_sum_np(bin_inv, v_vals, N_out)

    counts_f = bin_counts.astype(np.float64)
    u_avg = u_sum / counts_f
    v_avg = v_sum / counts_f

    elapsed = timer.time() - t0
    if verbose:
        print(f"Done in {elapsed:.1f}s.")

    return {
        "u_avg": u_avg,  # (N_out,) wavelengths
        "v_avg": v_avg,  # (N_out,) wavelengths
        # Bin metadata arrays (replaces list-of-dicts 'bins')
        "bin_inv": bin_inv,  # (N_unflagged,) → output bin index
        "bin_counts": bin_counts,  # (N_out,) unflagged samples per bin
        "bin_bl": mapping["bin_bl"],
        "bin_t_start": mapping["bin_t_start"],
        "bin_t_end": mapping["bin_t_end"],
        "bin_f_start": mapping["bin_f_start"],
        "bin_f_end": mapping["bin_f_end"],
        "not_flagged": not_flagged,  # (F, B, V)
        # Legacy fields
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
def average_visibilities(vis_flat, data, result, device=None):
    """
    Average a flat visibility vector using the BDA mapping from `apply_bda`.

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

    Flagged entries never contribute to the average.  The output is
    the **sum** of unflagged visibilities per bin (= mean × weight) so it
    can be passed directly to the adjoint NUFFT to produce a correct
    dirty image.

    Parameters
    ----------
    vis_flat : (N_unflagged,) complex vector of unflagged visibilities.
    data     : dict from scipy.io.loadmat (needs 'flag' and 'nFreqs').
    result   : dict returned by `apply_bda`.
    device   : torch.device or None (GPU scatter-add when not None).

    Returns
    -------
    vis_avg_weighted : (N_out,) complex, sum of unflagged visibilities per bin.
                       Same ordering as result['u_avg'] / result['v_avg'].
    weights          : (N_out,) int, number of unflagged samples per bin.
    """
    bin_inv = result["bin_inv"]  # (N_unflagged,)
    bin_counts = result["bin_counts"]  # (N_out,)
    N_out = result["total_output_samples"]

    # vis_flat entries align with bin_inv: both enumerate not_flagged in
    # C-order (f outer, b middle, v inner) – same as the classical
    # concatenation convention used by the measurement operator.
    vis_flat = np.asarray(vis_flat)
    if not np.iscomplexobj(vis_flat):
        vis_flat = vis_flat.astype(np.complex128)

    if device is not None:
        vis_sum = _scatter_sum_torch_complex(bin_inv, vis_flat, N_out, device)
    else:
        vis_sum = _scatter_sum_np_complex(bin_inv, vis_flat, N_out)

    return vis_sum, bin_counts.astype(int)


# ──────────────────────────────────────────────────────────────────────
def average_natural_weights(natural_weights_sqrt, result, device=None):
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
    natural_weights_sqrt : (F, B, V) float array, containing sqrt(w) values.
    result               : dict returned by `apply_bda`.
    device               : torch.device or None.

    Returns
    -------
    w_avg_sqrt : (N_out,) sqrt of summed weights per BDA bin.
    """
    bin_inv = result["bin_inv"]  # (N_unflagged,)
    N_out = result["total_output_samples"]
    not_flagged = result["not_flagged"]  # (F, B, V)

    natural_weights_sqrt = np.asarray(natural_weights_sqrt, dtype=np.float64)

    # Square → scatter-sum → sqrt
    w_vals = natural_weights_sqrt[not_flagged] ** 2  # (N_unflagged,)

    if device is not None:
        w_sum = _scatter_sum_torch(bin_inv, w_vals, N_out, device)
    else:
        w_sum = _scatter_sum_np(bin_inv, w_vals, N_out)

    return np.sqrt(w_sum)


# ──────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────
def plot_bda_diagnostics(result, save_path=None):
    """Plot BDA compression factors and uv-coverage."""
    import matplotlib

    matplotlib.use("Agg")
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
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA GPU for scatter-add (requires PyTorch + CUDA).",
    )
    args = parser.parse_args()

    device = None
    if args.cuda:
        import torch

        device = torch.device("cuda")

    data = sio.loadmat(args.input_file)
    result = apply_bda(
        data,
        args.max_avg_time,
        args.max_avg_freq,
        img_half_width_rad=args.img_half_width_rad,
        smearing_limit=args.smearing_limit,
        device=device,
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


if __name__ == "__main__":
    main()
