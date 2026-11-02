#!/usr/bin/env python
"""Prototype: Phases A (basin labeling) and B (saddle graph) as tensor ops,
to measure whether porting the already-vectorized parts of parallel_dendro
to a GPU/tensor backend is actually worth it, before attempting the much
harder problem of Phase D (the sequential tree build).

This does NOT touch Phase D at all -- it only re-implements the two fully
data-parallel phases and checks (a) correctness against parallel_dendro's
NumPy versions and (b) wall-clock time on whatever backend is available
(mps/cuda/cpu).
"""
import time
import numpy as np
import torch

import parallel_dendro as pd


def label_basins_torch(data_t, keep_t, device):
    """Same algorithm as parallel_dendro._label_basins, as tensor ops."""
    ndim = data_t.dim()
    dpad = torch.nn.functional.pad(data_t, [1, 1] * ndim, value=-float("inf"))
    kpad = torch.nn.functional.pad(keep_t, [1, 1] * ndim, value=False)
    flat_ids = torch.arange(dpad.numel(), device=device, dtype=torch.int64).reshape(dpad.shape)

    interior = tuple(slice(1, -1) for _ in range(ndim))
    best_val = dpad[interior].clone()
    best_id = flat_ids[interior].clone()

    for ax in range(ndim):
        for d in (1, -1):
            sl = list(interior)
            lo = 1 + d
            hi = dpad.shape[ax] - 1 + d
            sl[ax] = slice(lo, hi)
            sl = tuple(sl)
            nb_val, nb_id, nb_keep = dpad[sl], flat_ids[sl], kpad[sl]
            better = nb_keep & ((nb_val > best_val) | ((nb_val == best_val) & (nb_id < best_id)))
            best_val = torch.where(better, nb_val, best_val)
            best_id = torch.where(better, nb_id, best_id)

    ptr = torch.arange(dpad.numel(), device=device, dtype=torch.int64)
    keep_flat = keep_t.reshape(-1)
    ptr[flat_ids[interior].reshape(-1)[keep_flat]] = best_id.reshape(-1)[keep_flat]
    for _ in range(500):  # pointer jumping; bounded to avoid a runaway loop
        nxt = ptr[ptr]
        if torch.equal(nxt, ptr):
            break
        ptr = nxt

    basin_id = torch.full(data_t.shape, -1, device=device, dtype=torch.int64)
    basin_id[keep_t] = ptr[flat_ids[interior][keep_t]]
    return basin_id


def saddle_edges_torch(data_t, keep_t, basin_id_t, device):
    """Returns (a, b, w) tensors: one row per unique adjacent-voxel pair in
    different basins (NOT yet reduced to one edge per basin-pair -- that
    reduction is the part that's awkward on a GPU, see the writeup)."""
    ndim = data_t.dim()
    all_a, all_b, all_w = [], [], []
    for ax in range(ndim):
        sl_a = [slice(None)] * ndim
        sl_b = [slice(None)] * ndim
        sl_a[ax] = slice(0, -1)
        sl_b[ax] = slice(1, None)
        sl_a, sl_b = tuple(sl_a), tuple(sl_b)
        ka, kb = keep_t[sl_a], keep_t[sl_b]
        ba, bb = basin_id_t[sl_a], basin_id_t[sl_b]
        va, vb = data_t[sl_a], data_t[sl_b]
        mask = ka & kb & (ba != bb)
        if not torch.any(mask):
            continue
        all_a.append(ba[mask])
        all_b.append(bb[mask])
        all_w.append(torch.minimum(va[mask], vb[mask]))
    return torch.cat(all_a), torch.cat(all_b), torch.cat(all_w)


def main():
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__}, device = {device}")

    from astropy.io import fits
    with fits.open("M33/data/m33_C+D+tp_himidres_120pc.fits", memmap=True) as hdul:
        full = hdul[0].data
        mz, my_, mx = (s // 2 for s in full.shape)
        data = np.asarray(full[mz - 40:mz + 40, my_ - 300:my_ + 300, mx - 300:mx + 300],
                           dtype=np.float32)  # float32: more realistic and lighter on the GPU
    finite = np.isfinite(data)
    keep = finite & (data > 3.0)
    print(f"shape {data.shape}, kept {keep.sum():,}")

    # --- NumPy/CPU baseline (parallel_dendro's actual code) ---
    t0 = time.time()
    basin_id_np, _ = pd._label_basins(data.astype(np.float64), keep)
    t1 = time.time()
    edges_np = pd._saddle_edges(data.astype(np.float64), keep, basin_id_np)
    t2 = time.time()
    print(f"[NumPy/CPU]   labeling {t1 - t0:.3f}s   saddle-reduce {t2 - t1:.3f}s   "
          f"({len(edges_np)} basin-pair edges)")

    # --- Tensor backend ---
    data_t = torch.from_numpy(data).to(device)
    keep_t = torch.from_numpy(keep).to(device)
    if device != "cpu":
        torch.mps.synchronize() if device == "mps" else torch.cuda.synchronize()
    t0 = time.time()
    basin_id_t = label_basins_torch(data_t, keep_t, device)
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    a, b, w = saddle_edges_torch(data_t, keep_t, basin_id_t, device)
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    t2 = time.time()
    print(f"[{device}]   labeling {t1 - t0:.3f}s   raw saddle edges {t2 - t1:.3f}s   "
          f"({a.numel():,} raw adjacent-voxel pairs, before basin-pair reduction)")

    # Correctness: same partition? (compare via basin membership, not raw ids,
    # since the two implementations can legitimately assign different ids to
    # the same physical partition)
    basin_id_t_np = basin_id_t.cpu().numpy()
    same_partition = np.array_equal(
        basin_id_np[keep] == basin_id_np[keep][0],
        basin_id_t_np[keep] == basin_id_t_np[keep][0],
    )
    n_basins_np = len(np.unique(basin_id_np[keep]))
    n_basins_t = len(np.unique(basin_id_t_np[keep]))
    print(f"basin counts: numpy={n_basins_np}  {device}={n_basins_t}  "
          f"(equal partition structure requires a full check, this is a smoke test)")


if __name__ == "__main__":
    main()
