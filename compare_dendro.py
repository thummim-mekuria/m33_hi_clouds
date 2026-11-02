#!/usr/bin/env python
"""Compare parallel_dendro.compute() against real astrodendro on the local
M33 cube (or a crop of it).

    python compare_dendro.py --crop 40 40 40 --min-value 3 --min-delta 2 --min-npix 3
    python compare_dendro.py --full --min-value 3 --min-delta 2 --min-npix 3

Reports: whether index_map matches exactly, whether per-structure
(vmin, vmax, npix, is_leaf, parent, children) match after canonical
relabeling, and -- since exact value ties are a known, expected source of
disagreement (see parallel_dendro's module docstring) -- how many voxels
in the tested region sit in a tied value group, to contextualize any
mismatch found.
"""
import argparse
import time

import numpy as np
from astropy.io import fits
from astrodendro import Dendrogram

import parallel_dendro as pd

CUBE = "M33/data/m33_C+D+tp_himidres_120pc.fits"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tie_fraction(data, keep):
    """Fraction of kept voxels whose exact value is shared by >1 kept voxel."""
    v = data[keep]
    if v.size == 0:
        return 0.0
    _, counts = np.unique(v, return_counts=True)
    tied = counts[counts > 1].sum()
    return tied / v.size


def structure_table(stats_or_dendrogram, is_ref):
    if is_ref:
        return {s.idx: (s.is_leaf, round(float(s.vmin), 5), round(float(s.vmax), 5),
                         len(s.values(subtree=False)),
                         s.parent.idx if s.parent else None,
                         tuple(sorted(c.idx for c in s.children)))
                for s in stats_or_dendrogram}
    return {s["idx"]: (s["is_leaf"], round(s["vmin"], 5), round(s["vmax"], 5), s["npix"],
                        s["parent"], tuple(sorted(s["children"])))
            for s in stats_or_dendrogram}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, nargs=3, metavar=("NZ", "NY", "NX"),
                     help="use only the first NZ,NY,NX voxels along each axis")
    ap.add_argument("--full", action="store_true", help="use the whole local cube")
    ap.add_argument("--min-value", type=float, default=None)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--min-npix", type=int, default=0)
    ap.add_argument("--cube", type=str, default=CUBE, help="path to the FITS cube to test")
    args = ap.parse_args()
    if not args.full and not args.crop:
        ap.error("pass --crop NZ NY NX or --full")

    log(f"loading {args.cube}")
    data = fits.getdata(args.cube).astype(np.float64)
    if args.crop:
        # Centered crop: the cube has NaN margins outside the mosaic
        # footprint, so a corner-anchored crop can land entirely off-source.
        nz, ny, nx = args.crop
        mz, my_, mx = (s // 2 for s in data.shape)
        data = data[mz - nz // 2:mz + (nz + 1) // 2,
                    my_ - ny // 2:my_ + (ny + 1) // 2,
                    mx - nx // 2:mx + (nx + 1) // 2]
    log(f"working shape {data.shape}, {data.size:,} voxels")

    finite = np.isfinite(data)
    min_value = args.min_value
    if min_value is None:
        min_value = float(np.min(data[finite])) - 1.0
    keep = finite & (data > min_value)
    log(f"min_value={min_value:.4g}  kept {keep.sum():,} / {data.size:,} voxels "
        f"({100 * keep.sum() / data.size:.2f}%)")
    tf = tie_fraction(data, keep)
    log(f"fraction of kept voxels in an exact-value tie group: {tf:.4%} "
        "(ties are a known, benign source of tie-break disagreement -- see "
        "parallel_dendro's module docstring)")

    log("running astrodendro (reference)...")
    t0 = time.time()
    d = Dendrogram.compute(data, min_value=min_value, min_delta=args.min_delta,
                            min_npix=args.min_npix)
    log(f"astrodendro: {len(d)} structures in {time.time() - t0:.1f}s")

    log("running parallel_dendro...")
    t0 = time.time()
    my_map, my_stats = pd.compute(data, min_value=min_value, min_delta=args.min_delta,
                                   min_npix=args.min_npix)
    log(f"parallel_dendro: {len(my_stats)} structures in {time.time() - t0:.1f}s")

    map_match = np.array_equal(d.index_map, my_map)
    n_diff = int(np.sum(d.index_map != my_map))
    log(f"index_map identical: {map_match}" + ("" if map_match else
        f"  ({n_diff:,} / {data.size:,} voxels differ, {100 * n_diff / data.size:.4f}%)"))

    ref_table = structure_table(d, is_ref=True)
    my_table = structure_table(my_stats, is_ref=False)
    same_ids = set(ref_table) == set(my_table)
    n_stat_diff = sum(1 for k in ref_table if k in my_table and ref_table[k] != my_table[k])
    log(f"same set of structure ids: {same_ids}  "
        f"(ref n={len(ref_table)}, mine n={len(my_table)})")
    log(f"structures with differing (is_leaf, vmin, vmax, npix, parent, children): "
        f"{n_stat_diff} / {len(ref_table)}")

    if not map_match or n_stat_diff:
        log("first few mismatching structures:")
        shown = 0
        for k in sorted(set(ref_table) | set(my_table)):
            if ref_table.get(k) != my_table.get(k):
                log(f"  idx {k}: ref={ref_table.get(k)}  mine={my_table.get(k)}")
                shown += 1
                if shown >= 10:
                    break

    log("PASS" if (map_match and not n_stat_diff) else "MISMATCH (see counts above)")


if __name__ == "__main__":
    main()
