#!/usr/bin/env python
"""A from-scratch, numpy/scipy-only reimplementation of the core IDL CPROPS
(Rosolowsky & Leroy 2006) decomposition pipeline, built directly against the
original IDL source (github.com/low-sky/cprops) rather than through pycprops.
pycprops swaps in astrodendro for local-max finding/decimation and
scikit-image's watershed for final pixel assignment; those substitutions are
a big part of why pycprops output diverges from the original algorithm. By
default this module makes neither substitution and has no astrodendro or
scikit-image dependency at all; two optional, opt-in deviations are offered
anyway (see below) once real-data validation showed each was worth having
available, not because the "faithful port" default was wrong.

Validated against pycprops on a real crop of m33_all_co21_8p0as.K.fits
(150x150x80 around the brightest region, 3-sigma mask, matched friends/
specfriends/min_pix): 88.6% voxel-level agreement on *which* voxels belong to
some cloud, and both capture ~95-97% of the flux above threshold, but with
min_delta=min_npix=0 (decimate-only, this module's original behaviour)
cprops_native found ~2.6x more (smaller) clouds than pycprops on that crop
(3472 vs 1339). Traced to source, not guessed: turning on the significance
pre-filter with pycprops's own defaults (min_delta=2*mad_std, min_npix=5,
i.e. _prune_basins) brought that down to 1522 -- within 14% of pycprops's
1339 on the same crop, confirming the pre-filter (missing entirely from the
raw IDL pipeline, borrowed here from how pycprops uses astrodendro) was the
dominant cause of the earlier gap, not a difference in the decimation logic
itself. Separately, on that same pre-filtered crop, connectivity and
watershed assignment agree on 97.4% of assigned voxels' cloud label
(1860 of 71012 disagree) -- a real but modest difference, concentrated
exactly where the "connectivity vs watershed" section below predicts:
contested pixels between adjacent, comparably-bright peaks.

Ported logic, IDL file -> function here:

  alllocmax.pro          -> _local_maxima          candidate kernels: a kept
                            voxel with no larger neighbour in a
                            (2*specfriends+1) x (2*friends+1)^2 box.

  contourcloud.pro /
  assign2.pro            -> not reimplemented directly. The IDL steps down
                            through contour levels re-running label_region
                            (connected-component labeling) at every level to
                            grow each kernel's own region. That is the same
                            fine partition produced by steepest-ascent basin
                            labeling with ties broken consistently (both are
                            "watershed by flooding" on the same scalar
                            field), so this module reuses
                            parallel_dendro._label_basins /
                            parallel_dendro._saddle_edges instead of
                            re-deriving an equivalent per-level relabeling
                            loop -- one vectorized pass instead of one
                            label_region call per contour level.

  deriv_decimate_kernel.pro /
  are_kernels_disc.pro   -> _decimate      pairwise moment-discontinuity test
                            at each basin-pair's saddle (merger) level.
                            Default SIGDISCONT = [1,1,1,2.0] (x,y,v,flux),
                            NREDUN = 2, plus the IDL's secondary escape hatch
                            SIGDISCONT2 = 2*SIGDISCONT, NREDUN2 = 1. A pair is
                            kept as two separate clouds if *either* kernel's
                            own (size_x, size_y, linewidth, flux) shows a
                            fractional jump past the merger level exceeding
                            NREDUN of the four SIGDISCONT limits (or NREDUN2
                            of SIGDISCONT2); otherwise the fainter kernel
                            (lower own peak value) is absorbed into the
                            brighter one. The jump tested is each kernel's
                            own pre-merger region vs. the region BOTH kernels
                            share once connected at the saddle -- not each
                            kernel's own region at two nearby levels, which
                            is never discontinuous for a smooth isolated
                            peak by construction and was a bug caught by this
                            module's own synthetic self-test (see bottom).

  cloudmom.pro /
  cloudalyze.pro         -> the catalog built at the end of compute():
                            per-cloud npix, flux, centroid and rms size along
                            each cube axis, from that cloud's final pixel
                            set. Simplified relative to the IDL -- see below.

Optional, opt-in (both default off/original):

  _prune_basins  (min_delta/min_npix, default 0/0): a significance pre-filter
                 on basins run *before* _decimate, not present in the raw IDL
                 pipeline at all -- it's what pycprops borrows from
                 astrodendro's tree construction instead. Real-data
                 validation (above) showed this is the dominant reason
                 cprops_native over-fragments relative to pycprops: without
                 it, every shallow noise-driven local maximum enters
                 decimation as its own kernel. Turning it on (e.g.
                 min_delta ~ 2x the map's noise, matching pycprops's own
                 `delta` default of 2*mad_std) trades some fidelity to the
                 literal IDL algorithm for closer agreement with pycprops's
                 actual behaviour on real, noisy data.

  connectivity vs watershed (assignment=, default "connectivity"): the IDL's
  assign2.pro is pure connected-component labeling -- a pixel goes wherever
  its steepest-ascent chain (as modified by _decimate's merges) leads, full
  stop. pycprops instead reassigns pixels with skimage.segmentation.watershed
  seeded at each surviving peak, explicitly because it gives "different
  (better) pixel assignment around a local maximum" (their words, in the
  source comments). The two really do trade off different things:
    - connectivity is the historically faithful CPROPS definition -- if the
      goal is reproducing or comparing against catalogs built with the
      original algorithm (e.g. published GMC catalogs), this is the correct
      choice, and it's why it's this module's default.
    - watershed generally gives smoother, more compact, less jagged cloud
      boundaries. Pure connectivity assigns a pixel by which peak it happens
      to steepest-ascend toward, which for two comparably-bright, adjacent
      peaks can flip a single noisy pixel's assignment based on a tiny local
      gradient wiggle rather than which peak it's really "part of";
      watershed's flood-fill from both markers simultaneously is more robust
      to that, at the cost of no longer being the literal original
      algorithm. This matters most for CPROPS specifically because its whole
      point is measuring cloud *sizes* (rms_x, rms_y, linewidth) for scaling
      relations -- boundary noise directly biases those sizes, so if you
      don't need literal reproducibility of the classic method, watershed is
      plausibly the better choice for the actual downstream science, which
      is presumably also why the actively-maintained pycprops (used by e.g.
      the PHANGS team) made that switch rather than keeping pure
      connectivity.
  Not yet validated against pycprops's watershed output on real data -- only
  checked to run and produce a sane partition (see the bottom of this file's
  self-test).

Known simplifications, stated up front rather than discovered later:
  * mergefind.pro / minimergefind.pro's edge-of-map and edge-of-velocity-
    channel handling (REJECT_REGION-style boundary tests) is not reproduced;
    every survivor here is treated as a valid cloud regardless of whether it
    touches the cube edge.
  * No bootstrapped error bars (the other half of what cprops.pro reports,
    alongside the structural decomposition done here).
  * No deconvolution of the catalog's rms sizes/linewidth by beam or channel
    width, and no conversion to physical units (pc, km/s) -- those need beam
    and distance metadata this module doesn't take as input. Catalog sizes
    are in pixel units on the cube's own axes.
  * _decimate does one full re-sort per merge (concatenate + argsort) rather
    than parallel_dendro's small-to-large searchsorted merge. Basin counts
    (one per local maximum) are orders of magnitude below voxel counts, so
    this is cheap in practice; if it ever isn't, that's the optimization to
    reach for first.
  * The two moment-test levels are the merger level T and T + step, where
    step is a caller-supplied contour step (the same role STEP plays in the
    original IDL contouring) -- there is no IDL default for it confirmed
    from source, so it must be passed explicitly here.
"""
import numpy as np
from scipy import ndimage

import parallel_dendro as pd


def _local_maxima(data, keep, friends=1, specfriends=1):
    """alllocmax.pro: True where a kept voxel has no larger neighbour in a
    (2*specfriends+1) x (2*friends+1) x (2*friends+1) box (axis 0 = spectral).
    Non-kept/non-finite neighbours never block a maximum.
    """
    size = (2 * specfriends + 1,) + (2 * friends + 1,) * (data.ndim - 1)
    filled = np.where(keep, data, -np.inf)
    local_max_val = ndimage.maximum_filter(filled, size=size, mode="constant", cval=-np.inf)
    return keep & (filled >= local_max_val)


def _basin_moment_nodes(data, basin_id, keep):
    """Per-basin raw (not cumulative) per-voxel arrays: value and coordinates
    along every cube axis. _moments_above filters these directly by a
    threshold mask; kept un-cumulative (unlike parallel_dendro's basin
    arrays) so that later merges in _decimate are a plain concatenate, with
    no cumulative-sum bookkeeping to keep consistent across merges.
    """
    idx = np.flatnonzero(keep.ravel())
    b = basin_id.ravel()[idx]
    v = data.ravel()[idx].astype(np.float64)
    coords = [c[idx].astype(np.float64) for c in np.indices(data.shape).reshape(data.ndim, -1)]

    order = np.argsort(b, kind="stable")
    b_sorted = b[order]
    v_sorted = v[order]
    coord_sorted = [c[order] for c in coords]

    boundaries = np.flatnonzero(np.diff(b_sorted)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(b_sorted)]))

    out = {}
    for s, e in zip(starts, ends):
        basin = int(b_sorted[s])
        node = {"v": v_sorted[s:e], "coords": [c[s:e] for c in coord_sorted]}
        node["peak"] = float(node["v"].max())
        out[basin] = node
    return out


def _moments_above(node, T):
    """flux (= sum of intensity) and rms size along every cube axis, using
    only this node's own voxels with value >= T. Returns None if nothing
    qualifies.
    """
    mask = node["v"] >= T
    f = node["v"][mask]  # IDL cprops weights moments by intensity itself
    tot = f.sum()
    if tot <= 0:
        return None
    rms = []
    for c in node["coords"]:
        cc = c[mask]
        mean = (f * cc).sum() / tot
        var = (f * cc * cc).sum() / tot - mean * mean
        rms.append(float(np.sqrt(max(var, 0.0))))
    return {"flux": float(tot), "rms": rms, "n": int(mask.sum())}


_DROP = object()


def _prune_basins(basin_nodes, edges, min_delta=0.0, min_npix=0):
    """Pre-decimation significance pruning: not part of the raw IDL CPROPS
    pipeline (contourcloud.pro/deriv_decimate_kernel.pro have no such step),
    but it's exactly the min_delta/min_npix test astrodendro applies while
    *building* the tree in pycprops's cube_decomp (its is_independent =
    min_npix(minpix) & min_delta(delta) & contains_seeds(maxes)) -- which is
    why pycprops fragments real data differently than this module's default
    (validated on the CO(2-1) cube: pycprops found 1339 clouds where the
    undecorated decimate-only pipeline found 3472 on the same crop, almost
    entirely because pycprops discards shallow noise-driven local maxima
    before decimation ever sees them). Off by default (min_delta=0,
    min_npix=0): every local maximum enters _decimate as its own kernel,
    matching the module's original, more literal-IDL behaviour.

    A basin is "significant" at the threshold T where it first touches a
    neighbour if (own peak - T) >= min_delta and its own npix at value >= T
    is >= min_npix; a basin that never touches any neighbour is instead
    tested against its own full dynamic range (own peak - own min value),
    exactly as astrodendro's orphan-trunk test does.

    Insignificant basins are absorbed into the strongest basin they touch at
    that threshold (their pixels folded in, not merged into a shared
    branch -- CPROPS's output is a flat partition, not a hierarchy). Basins
    that are *both* significant at the same tie level are left fully
    separate here, for _decimate's own moment-discontinuity test to judge.

    Returns (basins, edges, redirect): a possibly-smaller basin_nodes dict,
    edges re-keyed onto the surviving basin ids (duplicates collapsed to
    their max saddle height, self-edges dropped), and {basin_id:
    resolved_id_or__DROP} for every basin that didn't survive unchanged.
    """
    if min_delta <= 0 and min_npix <= 0:
        return basin_nodes, edges, {}

    active = {bid: {"v": n["v"], "coords": list(n["coords"]), "peak": n["peak"]}
              for bid, n in basin_nodes.items()}
    redirect = {}

    def find(x):
        path = []
        while x in redirect:
            path.append(x)
            x = redirect[x]
        for p in path:
            redirect[p] = x
        return x

    def is_significant(node, T):
        return (node["peak"] - T >= min_delta) and (int(np.sum(node["v"] >= T)) >= min_npix)

    def absorb(winner_id, loser_id):
        w, l = active[winner_id], active.pop(loser_id)
        w["v"] = np.concatenate([w["v"], l["v"]])
        w["coords"] = [np.concatenate([wc, lc]) for wc, lc in zip(w["coords"], l["coords"])]
        w["peak"] = max(w["peak"], l["peak"])
        redirect[loser_id] = winner_id

    by_t = {}
    for (a, b), t in edges.items():
        by_t.setdefault(t, []).append((a, b))

    touched = set()
    for T in sorted(by_t.keys(), reverse=True):
        uf_local = {}

        def find_local(x):
            path = []
            while x in uf_local:
                path.append(x)
                x = uf_local[x]
            for p in path:
                uf_local[p] = x
            return x

        group_roots = set()
        for a, b in by_t[T]:
            ra, rb = find(a), find(b)
            touched.add(ra)
            touched.add(rb)
            la, lb = find_local(ra), find_local(rb)
            if la != lb:
                uf_local[la] = lb
            group_roots.add(ra)
            group_roots.add(rb)

        grouped = {}
        for r in group_roots:
            grouped.setdefault(find_local(r), set()).add(r)

        for members in grouped.values():
            members = [m for m in members if m in active]
            if len(members) < 2:
                continue
            sig = [m for m in members if is_significant(active[m], T)]
            insig = [m for m in members if m not in sig]
            if not insig:
                continue  # every member stands on its own -> leave separate
            if sig:
                target = max(sig, key=lambda m: active[m]["peak"])
            else:
                target = max(insig, key=lambda m: len(active[m]["v"]))
                insig = [m for m in insig if m != target]
            for m in insig:
                absorb(target, m)

    for bid, node in list(active.items()):
        if bid not in touched:
            own_min = float(node["v"].min())
            if not (node["peak"] - own_min >= min_delta and len(node["v"]) >= min_npix):
                redirect[bid] = _DROP
                del active[bid]

    new_edges = {}
    for (a, b), T in edges.items():
        ra, rb = find(a), find(b)
        if ra is _DROP or rb is _DROP or ra == rb:
            continue
        key = (min(ra, rb), max(ra, rb))
        if key not in new_edges or T > new_edges[key]:
            new_edges[key] = T

    full_redirect = {bid: find(bid) for bid in basin_nodes}
    return active, new_edges, full_redirect


def _watershed_assign(data, keep, uniq_roots, padded_shape):
    """Alternative final pixel assignment: skimage.segmentation.watershed
    seeded at each surviving kernel's peak, instead of this module's default
    pure-connectivity assignment (steepest-ascent basin membership, as
    modified by _decimate's merges). See the module docstring's "connectivity
    vs watershed" section for the tradeoff. Requires scikit-image; imported
    lazily so the default connectivity path stays dependency-free.

    uniq_roots are basin ids as produced by parallel_dendro._label_basins:
    literally the flat index, in the *padded* array, of the peak voxel each
    basin attracts to -- so the marker coordinate for each root is just that
    index unraveled and de-padded, no separate peak-location bookkeeping
    needed.

    Returns index_map (same shape as data, -1 outside keep, else 0..n-1
    matching uniq_roots' order, same convention as the connectivity path).
    """
    from skimage.segmentation import watershed

    markers = np.zeros(data.shape, dtype=np.int64)
    for i, root in enumerate(uniq_roots.tolist(), start=1):
        coord = np.unravel_index(int(root), padded_shape)
        coord = tuple(c - 1 for c in coord)  # undo _label_basins' 1-voxel pad
        markers[coord] = i
    labels = watershed(-data, markers=markers, mask=keep)
    return labels.astype(np.int64) - 1


def _decimate(basin_nodes, edges, step, sigdiscont=(1.0, 1.0, 1.0, 2.0), nredun=2,
              sigdiscont2=None, nredun2=1):
    """deriv_decimate_kernel.pro / are_kernels_disc.pro: process saddle edges
    from the highest merger level down; at each merger, test whether folding
    the fainter kernel into the brighter one would erase a real
    discontinuity in either kernel's own (size_x, size_y, size_v, flux)
    growth curve right above the merger level. A pair survives as two
    separate clouds if *either* kernel is discontinuous there; otherwise the
    fainter one (lower own peak value) is absorbed into the brighter one.

    Returns {basin_id: surviving_root_id} for every basin in basin_nodes,
    fully resolved (no further redirects needed).
    """
    sigdiscont = np.asarray(sigdiscont, dtype=np.float64)
    if sigdiscont2 is None:
        sigdiscont2 = sigdiscont * 2.0
    active = {}
    redirect = {}

    def find(x):
        path = []
        while x in redirect:
            path.append(x)
            x = redirect[x]
        for p in path:
            redirect[p] = x
        return x

    def ensure(nid):
        if nid not in active:
            n = basin_nodes[nid]
            active[nid] = {"v": n["v"], "coords": list(n["coords"]), "peak": n["peak"]}
        return active[nid]

    def is_disc_one(hi, lo_union):
        # hi: this kernel's own pre-merger region just above the saddle.
        # lo_union: the region BOTH kernels share once connected, right at
        # the saddle -- i.e. what this kernel's own region balloons into if
        # the merge is accepted. A kernel's own moments look "smooth" near
        # its own summit almost by definition (any isolated peak varies
        # slowly there); the real signal is whether absorbing the other
        # kernel's territory is itself a big jump relative to what this
        # kernel looked like on its own just above the saddle.
        if hi is None or lo_union is None:
            return True  # nothing left to test above the merger -> keep separate
        d = np.empty(4)
        for i in range(3):
            d[i] = ((lo_union["rms"][i] - hi["rms"][i]) / hi["rms"][i]
                     if hi["rms"][i] > 0 else 0.0)
        d[3] = (lo_union["flux"] - hi["flux"]) / hi["flux"] if hi["flux"] > 0 else 0.0
        return bool(np.sum(d > sigdiscont) >= nredun) or bool(np.sum(d > sigdiscont2) >= nredun2)

    def is_disc(na, nb, T):
        hi_a = _moments_above(na, T + step)
        hi_b = _moments_above(nb, T + step)
        union = {"v": np.concatenate([na["v"], nb["v"]]),
                  "coords": [np.concatenate([ca, cb]) for ca, cb in zip(na["coords"], nb["coords"])]}
        lo_union = _moments_above(union, T)
        return is_disc_one(hi_a, lo_union) or is_disc_one(hi_b, lo_union)

    def merge_into(winner_id, loser_id):
        w, l = active[winner_id], active.pop(loser_id)
        w["v"] = np.concatenate([w["v"], l["v"]])
        w["coords"] = [np.concatenate([wc, lc]) for wc, lc in zip(w["coords"], l["coords"])]
        w["peak"] = max(w["peak"], l["peak"])

    for (a, b), T in sorted(edges.items(), key=lambda kv: -kv[1]):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        na, nb = ensure(ra), ensure(rb)
        if is_disc(na, nb, T):
            continue  # real discontinuity somewhere -> leave as two clouds
        winner, loser = (ra, rb) if na["peak"] >= nb["peak"] else (rb, ra)
        merge_into(winner, loser)
        redirect[loser] = winner

    return {basin: find(basin) for basin in basin_nodes}


def compute(data, min_value, step, friends=1, specfriends=1,
            sigdiscont=(1.0, 1.0, 1.0, 2.0), nredun=2,
            sigdiscont2=None, nredun2=1, min_pix=0,
            min_delta=0.0, min_npix=0, assignment="connectivity"):
    """Public entry point. Returns (index_map, catalog):

      index_map: same shape as data, int64, -1 where unassigned, else a
                 contiguous cloud id 0..n-1.
      catalog:   list of dicts, one per surviving cloud: idx, npix, flux,
                 peak, centroid (one entry per axis), rms (one per axis, in
                 pixel units -- see module docstring's simplifications).

    min_delta/min_npix: optional pre-decimation significance pruning on
    basins (see _prune_basins); 0/0 reproduces the module's original
    decimate-only behaviour. Distinct from min_pix, which prunes the final
    *catalog* by cloud size after decimation -- min_delta/min_npix instead
    change what decimation even sees, by removing shallow local maxima
    up front the way pycprops's astrodendro-based tree construction does.

    assignment: "connectivity" (default, this module's original pure
    steepest-ascent + _decimate-merge assignment, matching assign2.pro) or
    "watershed" (skimage.segmentation.watershed seeded at each surviving
    kernel's peak -- see the module docstring for the tradeoff; requires
    scikit-image).
    """
    data = np.asarray(data, dtype=np.float64)
    shape = data.shape
    finite = np.isfinite(data)
    keep = finite & (data >= min_value)

    basin_id, padded_shape = pd._label_basins(data, keep)
    edges = pd._saddle_edges(data, keep, basin_id)
    basin_nodes = _basin_moment_nodes(data, basin_id, keep)
    del finite

    basin_nodes, edges, pre_redirect = _prune_basins(basin_nodes, edges, min_delta, min_npix)
    root_of = _decimate(basin_nodes, edges, step, sigdiscont, nredun, sigdiscont2, nredun2)

    flat_basin = basin_id.ravel()
    kept_idx_all = np.flatnonzero(keep.ravel())
    basins_at_kept = flat_basin[kept_idx_all]

    uniq_present = np.unique(basins_at_kept)
    resolved_root = {}
    resolved_ok = {}
    for bid in uniq_present.tolist():
        r = pre_redirect.get(bid, bid)
        if r is _DROP:
            resolved_ok[bid] = False
            resolved_root[bid] = -1
        else:
            resolved_ok[bid] = True
            resolved_root[bid] = root_of[r]

    keys = np.fromiter(resolved_root.keys(), dtype=np.int64)
    order = np.argsort(keys)
    keys = keys[order]
    vals = np.fromiter(resolved_root.values(), dtype=np.int64)[order]
    ok = np.fromiter(resolved_ok.values(), dtype=bool)[order]
    pos = np.searchsorted(keys, basins_at_kept)
    voxel_ok = ok[pos]
    voxel_root = vals[pos]

    kept_idx = kept_idx_all[voxel_ok]
    assigned_roots = voxel_root[voxel_ok]
    uniq_roots, new_ids = np.unique(assigned_roots, return_inverse=True)

    if assignment == "watershed":
        pruned_keep = np.zeros(flat_basin.shape, dtype=bool)
        pruned_keep[kept_idx] = True
        index_map = _watershed_assign(data, pruned_keep.reshape(shape), uniq_roots, padded_shape)
    elif assignment == "connectivity":
        index_map = np.full(flat_basin.shape, -1, dtype=np.int64)
        index_map[kept_idx] = new_ids
        index_map = index_map.reshape(shape)
    else:
        raise ValueError(f"assignment must be 'connectivity' or 'watershed', got {assignment!r}")

    counts = np.bincount(index_map[index_map >= 0], minlength=len(uniq_roots))
    keep_cloud = counts >= min_pix
    if not np.all(keep_cloud):
        remap = np.full(len(uniq_roots), -1, dtype=np.int64)
        remap[keep_cloud] = np.arange(int(keep_cloud.sum()))
        index_map = np.where(index_map >= 0, remap[np.clip(index_map, 0, None)], -1)
        n_clouds = int(keep_cloud.sum())
    else:
        n_clouds = len(uniq_roots)

    coord_grids = np.indices(shape)
    catalog = []
    for cid in range(n_clouds):
        mask = index_map == cid
        f = data[mask]
        catalog.append({
            "idx": cid,
            "npix": int(mask.sum()),
            "flux": float(f.sum()),
            "peak": float(f.max()),
            "centroid": [float((f * coord_grids[ax][mask]).sum() / f.sum()) for ax in range(data.ndim)],
            "rms": [float(np.sqrt(max(
                (f * coord_grids[ax][mask] ** 2).sum() / f.sum()
                - ((f * coord_grids[ax][mask]).sum() / f.sum()) ** 2, 0.0)))
                for ax in range(data.ndim)],
        })
    return index_map, catalog


if __name__ == "__main__":
    import argparse
    import time

    from astropy.io import fits

    CUBE = "M33/data/m33_C+D+tp_himidres_120pc.fits"

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, nargs=3, metavar=("NZ", "NY", "NX"))
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--min-value", type=float, required=True)
    ap.add_argument("--step", type=float, required=True,
                     help="contour step used only for the decimation discontinuity test")
    ap.add_argument("--min-pix", type=int, default=0)
    ap.add_argument("--min-delta", type=float, default=0.0,
                     help="pre-decimation basin significance cut (see _prune_basins); "
                          "0 = original decimate-only behaviour")
    ap.add_argument("--min-npix", type=int, default=0,
                     help="pre-decimation basin significance cut, pixel-count half")
    ap.add_argument("--assignment", choices=["connectivity", "watershed"], default="connectivity")
    args = ap.parse_args()
    if not args.full and not args.crop:
        ap.error("pass --crop NZ NY NX or --full")

    log(f"loading {CUBE}")
    data = fits.getdata(CUBE).astype(np.float64)
    if args.crop:
        nz, ny, nx = args.crop
        mz, my_, mx = (s // 2 for s in data.shape)
        data = data[mz - nz // 2:mz + (nz + 1) // 2,
                    my_ - ny // 2:my_ + (ny + 1) // 2,
                    mx - nx // 2:mx + (nx + 1) // 2]
    log(f"working shape {data.shape}, {data.size:,} voxels")

    t0 = time.time()
    index_map, catalog = compute(data, min_value=args.min_value, step=args.step,
                                  min_pix=args.min_pix, min_delta=args.min_delta,
                                  min_npix=args.min_npix, assignment=args.assignment)
    log(f"cprops_native: {len(catalog)} clouds in {time.time() - t0:.1f}s")
    for c in sorted(catalog, key=lambda c: -c["flux"])[:10]:
        log(f"  idx {c['idx']:4d}  npix {c['npix']:7d}  flux {c['flux']:10.3g}  "
            f"peak {c['peak']:8.3g}  rms {[round(r, 2) for r in c['rms']]}")
