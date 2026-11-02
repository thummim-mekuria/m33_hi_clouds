#!/usr/bin/env python
"""A from-scratch, array-based reimplementation of astrodendro's
Dendrogram.compute(), structured so the expensive steps are vectorized
(parallel/GPU-portable) and only a small event list is processed sequentially.

Only reproduces the default is_independent = all_true([min_delta(d), min_npix(n)])
pruning. Custom is_independent functions, custom neighbours(), and non-default
connectivity are not supported.

Design (see the accompanying conversation for the full derivation):

  Phase A - basin labeling (vectorized): every kept voxel follows steepest
            ascent, with ties broken by lowest flat voxel index, to its
            attracting peak. This is the finest possible partition and
            corresponds to astrodendro's initial per-peak leaves.
  Phase B - saddle graph (vectorized): for every pair of adjacent voxels in
            different basins, the saddle height is min(v_i, v_j); reduced by
            max over each unordered basin pair.
  Phase C - per-basin (value, voxel_id) arrays, sorted descending: lets any
            later step answer "how many of this lineage's own voxels have
            value > T" by binary search instead of replaying every voxel.
  Phase D - sequential tree build over the saddle graph (small: edge count,
            not voxel count). Edges are processed in descending threshold
            order, grouped by exact tie value so an N-way simultaneous
            convergence (real risk here: the cube's histogram showed value
            quantization) is resolved as one N-way event, matching
            astrodendro's simultaneous multi-neighbour test rather than as
            several sequential pairwise ones.
  Phase E - orphan-trunk pruning: a root that never touched anything is
            tested with the *other* branch of min_delta (its own vmax-vmin),
            exactly as astrodendro's _make_trunk does; if it fails, it is
            dropped (index_map = -1), not merged into anything.
  Final    - relabel surviving structures by ascending smallest flat voxel
            index, matching astrodendro's own convention, so a correct
            implementation should produce a *literally identical* index_map,
            not just an equivalent partition up to renumbering.

Known, accepted limitation: internal tie-breaking for *which* structure
survives when two equally-insignificant leaves merge with nothing else
adjacent (astrodendro's `merge.pop()`) is not reproduced bit-for-bit. This
cannot change the final partition or any structure's aggregate stats (both
losers' pixels end up in the same place either way) -- only which of several
otherwise-interchangeable structure identities is nominally kept alive
mid-construction, which astrodendro itself does not treat as meaningful.
"""
import numpy as np


def _neighbor_shifts(ndim):
    """(axis, direction) pairs for 2*ndim face-connectivity neighbours."""
    shifts = []
    for ax in range(ndim):
        for d in (1, -1):
            shifts.append((ax, d))
    return shifts


def _index_dtype(n):
    """Smallest signed int dtype that can hold every value in range(n).

    flat_ids/ptr/best_id/basin_id all store flat voxel indices into the
    *padded* array, and were previously hardcoded int64 regardless of how
    many voxels that actually needs representing. At real-cube scale that
    is the dominant memory cost in _label_basins: several full-padded-size
    int64 arrays (flat_ids, ptr, and ptr's replacement each pointer-jump
    iteration) alive at once, 8 bytes/voxel each, on top of the data and
    keep arrays -- measured at 12.3 GB *per array* for a 1.5-billion-voxel
    cube, enough to OOM a 24 GB machine before Phase A even finishes.
    int32 (max ~2.1 billion) covers any cube that fits in memory as a dense
    array at all in the first place, so this is free: same values, same
    comparisons, half the bytes, with a correctness-preserving fallback to
    int64 for the (currently unreachable in practice) case of a padded
    array bigger than int32 can index.
    """
    return np.int32 if n <= np.iinfo(np.int32).max else np.int64


def _label_basins(data, keep):
    """Phase A. Returns basin_id, a flat-index (into the *padded* array)
    array of shape data.shape: every kept voxel maps to the flat id of the
    peak (or plateau leader) its steepest-ascent chain terminates at.
    Non-kept voxels get -1.
    """
    ndim = data.ndim
    pad = [(1, 1)] * ndim
    dpad = np.pad(data, pad, mode="constant", constant_values=-np.inf)
    kpad = np.pad(keep, pad, mode="constant", constant_values=False)
    padded_shape = dpad.shape
    idx_dtype = _index_dtype(dpad.size)
    flat_ids = np.arange(dpad.size, dtype=idx_dtype).reshape(dpad.shape)

    interior = tuple(slice(1, -1) for _ in range(ndim))
    best_val = dpad[interior].copy()
    best_id = flat_ids[interior].copy()

    for ax, d in _neighbor_shifts(ndim):
        sl = list(interior)
        lo = 1 + d
        hi = dpad.shape[ax] - 1 + d
        sl[ax] = slice(lo, hi)
        sl = tuple(sl)
        nb_val = dpad[sl]
        nb_id = flat_ids[sl]
        nb_keep = kpad[sl]
        better = nb_keep & ((nb_val > best_val) | ((nb_val == best_val) & (nb_id < best_id)))
        best_val = np.where(better, nb_val, best_val)
        best_id = np.where(better, nb_id, best_id)

    # dpad/kpad/best_val are not read again after this point (best_id has
    # already absorbed everything needed from them) -- freed here, before
    # the pointer-jump phase allocates its own full-padded-size arrays, to
    # keep several full-cube-sized arrays from being live at once for no
    # reason.
    del dpad, kpad, best_val

    # Pointer-jump to the fixed point over the *global* (padded) id space.
    ptr = np.arange(flat_ids.size, dtype=idx_dtype)
    ptr[flat_ids[interior][keep]] = best_id[keep]
    while True:
        nxt = ptr[ptr]
        if np.array_equal(nxt, ptr):
            break
        ptr = nxt

    basin_id = np.full(data.shape, -1, dtype=idx_dtype)
    basin_id[keep] = ptr[flat_ids[interior][keep]]
    return basin_id, padded_shape


def _saddle_edges(data, keep, basin_id):
    """Phase B. Returns dict {(lo_basin, hi_basin): saddle_height}."""
    ndim = data.ndim
    edges = {}
    for ax, d in _neighbor_shifts(ndim):
        if d != 1:
            continue  # each unordered pair of adjacent voxels only once
        sl_a = [slice(None)] * ndim
        sl_b = [slice(None)] * ndim
        sl_a[ax] = slice(0, -1)
        sl_b[ax] = slice(1, None)
        sl_a, sl_b = tuple(sl_a), tuple(sl_b)

        ka, kb = keep[sl_a], keep[sl_b]
        ba, bb = basin_id[sl_a], basin_id[sl_b]
        va, vb = data[sl_a], data[sl_b]
        mask = ka & kb & (ba != bb)
        if not np.any(mask):
            continue
        a, b = ba[mask], bb[mask]
        w = np.minimum(va[mask], vb[mask])
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        for l, h, wv in zip(lo.tolist(), hi.tolist(), w.tolist()):
            key = (l, h)
            if key not in edges or wv > edges[key]:
                edges[key] = wv
    return edges


def _basin_value_voxel_arrays(data, basin_id, keep):
    """Phase C. dict {basin_flat_id: (values_desc, voxel_flat_ids_desc)}.
    voxel_flat_ids are flat indices into the *unpadded* data array (i.e.
    ordinary np.ravel indices), used to write the final labeling.
    """
    idx = np.flatnonzero(keep.ravel())
    if idx.size == 0:
        return {}
    b = basin_id.ravel()[idx]
    v = data.ravel()[idx]
    order = np.argsort(b, kind="stable")
    b_sorted, v_sorted, idx_sorted = b[order], v[order], idx[order]
    boundaries = np.flatnonzero(np.diff(b_sorted)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(b_sorted)]))

    out = {}
    for s, e in zip(starts, ends):
        basin = int(b_sorted[s])
        vv = v_sorted[s:e]
        ii = idx_sorted[s:e]
        o = np.argsort(vv)[::-1]  # descending
        out[basin] = (vv[o].astype(np.float64), ii[o])
    return out


def _find(redirect, x):
    """Union-find with path compression."""
    path = []
    while x in redirect:
        path.append(x)
        x = redirect[x]
    for p in path:
        redirect[p] = x
    return x


def _concat_sorted(a_v, a_i, b_v, b_i):
    """Merge two already-descending-sorted (value, id) arrays into one
    descending-sorted array, without re-sorting everything from scratch.

    The original version did np.concatenate + a full argsort of the
    combined array: O((n+m) log(n+m)) every call, and -- since one array is
    typically a small leaf being folded into a much bigger accumulated
    structure -- that cost is paid at *full combined size* on every merge a
    structure ever takes part in, which is what made Phase D dominate both
    the runtime and memory profile (each call also allocates a fresh
    concatenated array purely to sort it, then discards it).

    This version finds where the *smaller* array's elements insert into the
    *larger* one via a single vectorized searchsorted call -- O(m log n) for
    the search, plus one O(n+m) pass to write the output (unavoidable: the
    result itself has n+m elements). For the common case of a tiny leaf
    merging into a large accumulated branch (m << n), that replaces an
    O(n log n) sort with an O(n) copy -- and, summed over a structure's
    whole absorption history, that is the small-to-large-merge argument
    that makes the total work close to linear rather than n log n.

    (Two alternatives were tried and measured worse on real data, worth
    recording so they aren't retried: (1) representing each active
    structure as a list of un-merged chunks, consolidated lazily, to avoid
    touching all of `a` on every absorb -- this trades the cost for a
    slower `npix_above` that sums over every chunk, and was slower at every
    scale tested, including the large-crop case it was meant to fix; (2)
    replacing this function's boolean-mask output construction with a
    Python-level loop that interleaves contiguous slices of `a` via
    np.concatenate -- a cProfile run pointed at the boolean-mask assignment
    as the hot line, but the "fix" was slower still, most likely because
    cProfile's own per-call overhead misrepresented a handful of vectorized
    NumPy ops against a Python-level loop building many small arrays. The
    plain searchsorted + boolean-mask version below is what actually
    measured fastest across 120x120x40 (0.87s), 300x300x60 (3.96s), and
    600x600x80 (42.7s) crops of the real cube.)
    """
    if len(a_v) == 0:
        return b_v, b_i
    if len(b_v) == 0:
        return a_v, a_i
    if len(a_v) < len(b_v):
        a_v, a_i, b_v, b_i = b_v, b_i, a_v, a_i  # a is now the larger array
    n, m = len(a_v), len(b_v)
    # Descending order: negate both sides so ascending-searchsorted applies.
    pos = np.searchsorted(-a_v, -b_v, side="right")
    b_dest = pos + np.arange(m)
    out_v = np.empty(n + m, dtype=a_v.dtype)
    out_i = np.empty(n + m, dtype=a_i.dtype)
    out_v[b_dest] = b_v
    out_i[b_dest] = b_i
    a_slot = np.ones(n + m, dtype=bool)
    a_slot[b_dest] = False
    out_v[a_slot] = a_v
    out_i[a_slot] = a_i
    return out_v, out_i


def _build_tree(edges, basin_arrays, min_delta, min_npix, next_id_start):
    """Phase D. Sequential tree build over the (small) saddle-edge list.

    Returns (tree, active, redirect):
      tree: dict node_id -> {is_leaf, own_values, own_voxel_ids, parent, children}
            for every node that ever became a *visible* structure (frozen
            child or created branch).
      active: whatever is still live and un-frozen when edges run out --
              i.e. trunk candidates, handled by the caller (Phase E).
      redirect: the union-find map, needed by the caller to resolve any
                fine basin id (including ones never touched by an edge at
                all -- untouched, isolated basins) to its current root.
    """
    active = {}
    tree = {}
    redirect = {}
    next_id = [next_id_start]

    def ensure_active(nid):
        if nid not in active:
            v, ids = basin_arrays[nid]
            active[nid] = {"v": v, "ids": ids,
                           "own_vmax": float(v[0]) if len(v) else -np.inf,
                           "is_leaf": True}
        return active[nid]

    def npix_above(node, T):
        v = node["v"]
        return int(np.searchsorted(-v, -T, side="left"))

    def insignificant(node, T):
        if not node["is_leaf"]:
            return False  # branches are never merge-candidates
        if node["own_vmax"] == T:
            return True
        return not ((node["own_vmax"] - T >= min_delta) and (npix_above(node, T) >= min_npix))

    def absorb_wholesale(winner_id, loser_ids):
        w = active[winner_id]
        for lid in loser_ids:
            l = active.pop(lid)
            w["v"], w["ids"] = _concat_sorted(w["v"], w["ids"], l["v"], l["ids"])
            w["own_vmax"] = max(w["own_vmax"], l["own_vmax"])
            redirect[lid] = winner_id

    def freeze_into(node_id, T, target_id):
        node = active.pop(node_id)
        v, ids = node["v"], node["ids"]
        cnt = int(np.searchsorted(-v, -T, side="left"))  # count strictly > T
        # .copy() is essential here: v[:cnt] alone is a *view*, and
        # own_values is kept forever in `tree` -- an unretired view would
        # keep the *entire* (possibly much larger) v buffer alive for as
        # long as the program runs, for every frozen structure. That is
        # what was inflating peak memory before this was added: profiling
        # showed _concat_sorted's output buffers were the dominant live
        # allocation, with far more of them resident than the number of
        # final surviving structures could explain.
        own_v, own_ids = v[:cnt].copy(), ids[:cnt].copy()
        rem_v, rem_i = v[cnt:], ids[cnt:]
        entry = tree.setdefault(node_id, {"children": []})
        entry["is_leaf"] = node["is_leaf"]
        entry["own_values"] = own_v
        entry["own_voxel_ids"] = own_ids
        entry["parent"] = target_id
        redirect[node_id] = target_id
        t = active[target_id]
        t["v"], t["ids"] = _concat_sorted(t["v"], t["ids"], rem_v, rem_i)

    by_t = {}
    for (a, b), t in edges.items():
        by_t.setdefault(t, []).append((a, b))

    for T in sorted(by_t.keys(), reverse=True):
        pairs = by_t[T]
        # Union same-threshold pairs first (pure connectivity, no
        # significance test) so an exact N-way tie is resolved as one
        # simultaneous event, matching astrodendro's single-pixel,
        # multi-neighbour test rather than several sequential pairwise ones.
        uf_local = {}

        def find_local(x):
            path = []
            while x in uf_local:
                path.append(x)
                x = uf_local[x]
            for p in path:
                uf_local[p] = x
            return x

        touched = set()
        for a, b in pairs:
            ra, rb = _find(redirect, a), _find(redirect, b)
            ensure_active(ra)
            ensure_active(rb)
            touched.add(ra)
            touched.add(rb)
            la, lb = find_local(ra), find_local(rb)
            if la != lb:
                uf_local[la] = lb

        grouped = {}
        for r in touched:
            grouped.setdefault(find_local(r), set()).add(r)

        for members in grouped.values():
            members = list(members)
            if len(members) < 2:
                continue
            merge_cands = [m for m in members if insignificant(active[m], T)]
            significant = [m for m in members if m not in merge_cands]

            if len(significant) == 0:
                winner = max(merge_cands, key=lambda m: len(active[m]["v"]))
                losers = [m for m in merge_cands if m != winner]
                absorb_wholesale(winner, losers)
            elif len(significant) == 1:
                absorb_wholesale(significant[0], merge_cands)
            else:
                branch_id = next_id[0]
                next_id[0] += 1
                active[branch_id] = {"v": np.empty(0), "ids": np.empty(0, dtype=np.int64),
                                      "own_vmax": T, "is_leaf": False}
                for m in significant:
                    freeze_into(m, T, branch_id)
                if merge_cands:
                    absorb_wholesale(branch_id, merge_cands)
                tree[branch_id] = {"is_leaf": False, "children": list(significant),
                                    "parent": None, "own_values": None, "own_voxel_ids": None}

    return tree, active, redirect


def _finalize(tree, active, redirect, basin_arrays, min_delta, min_npix):
    """Phase E. Every basin that was never dissolved into something else --
    whether it never touched an edge at all, or it's still 'active' (live,
    unfrozen) once the edge list runs out -- becomes a trunk item. A trunk
    *leaf* (is_leaf, and never merged with anything) is tested with the
    *other* branch of min_delta (its own vmax-vmin), exactly as astrodendro's
    _make_trunk does; on failure its pixels are dropped (final id: -1),
    not folded into anything else. Mutates `tree` in place.
    """
    dissolved = set(redirect.keys())
    # Ids that are fully finalized already (frozen children, or branches that
    # later themselves froze into something else) -- these have real
    # own_values, not the None placeholder a branch gets at creation.
    finalized = {k for k, v in tree.items() if v.get("own_values") is not None}

    # Candidates for "still a trunk item": every fine basin (whether or not
    # it was ever touched by an edge) plus every id still live in `active`
    # (this is where a *branch* that survived to the very end shows up --
    # branches never appear in basin_arrays, only in `active`/`tree`).
    candidates = set(basin_arrays.keys()) | set(active.keys())

    for basin_id in candidates:
        if basin_id in dissolved or basin_id in finalized:
            continue
        if basin_id in active:
            node = active[basin_id]
            vv, ii, is_leaf = node["v"], node["ids"], node["is_leaf"]
        else:
            vv, ii = basin_arrays[basin_id]
            is_leaf = True  # never touched by any edge

        if is_leaf:
            own_vmax, own_vmin = float(vv[0]), float(vv[-1])
            npix = len(vv)
            passes = (own_vmax - own_vmin >= min_delta) and (npix >= min_npix)
            if not passes:
                continue  # dropped: no tree entry, pixels stay unassigned (-1)

        tree[basin_id] = {"is_leaf": is_leaf, "own_values": vv, "own_voxel_ids": ii,
                           "parent": None, "children": tree.get(basin_id, {}).get("children", [])}


def _assemble(data_shape, tree):
    """Final relabel: sort surviving structures by ascending smallest own
    voxel id (equivalent to astrodendro's lexicographic smallest_index,
    since a C-order flat index IS the lexicographic (z,y,x,...) order),
    assign contiguous ids 0..n-1, and build index_map + a stats table.
    """
    items = list(tree.items())
    items.sort(key=lambda kv: int(kv[1]["own_voxel_ids"].min()))
    old_to_new = {old_id: i for i, (old_id, _) in enumerate(items)}

    # index_map's values are just the relabeled structure count (0..n-1,
    # n = len(items)), never voxel indices -- but the array itself was
    # hardcoded int64 at full data_shape size regardless, doubling the
    # output's memory for no reason at real-cube scale (12.2 GB vs 6.1 GB
    # for a 1.5-billion-voxel cube). int32 covers any structure count that
    # will ever occur in practice (>2.1 billion distinct structures is not
    # a realistic outcome), with the same int64 fallback guard as Phase A.
    map_dtype = _index_dtype(max(len(items), 1))
    index_map = np.full(int(np.prod(data_shape)), -1, dtype=map_dtype)
    stats = []
    for old_id, entry in items:
        new_id = old_to_new[old_id]
        index_map[entry["own_voxel_ids"]] = new_id
        parent = entry["parent"]
        stats.append({
            "idx": new_id,
            "is_leaf": entry["is_leaf"],
            "vmin": float(entry["own_values"].min()),
            "vmax": float(entry["own_values"].max()),
            "npix": len(entry["own_values"]),
            "parent": old_to_new[parent] if parent is not None else None,
            "children": sorted(old_to_new[c] for c in entry["children"]),
        })
    return index_map.reshape(data_shape), stats


def compute(data, min_value=None, min_delta=0.0, min_npix=0):
    """Public entry point. Returns (index_map, stats) -- see _assemble.

    Frees basin_id/keep/finite (each a full-crop-sized array, ~13 bytes/voxel
    combined at float64+bool+int64) as soon as Phase D no longer needs them,
    since they otherwise stay alive -- doing nothing -- through the whole
    (previously dominant, still largest) tree-build step. Note this only
    reduces *this function's own* peak: `data` itself is deleted too once
    only its .shape is still needed, but if the caller keeps its own
    reference to the array it passed in (as e.g. compare_dendro.py's test
    harness does), that copy isn't affected -- del here only helps once the
    caller also lets go of it, or in a pipeline where compute() is the last
    thing holding a reference.

    Only promotes to float64 when `data` isn't already some floating dtype
    (e.g. an integer array) -- previously this forced a float64 copy
    unconditionally, silently doubling memory for float32 input (6.1 GB ->
    12.2 GB at 1.5 billion voxels) even though float32 is what the rest of
    this project already uses for cube data (see gpu_dendro_proto.py) and
    costs no real precision for typical intensity/brightness values.
    """
    data = np.asarray(data)
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float64)
    shape = data.shape
    finite = np.isfinite(data)
    if min_value is None:
        min_value = float(np.min(data[finite])) - 1.0
    keep = finite & (data > min_value)
    del finite

    basin_id, padded_shape = _label_basins(data, keep)
    edges = _saddle_edges(data, keep, basin_id)
    basin_arrays = _basin_value_voxel_arrays(data, basin_id, keep)
    del data, keep, basin_id

    next_id_start = int(np.prod(padded_shape)) + 1
    tree, active, redirect = _build_tree(edges, basin_arrays, min_delta, min_npix, next_id_start)
    _finalize(tree, active, redirect, basin_arrays, min_delta, min_npix)
    return _assemble(shape, tree)
