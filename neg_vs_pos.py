"""Inverted-cube test: false-positive rate of the noise-weighted dendrogram vs distance from the mosaic edge.

For each threshold (10/5/2 sigma, min_delta 2 sigma), compares the -T/sigma run (pure noise) with the
+T/sigma run, by distance-to-edge bin and by the r5/r10/r15 erosion masks.
"""
import os
import numpy as np
from astropy.io import fits
from scipy.ndimage import distance_transform_edt, binary_dilation

MAPS = '/Users/thummimm/Documents/research/data/maps/'
M33 = '/Users/thummimm/Documents/research/M33/'
S = '/private/tmp/claude-501/-Users-thummimm/185a7d79-b805-4886-a9e7-9d353a1244c2/scratchpad/'
C = 25
cache = S + 'masks2d.npz'

if os.path.exists(cache):
    z = np.load(cache); foot = z['foot']; keep = {r: z[f'r{r}'] for r in (5, 10, 15)}
else:
    cube = fits.open(MAPS + 'm33_all_co21_8p0as.K.fits', memmap=True)[0].data
    foot = np.zeros(cube.shape[1:], bool)
    for i in range(0, cube.shape[0], C):
        foot |= np.isfinite(np.asarray(cube[i:i + C])).any(0)
    keep = {}
    for r in (5, 10, 15):
        e = np.load(M33 + f'ipynbs/m33_co21_8p0as_eroded_r{r}.npy', mmap_mode='r')
        k = np.zeros(foot.shape, bool)
        for i in range(0, e.shape[0], C):
            k |= np.isfinite(e[i:i + C]).any(0)
        keep[r] = k
    np.savez_compressed(cache, foot=foot, **{f'r{r}': k for r, k in keep.items()})

dist = distance_transform_edt(foot)
bins = [0, 2, 5, 10, 15, 30, 70, np.inf]
area = np.histogram(dist[foot], bins)[0]


def load(tag):
    """Per-pixel voxel counts, structure ids per pixel bin, and leaf/trunk counts for one run."""
    f = MAPS + f'm33_all_co21_8p0as_{tag}_parallel_dendro_index_map.fits'
    if not os.path.exists(f):
        return None
    idx = fits.open(f, memmap=True)[0].data
    cnt = np.zeros(foot.shape, np.int64)
    ids = set()
    lost = {r: set() for r in keep}
    for i in range(0, idx.shape[0], C):
        ix = np.asarray(idx[i:i + C]); m = ix >= 0
        cnt += m.sum(0)
        ids.update(np.unique(ix[m]).tolist())
        for r, k in keep.items():
            lost[r].update(np.unique(ix[m & ~k[None]]).tolist())
    return cnt, ids, lost


pos_ref = load('snr5d2')[0] > 0                                      # real-emission sky footprint at 5 sigma
near_pos = binary_dilation(pos_ref, iterations=10)                    # within 2 beams (10 px) of it

for v in (10, 5, 2):
    P, N = load(f'snr{v}d2'), load(f'negsnr{v}d2')
    if N is None:
        print(f'\n{v} sigma: no inverted run yet'); continue
    (pc, pids, plost), (nc, nids, nlost) = P, N
    print(f'\n===== {v} sigma (min_delta 2) =====')
    print(f'structures: real {len(pids):,}   noise-only {len(nids):,}   -> {100*len(nids)/max(len(pids),1):.1f}% of real count')
    print(f'voxels:     real {pc.sum():,}   noise-only {nc.sum():,}')
    for r in keep:
        print(f'  r{r}-eroded px: real {len(plost[r]):,} structures / {pc[~keep[r]].sum():,} vox;'
              f'  noise-only {len(nlost[r]):,} structures / {nc[~keep[r]].sum():,} vox')
    print('  dist px    area px   real vox/px   noise vox/px   noise/real')
    ph = np.histogram(dist, bins, weights=pc)[0]; nh = np.histogram(dist, bins, weights=nc)[0]
    for a, b, A, p, n in zip(bins[:-1], bins[1:], area, ph, nh):
        print(f'  {a:>3.0f}-{b:<4.0f} {A:9,d}   {p/A:10.4f}   {n/A:11.4f}   {n/p if p else float("nan"):9.3f}')
    if nc.sum():
        print(f'  noise-only voxels in px within 10 px of 5-sigma real emission: {100*nc[near_pos].sum()/nc.sum():.1f}%'
              f'  (those px are {100*near_pos[foot].mean():.1f}% of the footprint)')
