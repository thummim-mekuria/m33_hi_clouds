"""2x3 inverted-cube test: noise-weighted dendrograms on +T/sigma (top row) vs -T/sigma (bottom row) at
S/N floors of 10, 5 and 2 sigma (columns; min_delta = 2 sigma, min_npix = 5, un-eroded cube; made by
run_pd_snr.py <v> 2 [--invert]). Every structure in the -T/sigma row is noise, so it is the false-positive
map for the panel above it. Each panel shows assigned voxels per sky pixel (velocity-projected) on the
mosaic footprint, with the r15 erosion boundary dashed (masks from data/m33_co21_8p0as_eroded_masks.fits).
"""
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from astropy.io import fits
from astropy.wcs import WCS

MAPS = '/Users/thummimm/Documents/research/data/maps/'
M33 = '/Users/thummimm/Documents/research/M33/'
CACHE = M33 + 'data/pos_neg_snr_counts.npz'
LEVELS = (10, 5, 2)
CHUNK = 25

masks = fits.open(M33 + 'data/m33_co21_8p0as_eroded_masks.fits')
wcs = WCS(masks['FOOT'].header)
foot = masks['FOOT'].data.astype(bool)
keep15 = masks['R15'].data.astype(bool)


def project(tag):
    """Assigned voxels per sky pixel, structure count, and per-structure mean (y, x) for one run."""
    idx = fits.open(MAPS + f'm33_all_co21_8p0as_{tag}_parallel_dendro_index_map.fits', memmap=True)[0].data
    cnt = np.zeros(foot.shape, np.int32)
    n = sy = sx = np.zeros(0)
    for i in range(0, idx.shape[0], CHUNK):
        ix = np.asarray(idx[i:i + CHUNK]); m = ix >= 0
        cnt += m.sum(0, dtype=np.int32)
        _, y, x = np.nonzero(m); s = ix[m]
        if s.size:
            L = max(len(n), int(s.max()) + 1)
            n, sy, sx = (np.pad(a, (0, L - len(a))) + np.bincount(s, weights=w, minlength=L)
                         for a, w in ((n, None), (sy, y), (sx, x)))
    ids = np.nonzero(n)[0]
    cen = np.column_stack([sy[ids] / n[ids], sx[ids] / n[ids]])
    return cnt, len(ids), cen


if os.path.exists(CACHE):
    z = np.load(CACHE)
    runs = {(s, v): (z[f'{s}{v}_cnt'], int(z[f'{s}{v}_n']), z[f'{s}{v}_cen']) for s in '+-' for v in LEVELS}
else:
    runs = {}
    for v in LEVELS:
        for s, tag in (('+', f'snr{v}d2'), ('-', f'negsnr{v}d2')):
            runs[s, v] = project(tag)
            print(f'{s}T/sigma > {v}: {runs[s, v][1]:,} structures, {runs[s, v][0].sum():,} voxels', flush=True)
    np.savez_compressed(CACHE, **{f'{s}{v}_{k}': x for (s, v), r in runs.items()
                                  for k, x in zip(('cnt', 'n', 'cen'), r)})

vmax = max(r[0].max() for r in runs.values())
cmap = ListedColormap(plt.cm.Blues(np.linspace(0.35, 1, 256)))   # one hue; floor kept off the grey footprint
norm = LogNorm(1, vmax)
bg = np.where(foot, 1.0, np.nan)

fig, axes = plt.subplots(2, 3, figsize=(13.5, 15.5), subplot_kw=dict(projection=wcs), constrained_layout=True)
for j, v in enumerate(LEVELS):
    for i, (s, label) in enumerate((('+', '+T/σ (real)'), ('-', '−T/σ (noise only)'))):
        ax = axes[i, j]
        cnt, n, cen = runs[s, v]
        ax.imshow(bg, origin='lower', cmap='gray', vmin=0, vmax=1.09, interpolation='nearest')
        im = ax.imshow(np.where(cnt > 0, cnt, np.nan), origin='lower', cmap=cmap, norm=norm, interpolation='nearest')
        ax.contour(keep15.astype(float), levels=[0.5], colors='0.35', linewidths=0.6, linestyles='--')
        if 0 < n <= 50:                                           # sparse panels: ring each structure so it is visible
            ax.scatter(cen[:, 1], cen[:, 0], s=90, facecolors='none', edgecolors='#c2410c', linewidths=1.2)
        rim = cnt[foot & ~keep15].sum()
        ax.set_title(f'{label}, {"−" if s == "-" else ""}T/σ > {v}\n{n:,} structures · {cnt.sum():,} voxels · '
                     f'{100 * rim / max(cnt.sum(), 1):.1f}% in r15 rim', fontsize=10)
        if n == 0:
            ax.text(0.5, 0.5, 'no structures', transform=ax.transAxes, ha='center', va='center', fontsize=12,
                    color='0.25', bbox=dict(facecolor='white', edgecolor='0.6', boxstyle='round,pad=0.4'))
        ax.coords[0].set_major_formatter('hh:mm:ss'); ax.coords[1].set_major_formatter('dd:mm')
        ax.coords[0].set_ticklabel(exclude_overlapping=True)
        ax.set_xlabel('RA (J2000)' if i == 1 else ' '); ax.set_ylabel('Dec (J2000)' if j == 0 else ' ')
        if j:
            ax.coords[1].set_ticklabel_visible(False)
fig.colorbar(im, ax=axes, shrink=0.45, location='right', label='structure voxels per sky pixel (velocity-projected)')
fig.get_layout_engine().set(rect=(0, 0, 1, 0.965))
fig.text(0.5, 0.995, 'Inverted-cube test of the noise-weighted CO(2-1) dendrogram: real (+T/σ) vs noise-only (−T/σ)\n'
         'min_delta = 2σ, min_npix = 5, not eroded; grey = mosaic footprint, dashed = r15 erosion boundary; '
         'rings mark structures in sparse panels', ha='center', va='top', fontsize=12)
fig.savefig(M33 + 'm33_co21_pos_vs_neg_snr_grid.png', dpi=200)
print('saved', M33 + 'm33_co21_pos_vs_neg_snr_grid.png')
