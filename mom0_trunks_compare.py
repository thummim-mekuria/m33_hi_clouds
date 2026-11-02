"""Zeroth moment + per-trunk outlines: r15-eroded dendrogram vs noise-weighted (S/N) dendrogram.

mom0 is the plain channel sum x dv (K km/s, no clipping), as in the 2026-10-09
mom0_trunks_r15.py. Each voxel's structure is mapped to its trunk (root of its tree),
trunk ids are projected along velocity, and each trunk's projected boundary is drawn.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from skimage.segmentation import find_boundaries
from astropy.io import fits
from astropy.wcs import WCS

MAPS = '/Users/thummimm/Documents/research/data/maps/'
CUBE = MAPS + 'm33_all_co21_8p0as.K.fits'
ERODED = '/Users/thummimm/Documents/research/M33/ipynbs/m33_co21_8p0as_eroded_r15.npy'
RUNS = {'r15': MAPS + 'm33_co21_8p0as_eroded_r15_parallel_dendro',
        'snr': MAPS + 'm33_all_co21_8p0as_snr_parallel_dendro'}
FIGDIR = '/Users/thummimm/Documents/research/M33/'
CHUNK = 20


def trunk_lookup(catalog):
    """Array mapping structure idx -> idx of its trunk (root of its tree)."""
    c = pd.read_csv(catalog, usecols=['idx', 'parent'])
    parent = np.full(c.idx.max() + 1, -1, dtype=np.int64)
    parent[c.idx.values] = c.parent.values
    root = np.arange(parent.size)
    p = parent.copy()
    while (p >= 0).any():                          # pointer-jump up the tree until every node hits a root
        up = p >= 0
        root[up] = p[up]
        p[up] = parent[p[up]]
    return root, (parent == -1).sum()


hdr = fits.getheader(CUBE)
wcs = WCS(hdr).celestial
dv = abs(hdr['CDELT3']) / (1000.0 if str(hdr.get('CUNIT3', 'm/s')).strip() == 'm/s' else 1.0)

# eroded-cube 2D mask, rebuilt as in 05_decomposition_time.ipynb cell 23, checked against the saved npy
cube = fits.open(CUBE, memmap=True)[0].data
nv, ny, nx = cube.shape
mom0 = np.zeros((ny, nx)); seen = np.zeros((ny, nx), bool)
roots = {k: trunk_lookup(v + '_catalog.csv') for k, v in RUNS.items()}
idxs = {k: fits.open(v + '_index_map.fits', memmap=True)[0].data for k, v in RUNS.items()}
tmap = {k: np.full((ny, nx), -1, dtype=np.int64) for k in RUNS}      # projected trunk id
nvox = {k: 0 for k in RUNS}
for i in range(0, nv, CHUNK):
    c = np.asarray(cube[i:i + CHUNK], dtype=np.float64)
    mom0 += np.nansum(c, axis=0); seen |= np.isfinite(c).any(axis=0)
    for k in RUNS:
        ix = np.asarray(idxs[k][i:i + CHUNK])
        nvox[k] += int(np.count_nonzero(ix >= 0))
        t = np.where(ix >= 0, roots[k][0][np.clip(ix, 0, None)], -1)
        tmap[k] = np.maximum(tmap[k], t.max(axis=0))
mom0 = np.where(seen, mom0 * dv, np.nan)

r = 15
yy, xx = np.meshgrid(np.arange(2 * r + 1) - r, np.arange(2 * r + 1) - r)
emask = nd.binary_erosion(seen, yy**2 + xx**2 < r**2)
ero = np.load(ERODED, mmap_mode='r')
for ch in (50, 200, 350):
    assert np.array_equal(np.isfinite(ero[ch]), emask & np.isfinite(cube[ch])), ch
mom0_ero = np.where(emask, mom0, np.nan)

# --- numbers
rim = seen & ~emask                                   # the band erosion blanks
foot = {k: tmap[k] >= 0 for k in RUNS}
print('valid pixels %d, eroded-away rim %d (%.1f%%)' % (seen.sum(), rim.sum(), 100 * rim.mean() / seen.mean()))
for k in RUNS:
    root, ntrunk = roots[k]
    print(f'{k}: {root.size:,} structures, {ntrunk:,} trunks, {nvox[k]:,} voxels assigned, '
          f'footprint {foot[k].sum():,} px ({foot[k][rim].sum():,} in the r15 rim)')
both = foot['r15'] & foot['snr']
print('footprint overlap: %d px; r15-only %d; snr-only %d (of which %d in the interior)'
      % (both.sum(), (foot['r15'] & ~foot['snr']).sum(), (foot['snr'] & ~foot['r15']).sum(),
         (foot['snr'] & ~foot['r15'] & emask).sum()))
np.savez_compressed(FIGDIR + 'ipynbs/mom0_trunkmaps_r15_vs_snr.npz', mom0=mom0, emask=emask,
                    tmap_r15=tmap['r15'], tmap_snr=tmap['snr'])


# --- figures
def outline(t):
    b = find_boundaries(t + 1, mode='inner') & (t >= 0)
    return np.ma.masked_where(~b, b)


vmax = np.nanpercentile(mom0_ero, 99.5)
panels = [('r15', mom0_ero, 'eroded r=15 px, T > 0.5 K'), ('snr', mom0, 'noise-weighted, T/σ > 10')]
colors = {'r15': 'crimson', 'snr': 'dodgerblue'}


def draw(ax, img, key, title, box=None):
    ax.imshow(img, origin='lower', cmap='gray_r', vmin=0, vmax=vmax, interpolation='nearest')
    ax.imshow(outline(tmap[key]), origin='lower', cmap=matplotlib.colors.ListedColormap([colors[key]]),
              interpolation='nearest')
    ax.set_title(title, fontsize=10)
    ax.coords[0].set_major_formatter('hh:mm:ss'); ax.coords[1].set_major_formatter('dd:mm')
    ax.set_xlabel('RA (J2000)'); ax.set_ylabel('Dec (J2000)')
    if box:
        (x0, x1), (y0, y1) = box
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)


fig = plt.figure(figsize=(13, 12), constrained_layout=True)
for j, (k, img, title) in enumerate(panels):
    draw(fig.add_subplot(1, 2, j + 1, projection=wcs), img, k,
         f'{title}: {roots[k][1]:,} trunks')
fig.suptitle('CO(2-1) 8″ zeroth moment (K km/s, stretch 0 to %.1f) with dendrogram trunk outlines' % vmax)
fig.savefig(FIGDIR + 'm33_co21_trunks_r15_vs_snr.png', dpi=300)

# zooms: brightest eroded-mom0 region, and the horizontal-stripe band
yc, xc = np.unravel_index(np.nanargmax(nd.uniform_filter(np.nan_to_num(mom0_ero), 25)), mom0.shape)
boxes = {'brightest region': ((xc - 150, xc + 150), (yc - 150, yc + 150)),
         'stripe band': ((250, 950), (1300, 1550))}
fig = plt.figure(figsize=(13, 11), constrained_layout=True)
n = 1
for name, box in boxes.items():
    for k, img, title in panels:
        draw(fig.add_subplot(2, 2, n, projection=wcs), img, k, f'{title} — {name}', box); n += 1
fig.savefig(FIGDIR + 'm33_co21_trunks_r15_vs_snr_zoom.png', dpi=200)
print('zoom centre (x, y):', xc, yc)
