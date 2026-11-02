"""4x5 grid of CO(2-1) dendrogram trunk footprints and SCIMES clouds: rows = unweighted (T > 0.5 K)
footprints / their SCIMES clouds / noise-weighted (T/sigma > 10) footprints / their SCIMES clouds;
columns = input / not eroded / eroded r = 5, 10, 15 px. The input column shows the original cube's zeroth
moment (row 1) and the noise_map_co.py sigma map the weighted runs divide by (row 3).

Each panel: footprint of every voxel assigned to any structure (= union of all trunks), projected
along velocity, filled on the plain-sum zeroth moment (K km/s) of the original cube, with the
eroded rim blanked in the eroded columns. Erosion masks are rebuilt as in
05_decomposition_time.ipynb and checked against the saved eroded cubes.
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.ndimage as nd
from astropy.io import fits
from astropy.wcs import WCS

MAPS = '/Users/thummimm/Documents/research/data/maps/'
M33 = '/Users/thummimm/Documents/research/M33/'
CUBE = MAPS + 'm33_all_co21_8p0as.K.fits'
NOISE = MAPS + 'm33_all_co21_8p0as_noise_map.fits'
COLS = [0, 5, 10, 15]
ROWS = [('unweighted, T > 0.5 K', 'crimson',
         ['m33_all_co21_8p0as_parallel_dendro'] + [f'm33_co21_8p0as_eroded_r{r}_parallel_dendro' for r in COLS[1:]]),
        ('noise-weighted, T/σ > 10', 'dodgerblue',
         ['m33_all_co21_8p0as_snr_parallel_dendro'] + [f'm33_co21_8p0as_eroded_r{r}_snr_parallel_dendro' for r in COLS[1:]])]
CHUNK = 20

wcs = WCS(fits.getheader(CUBE)).celestial
cube = fits.open(CUBE, memmap=True)[0].data
z = np.load(M33 + 'ipynbs/mom0_trunkmaps_r15_vs_snr.npz')
mom0 = z['mom0']
seen = np.isfinite(mom0)

emask = {0: seen}
for r in COLS[1:]:
    yy, xx = np.meshgrid(np.arange(2 * r + 1) - r, np.arange(2 * r + 1) - r)
    emask[r] = nd.binary_erosion(seen, yy**2 + xx**2 < r**2)
    ero = np.load(M33 + f'ipynbs/m33_co21_8p0as_eroded_r{r}.npy', mmap_mode='r')
    for ch in (50, 200, 350):
        assert np.array_equal(np.isfinite(ero[ch]), emask[r] & np.isfinite(cube[ch])), (r, ch)

# Per-cloud SCIMES footprints come from the structure -> cloud lookup applied to the index map, not from
# the saved _scimes_cloudmap.fits, which keeps one cloud per sky pixel and so hides line-of-sight overlaps.
ny, nx = seen.shape
foot, counts, clouds = {}, {}, {}
for _, _, prefixes in ROWS:
    for p in prefixes:
        idx = fits.open(MAPS + p + '_index_map.fits', memmap=True)[0].data
        lookup = MAPS + p + '_scimes_lookup.csv'
        cloud_of = pd.read_csv(lookup).cloud_idx.values if os.path.exists(lookup) else None
        f = np.zeros(seen.shape, bool)
        codes = []
        for i in range(0, idx.shape[0], CHUNK):
            ix = np.asarray(idx[i:i + CHUNK])
            f |= (ix >= 0).any(axis=0)
            if cloud_of is not None:
                c = np.where(ix >= 0, cloud_of[np.clip(ix, 0, None)], -1)
                z, y, x = np.nonzero(c >= 0)
                codes.append(np.unique(c[z, y, x] * (ny * nx) + y * nx + x))
        foot[p] = f
        par = pd.read_csv(MAPS + p + '_catalog.csv', usecols=['parent']).parent.values
        counts[p] = (par.size, int((par == -1).sum()))
        msg = f'{p}: {counts[p][0]:,} structures, {counts[p][1]:,} trunks, footprint {f.sum():,} px'
        if cloud_of is not None:
            codes = np.unique(np.concatenate(codes))
            ncat = len(pd.read_csv(MAPS + p + '_scimes_clouds.csv'))
            clouds[p] = (codes // (ny * nx), codes % (ny * nx), ncat)
            msg += f'; SCIMES {ncat} clouds, {np.unique(clouds[p][0]).size} with voxels'
        print(msg, flush=True)

sigma = fits.getdata(NOISE)
print(f'sigma: median {np.nanmedian(sigma):.4f} K, 1-99% {np.nanpercentile(sigma, 1):.4f}-'
      f'{np.nanpercentile(sigma, 99):.4f} K', flush=True)

vmax = np.nanpercentile(mom0[emask[15]], 99.5)
NR, NC = 2 * len(ROWS), len(COLS) + 1
fig = plt.figure(figsize=(21.5, 31), constrained_layout=True)

def _coords(ax, i, j):
    ax.coords[0].set_major_formatter('hh:mm:ss'); ax.coords[1].set_major_formatter('dd:mm')
    ax.coords[0].set_ticklabel(exclude_overlapping=True)
    ax.set_xlabel('RA (J2000)' if i == NR - 1 else ' '); ax.set_ylabel('Dec (J2000)' if j == 0 else ' ')
    if j: ax.coords[1].set_ticklabel_visible(False)

def _outlines(ax, cid, pix, color):
    """One contour per cloud around its velocity-projected footprint."""
    order = np.argsort(cid, kind='stable')
    cid, pix = cid[order], pix[order]
    _, start = np.unique(cid, return_index=True)
    for sel in np.split(pix, start[1:]):
        y, x = sel // nx, sel % nx
        y0, x0 = y.min() - 1, x.min() - 1
        sub = np.zeros((y.max() - y0 + 2, x.max() - x0 + 2))
        sub[y - y0, x - x0] = 1
        ax.contour(np.arange(x0, x0 + sub.shape[1]), np.arange(y0, y0 + sub.shape[0]), sub,
                   levels=[0.5], colors=[color], linewidths=0.5)

ax = fig.add_subplot(NR, NC, 1, projection=wcs)
quiet = np.isfinite(mom0) & (sigma < 0.1)                 # interior, away from the noisy tile rims
lo, hi = np.nanpercentile(mom0[quiet], [90, 99.9])         # floor above the noise speckle; top at the bright clouds
im = ax.imshow(mom0, origin='lower', cmap='viridis', vmin=lo, vmax=hi, interpolation='nearest')
fig.colorbar(im, ax=ax, shrink=0.6, label='K km/s', extend='both')
ax.set_title(f'input: original cube\nzeroth moment (plain sum), {lo:.1f}–{hi:.1f} K km/s', fontsize=10)
_coords(ax, 0, 0)
ax = fig.add_subplot(NR, NC, 2 * NC + 1, projection=wcs)
im = ax.imshow(sigma, origin='lower', cmap='magma_r', interpolation='nearest',
               norm=LogNorm(*np.nanpercentile(sigma, [1, 99])))
fig.colorbar(im, ax=ax, shrink=0.6, label='σ (K)')
ax.set_title(f'input: noise map σ (median {np.nanmedian(sigma):.3f} K)\n'
             'median|negative T| / 0.6745 per spectrum', fontsize=10)
_coords(ax, 2, 0)

for i, (rowlabel, color, prefixes) in enumerate(ROWS):
    for j, (r, p) in enumerate(zip(COLS, prefixes), start=1):
        erolabel = "not eroded" if r == 0 else f"eroded r = {r} px"
        for k in (0, 1):                                    # k = 0: dendrogram footprint, 1: SCIMES clouds
            row = 2 * i + k
            ax = fig.add_subplot(NR, NC, NC * row + j + 1, projection=wcs)
            ax.imshow(np.where(emask[r], mom0, np.nan), origin='lower', cmap='gray_r', vmin=0, vmax=vmax,
                      interpolation='nearest')
            if k == 0:
                ax.contourf(foot[p].astype(float), levels=[0.5, 1.5], colors=[color])
                ns, nt = counts[p]
                ax.set_title(f'{rowlabel}\n{erolabel}: {ns:,} structures, {nt:,} trunks', fontsize=10)
            elif p in clouds:
                cid, pix, ncat = clouds[p]
                _outlines(ax, cid, pix, color)
                ax.set_title(f'SCIMES on {rowlabel}\n{erolabel}: {ncat:,} clouds (outlined)', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'SCIMES not run:\n~128 GB per affinity matrix', transform=ax.transAxes,
                        ha='center', va='center', fontsize=11,
                        bbox=dict(facecolor='white', edgecolor='0.4', boxstyle='round,pad=0.5'))
                ax.set_title(f'SCIMES on {rowlabel}\n{erolabel}', fontsize=10)
            _coords(ax, row, j)
fig.get_layout_engine().set(rect=(0, 0, 1, 0.985))          # leave a strip for the overall title
fig.text(0.5, 0.997, 'CO(2-1) 8″ parallel_dendro trunk footprints (filled) and SCIMES clouds (outlined) on the '
         f'zeroth moment, 0–{vmax:.1f} K km/s; min_npix = 5, min_delta = 0.3 K / 6σ',
         ha='center', va='top', fontsize=12)
fig.savefig(M33 + 'm33_co21_dendro_grid_4x5.png', dpi=200)
