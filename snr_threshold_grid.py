"""3x3 grid of noise-weighted CO(2-1) dendrograms at different S/N floors: rows = min_value 10, 5, 2 sigma
(all with min_delta = 2 sigma, min_npix = 5, un-eroded cube; made by run_pd_snr.py <v> 2); columns = input /
dendrogram footprint (filled) / SCIMES clouds (outlined). The input column shows the original cube's zeroth
moment (row 1) and the noise_map_co.py sigma map the S/N cube divides by (row 2).
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.wcs import WCS

MAPS = '/Users/thummimm/Documents/research/data/maps/'
M33 = '/Users/thummimm/Documents/research/M33/'
CUBE = MAPS + 'm33_all_co21_8p0as.K.fits'
NOISE = MAPS + 'm33_all_co21_8p0as_noise_map.fits'
ROWS = [(10, 'navy'), (5, 'dodgerblue'), (2, 'darkorange')]
CHUNK = 20

wcs = WCS(fits.getheader(CUBE)).celestial
mom0 = np.load(M33 + 'ipynbs/mom0_trunkmaps_r15_vs_snr.npz')['mom0']
sigma = fits.getdata(NOISE)
ny, nx = mom0.shape

# Per-cloud SCIMES footprints from the structure -> cloud lookup applied to the index map (the saved
# _scimes_cloudmap.fits keeps one cloud per sky pixel and hides line-of-sight overlaps).
foot, counts, clouds = {}, {}, {}
for v, _ in ROWS:
    p = f'm33_all_co21_8p0as_snr{v}d2_parallel_dendro'
    idx = fits.open(MAPS + p + '_index_map.fits', memmap=True)[0].data
    lookup = MAPS + p + '_scimes_lookup.csv'
    cloud_of = pd.read_csv(lookup).cloud_idx.values if os.path.exists(lookup) else None
    f = np.zeros(mom0.shape, bool)
    codes = []
    for i in range(0, idx.shape[0], CHUNK):
        ix = np.asarray(idx[i:i + CHUNK])
        f |= (ix >= 0).any(axis=0)
        if cloud_of is not None:
            c = np.where(ix >= 0, cloud_of[np.clip(ix, 0, None)], -1)
            z, y, x = np.nonzero(c >= 0)
            codes.append(np.unique(c[z, y, x] * (ny * nx) + y * nx + x))
    foot[v] = f
    cat = pd.read_csv(MAPS + p + '_catalog.csv', usecols=['parent', 'is_leaf'])
    counts[v] = (len(cat), int((cat.parent == -1).sum()), int(cat.is_leaf.sum()))
    msg = f'{p}: {counts[v][0]:,} structures, {counts[v][1]:,} trunks, {counts[v][2]:,} leaves, footprint {f.sum():,} px'
    if cloud_of is not None:
        codes = np.unique(np.concatenate(codes))
        ncat = len(pd.read_csv(MAPS + p + '_scimes_clouds.csv'))
        clouds[v] = (codes // (ny * nx), codes % (ny * nx), ncat)
        msg += f'; SCIMES {ncat} clouds'
    print(msg, flush=True)

vmax = np.nanpercentile(mom0[np.isfinite(mom0) & (sigma < 0.1)], 99.5)
NR, NC = len(ROWS), 3
fig = plt.figure(figsize=(13.5, 23), constrained_layout=True)

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
ax = fig.add_subplot(NR, NC, NC + 1, projection=wcs)
im = ax.imshow(sigma, origin='lower', cmap='magma_r', interpolation='nearest',
               norm=LogNorm(*np.nanpercentile(sigma, [1, 99])))
fig.colorbar(im, ax=ax, shrink=0.6, label='σ (K)')
ax.set_title(f'input: noise map σ (median {np.nanmedian(sigma):.3f} K)\n'
             'median|negative T| / 0.6745 per spectrum', fontsize=10)
_coords(ax, 1, 0)

for i, (v, color) in enumerate(ROWS):
    for j in (1, 2):
        ax = fig.add_subplot(NR, NC, NC * i + j + 1, projection=wcs)
        ax.imshow(mom0, origin='lower', cmap='gray_r', vmin=0, vmax=vmax, interpolation='nearest')
        if j == 1:
            ax.contourf(foot[v].astype(float), levels=[0.5, 1.5], colors=[color])
            ns, nt, nl = counts[v]
            ax.set_title(f'T/σ > {v}, min_delta = 2σ\n{ns:,} structures, {nt:,} trunks, {nl:,} leaves', fontsize=10)
        elif v in clouds:
            cid, pix, ncat = clouds[v]
            _outlines(ax, cid, pix, color)
            ax.set_title(f'SCIMES on T/σ > {v}\n{ncat:,} clouds (outlined)', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'SCIMES not run:\ntoo many leaves for\nthe affinity matrix', transform=ax.transAxes,
                    ha='center', va='center', fontsize=11,
                    bbox=dict(facecolor='white', edgecolor='0.4', boxstyle='round,pad=0.5'))
            ax.set_title(f'SCIMES on T/σ > {v}', fontsize=10)
        _coords(ax, i, j)
fig.get_layout_engine().set(rect=(0, 0, 1, 0.985))          # leave a strip for the overall title
fig.text(0.5, 0.997, 'CO(2-1) 8″ noise-weighted parallel_dendro at three S/N floors: footprints (filled) and '
         f'SCIMES clouds (outlined)\non the zeroth moment, 0–{vmax:.1f} K km/s; min_delta = 2σ, min_npix = 5, '
         'not eroded', ha='center', va='top', fontsize=12)
fig.savefig(M33 + 'm33_co21_snr_threshold_grid.png', dpi=200)
