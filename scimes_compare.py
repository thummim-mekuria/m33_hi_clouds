"""Compare the SCIMES cloud catalogs of the r5 / r10 / r15 eroded and noise-weighted (snr) runs.

Per cloud, from the index map + parallel_dendro_scimes.py's structure->cloud lookup (read in
channel chunks): voxel count, summed T*dv from the original K cube, and the voxel centroid,
whose distance from the edge of the CO mosaic coverage tests for edge noise.

Figures: m33_co21_scimes_compare_maps.png (cloud footprints per run on mom0) and
m33_co21_scimes_compare_dists.png (CDFs of size, luminosity proxy, distance from edge).
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from astropy.io import fits
from astropy.wcs import WCS

MAPS = '/Users/thummimm/Documents/research/data/maps/'
M33 = '/Users/thummimm/Documents/research/M33/'
CUBE = MAPS + 'm33_all_co21_8p0as.K.fits'
RUNS = {  # label, prefix, colour (reference categorical palette, slots 1-4)
    'r5':  ('eroded r=5 px, T > 0.5 K',  'm33_co21_8p0as_eroded_r5_parallel_dendro',  '#2a78d6'),
    'r10': ('eroded r=10 px, T > 0.5 K', 'm33_co21_8p0as_eroded_r10_parallel_dendro', '#eb6834'),
    'r15': ('eroded r=15 px, T > 0.5 K', 'm33_co21_8p0as_eroded_r15_parallel_dendro', '#1baf7a'),
    'snr': ('noise-weighted, T/σ > 10',  'm33_all_co21_8p0as_snr_parallel_dendro',    '#eda100'),
}
CHUNK = 20
INK, INK2 = '#0b0b0b', '#52514e'

hdr = fits.getheader(CUBE)
wcs = WCS(hdr).celestial
dv = abs(hdr['CDELT3']) / (1000.0 if str(hdr.get('CUNIT3', 'm/s')).strip() == 'm/s' else 1.0)
cube = fits.open(CUBE, memmap=True)[0].data
nv, ny, nx = cube.shape
z = np.load(M33 + 'ipynbs/mom0_trunkmaps_r15_vs_snr.npz')
mom0 = z['mom0']
seen = np.isfinite(mom0)
edge_dist = nd.distance_transform_edt(seen)          # px from the nearest pixel with no data

yy, xx = np.mgrid[0:ny, 0:nx]
clouds = {}
for k, (_, prefix, _) in RUNS.items():
    lk = pd.read_csv(MAPS + prefix + '_scimes_lookup.csv').cloud_idx.values
    ids = np.unique(lk[lk >= 0])
    slot = np.full(lk.size, -1); slot[lk >= 0] = np.searchsorted(ids, lk[lk >= 0])   # structure -> cloud row
    n = len(ids)
    npix, flux, sy, sx = (np.zeros(n) for _ in range(4))
    idx = fits.open(MAPS + prefix + '_index_map.fits', memmap=True)[0].data
    for i in range(0, nv, CHUNK):
        ix = np.asarray(idx[i:i + CHUNK])
        m = ix >= 0
        c = slot[ix[m]]
        keep = c >= 0
        c = c[keep]
        t = np.asarray(cube[i:i + CHUNK], dtype=np.float64)[m][keep]
        y = np.broadcast_to(yy, ix.shape)[m][keep]; x = np.broadcast_to(xx, ix.shape)[m][keep]
        npix += np.bincount(c, minlength=n); flux += np.bincount(c, weights=t, minlength=n)
        sy += np.bincount(c, weights=y, minlength=n); sx += np.bincount(c, weights=x, minlength=n)
    cy, cx = sy / npix, sx / npix
    d = edge_dist[np.round(cy).astype(int), np.round(cx).astype(int)]
    clouds[k] = pd.DataFrame(dict(cloud_idx=ids, npix=npix, lum_Kkms_px=flux * dv, y=cy, x=cx, edge_dist_px=d))
    clouds[k].to_csv(MAPS + prefix + '_scimes_cloudprops.csv', index=False)
    print(f'{k}: {n} clouds; median npix {np.median(npix):.0f}; '
          f'centroid within 5/10/15/30 px of an edge: '
          + ' / '.join(f'{np.mean(d <= r):.0%}' for r in (5, 10, 15, 30)), flush=True)

# --- figure 1: cloud footprints on mom0, central region of the mosaic
vmax = np.nanpercentile(mom0[z['emask']], 99.5)
fig = plt.figure(figsize=(12, 13), constrained_layout=True)
for j, (k, (label, prefix, color)) in enumerate(RUNS.items()):
    cmap2d = fits.getdata(MAPS + prefix + '_scimes_cloudmap.fits')
    ax = fig.add_subplot(2, 2, j + 1, projection=wcs)
    ax.imshow(mom0, origin='lower', cmap='gray_r', vmin=0, vmax=vmax, interpolation='nearest')
    ax.contourf((cmap2d >= 0).astype(float), levels=[0.5, 1.5], colors=[color])
    ax.contour(seen.astype(float), levels=[0.5], colors=INK2, linewidths=0.4, linestyles='--')
    ax.set_title(f'{label}: {len(clouds[k]):,} SCIMES clouds', fontsize=11, color=INK)
    ax.coords[0].set_major_formatter('hh:mm:ss'); ax.coords[1].set_major_formatter('dd:mm')
    ax.set_xlabel('RA (J2000)'); ax.set_ylabel('Dec (J2000)')
    ax.set_xlim(150, 1250); ax.set_ylim(600, 1800)
fig.suptitle('CO(2-1) 8″: SCIMES cloud footprints (projected along velocity) on the zeroth moment', color=INK)
fig.savefig(M33 + 'm33_co21_scimes_compare_maps.png', dpi=200)

# --- figure 2: CDFs
def cdf(ax, vals, color, label):
    v = np.sort(vals)
    ax.step(v, np.arange(1, v.size + 1) / v.size, where='post', color=color, lw=2, label=label)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
for k, (label, _, color) in RUNS.items():
    c = clouds[k]
    lab = f'{k} ({len(c):,})'
    cdf(axs[0], c.npix, color, lab)
    cdf(axs[1], c.lum_Kkms_px, color, lab)
    cdf(axs[2], c.edge_dist_px, color, lab)
axs[0].set_xscale('log'); axs[0].set_xlabel('cloud size [voxels]')
axs[1].set_xscale('log'); axs[1].set_xlabel('luminosity proxy  Σ T dv  [K km/s × pixels]')
axs[2].set_xscale('symlog', linthresh=10); axs[2].set_xlabel('centroid distance from mosaic edge [px]')
for r in (5, 10, 15):
    axs[2].axvline(r, color=INK2, lw=0.6, ls=':')
    axs[2].text(r, 1.01, f'r{r}', ha='center', va='bottom', fontsize=8, color=INK2, transform=axs[2].get_xaxis_transform())
axs[0].axvline(25 * np.pi / (4 * np.log(2)), color=INK2, lw=0.6, ls=':')   # beam area, FWHM 5 px
axs[0].text(25 * np.pi / (4 * np.log(2)), 0.02, ' 1 beam\n (1 channel)', fontsize=8, color=INK2)
for ax in axs:
    ax.set_ylim(0, 1); ax.set_ylabel('fraction of clouds ≤ x', color=INK)
    ax.grid(color='#e6e5e1', lw=0.5); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
axs[0].legend(title='run (clouds)', frameon=False, loc='lower right')
fig.suptitle('SCIMES cloud properties: eroded (r5, r10, r15) vs noise-weighted (snr) dendrograms', color=INK)
fig.savefig(M33 + 'm33_co21_scimes_compare_dists.png', dpi=200)
