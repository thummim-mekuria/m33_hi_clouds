"""r15-eroded vs noise-weighted (S/N) CO trunks, placed on the full M33 disk (DSS optical).

Uses the projected trunk maps cached by mom0_trunks_compare.py. Trunks are drawn as filled
footprints in the CO cube's own WCS (transformed onto the DSS axes, no regridding); the
dashed line is the edge of the CO mosaic coverage, and the dotted line in the r15 panel is
the edge after the 15 px erosion.
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from astropy.coordinates import SkyCoord

M33 = '/Users/thummimm/Documents/research/M33/'
CUBE = '/Users/thummimm/Documents/research/data/maps/m33_all_co21_8p0as.K.fits'

dss = fits.open(M33 + 'm33_dss.fits')[0]
dss_wcs = WCS(dss.header).celestial
co_wcs = WCS(fits.getheader(CUBE)).celestial
z = np.load(M33 + 'ipynbs/mom0_trunkmaps_r15_vs_snr.npz')
seen = np.isfinite(z['mom0'])
panels = [('r15', 'eroded r=15 px, T > 0.5 K', 'crimson'), ('snr', 'noise-weighted, T/σ > 10', 'dodgerblue')]
MAPS = '/Users/thummimm/Documents/research/data/maps/'
cats = {'r15': MAPS + 'm33_co21_8p0as_eroded_r15_parallel_dendro_catalog.csv',
        'snr': MAPS + 'm33_all_co21_8p0as_snr_parallel_dendro_catalog.csv'}
ntrunks = {k: int((np.loadtxt(f, delimiter=',', skiprows=1, usecols=5) == -1).sum()) for k, f in cats.items()}

# field of view: centred on the M33 nucleus (1h33m50.9s +30d39m36s), framing the whole CO mosaic
corners = SkyCoord(['1h31m21s +30d02m36s', '1h36m21s +30d02m36s', '1h31m21s +31d16m36s', '1h36m21s +31d16m36s'])
cx, cy = dss_wcs.world_to_pixel(corners)
xlim, ylim = (cx.min(), cx.max()), (cy.min(), cy.max())

norm = ImageNormalize(dss.data, interval=PercentileInterval(99.7), stretch=AsinhStretch(0.05))
fig = plt.figure(figsize=(11, 6.7), constrained_layout=True)
for j, (k, title, color) in enumerate(panels):
    ax = fig.add_subplot(1, 2, j + 1, projection=dss_wcs)
    ax.imshow(dss.data, origin='lower', cmap='gray_r', norm=norm)
    tr = ax.get_transform(co_wcs)
    foot = (z['tmap_' + k] >= 0).astype(float)
    ax.contourf(foot, levels=[0.5, 1.5], colors=[color], transform=tr)
    ax.contour(seen.astype(float), levels=[0.5], colors='k', linewidths=0.6, linestyles='--', transform=tr)
    if k == 'r15':
        ax.contour(z['emask'].astype(float), levels=[0.5], colors='k', linewidths=0.4, linestyles=':', transform=tr)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)   # set explicitly: contours in another WCS stretch the limits
    ax.set_title(f'{title}: {ntrunks[k]:,} trunks', fontsize=11)
    ax.coords[0].set_major_formatter('hh:mm:ss'); ax.coords[1].set_major_formatter('dd:mm')
    ax.set_xlabel('RA (J2000)'); ax.set_ylabel('Dec (J2000)')
fig.suptitle('CO(2-1) 8″ dendrogram trunks on the M33 disk (DSS); dashed = CO mosaic coverage')
fig.savefig(M33 + 'm33_co21_trunks_r15_vs_snr_disk.png', dpi=250)
