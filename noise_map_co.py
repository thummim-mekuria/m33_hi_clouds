"""Per-pixel noise map of the CO(2-1) cube, from the negative side of each spectrum.

Reconstructed from the 2026-10-09 session (transcript 8a3cec26), where it was made by
sed-substituting noise_map_himidres.py and written to a scratchpad that was later
cleared. Method unchanged; only the output paths are now absolute (data/maps/) and the
FITS map carries the cube's 2D celestial header.
"""
import warnings
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

F = '/Users/thummimm/Documents/research/data/maps/m33_all_co21_8p0as.K.fits'
OUT = '/Users/thummimm/Documents/research/data/maps/m33_all_co21_8p0as_noise_map'
FGND_CHAN = 0           # CO cube: no Milky Way foreground trim
MIN_VALUE = 0.5         # K, the dendrogram floor used for the CO cube
ROWS = 30               # sky rows per chunk, to keep memory small
MIN_NEG = 20            # need this many negative voxels along a spectrum to trust sigma

with fits.open(F, memmap=True) as h:
    d = h[0].data
    hdr2d = WCS(h[0].header).celestial.to_header()
    nv, ny, nx = d.shape
    sigma = np.full((ny, nx), np.nan, dtype=np.float32)
    for j in range(0, ny, ROWS):
        v = np.asarray(d[FGND_CHAN:, j:j + ROWS, :], dtype=np.float32)   # (nv', rows, nx)
        neg = np.where(v < 0, -v, np.nan)                                 # |negative values| only
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            s = np.nanmedian(neg, axis=0) / 0.6745                        # half-Gaussian: median|v| = 0.6745 sigma
        s[np.sum(np.isfinite(neg), axis=0) < MIN_NEG] = np.nan
        sigma[j:j + ROWS] = s

hdr2d['BUNIT'] = 'K'
fits.writeto(OUT + '.fits', sigma, hdr2d, overwrite=True)

ratio = MIN_VALUE / sigma                                                 # min_value in units of local sigma
fig, ax = plt.subplots(1, 2, figsize=(13, 7), constrained_layout=True)
im0 = ax[0].imshow(sigma, origin='lower', cmap='magma', vmin=np.nanpercentile(sigma, 1), vmax=np.nanpercentile(sigma, 99))
fig.colorbar(im0, ax=ax[0], shrink=0.7, label=r'$\sigma$ [K]')
ax[0].set_title(f'per-pixel noise (channels {FGND_CHAN}+; negative-side median/0.6745)')
im1 = ax[1].imshow(ratio, origin='lower', cmap='viridis', vmin=0, vmax=8)
ax[1].contour(ratio, levels=[3], colors='w', linewidths=0.6)
fig.colorbar(im1, ax=ax[1], shrink=0.7, label=f'min_value / sigma  ({MIN_VALUE:g} K / sigma)')
ax[1].set_title('threshold in local-sigma units (white contour = 3 sigma)')
for a in ax: a.set_xlabel('x [pix]'); a.set_ylabel('y [pix]')
fig.savefig(OUT + '.png', dpi=110)

ok = np.isfinite(sigma)
print('valid pixels: %d / %d' % (ok.sum(), sigma.size))
print('sigma [K]: p1 %.2f  p16 %.2f  median %.2f  p84 %.2f  p99 %.2f' % tuple(np.nanpercentile(sigma, [1, 16, 50, 84, 99])))
print('fraction of valid pixels where min_value < 3 sigma: %.3f   < 2 sigma: %.3f' % (np.mean(ratio[ok] < 3), np.mean(ratio[ok] < 2)))
