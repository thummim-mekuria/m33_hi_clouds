"""Noise fraction of the CO(2-1) T/sigma cube vs threshold N: voxels below -N sigma over voxels above +N sigma.

Assumes noise symmetric about zero, so N(< -N sigma) estimates the noise voxels above +N sigma. Voxel counts,
before min_npix/min_delta, so this is not the false-positive rate of the dendrogram structures. Negative
artifacts next to bright CO (2026-10-28) make it a slight over-estimate. Histogram cached in data/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits

CUBE = '/Users/thummimm/Documents/research/data/maps/m33_all_co21_8p0as.K.fits'
NOISE = '/Users/thummimm/Documents/research/data/maps/m33_all_co21_8p0as_noise_map.fits'
CACHE = '/Users/thummimm/Documents/research/M33/data/m33_co21_snr_hist.npz'
OUT = '/Users/thummimm/Documents/research/M33/m33_co21_noise_frac_vs_snr.png'
COLOR = '#2a78d6'
MARK = (3, 4, 5)                                       # thresholds labelled on the curve

edges = np.arange(0, 100.0001, 0.05)                   # |T/sigma| bins; top bin catches everything brighter
if os.path.exists(CACHE):
    c = np.load(CACHE)
    hpos, hneg = c['hpos'], c['hneg']
else:
    sigma = fits.getdata(NOISE).astype(np.float32)
    hpos = np.zeros(len(edges) - 1, dtype=np.int64)
    hneg = hpos.copy()
    with fits.open(CUBE, memmap=True) as h:
        src = h[0].data
        for i in range(0, src.shape[0], 20):
            snr = np.asarray(src[i:i + 20], dtype=np.float32) / sigma
            snr = snr[np.isfinite(snr)]
            hpos += np.histogram(np.minimum(snr[snr > 0], edges[-1] - 0.01), edges)[0]
            hneg += np.histogram(np.minimum(-snr[snr < 0], edges[-1] - 0.01), edges)[0]
    np.savez(CACHE, edges=edges, hpos=hpos, hneg=hneg)

N = edges[:-1]
npos = hpos[::-1].cumsum()[::-1]                       # voxels above +N sigma
nneg = hneg[::-1].cumsum()[::-1]                       # voxels below -N sigma
frac = 100 * nneg / npos

for n in (2, 3, 4, 5, 10):
    i = np.argmin(abs(N - n))
    print(f'{n:>3} sigma: +{npos[i]:,}  -{nneg[i]:,}  {frac[i]:.2f}%')

keep = (N >= 1) & (N <= 8) & (nneg > 0)
fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
ax.plot(N[keep], frac[keep], color=COLOR, lw=2)
for n in MARK:
    i = np.argmin(abs(N - n))
    ax.plot(N[i], frac[i], 'o', ms=7, color=COLOR, mec='w', mew=1.5, zorder=3)
    ax.annotate(f'{n}σ: {frac[i]:.1f}%', (N[i], frac[i]), xytext=(8, 6), textcoords='offset points',
                fontsize=10, color='#1f1f1f')

ax.set_yscale('log')
ax.set_xlim(1, 8)
ax.set_ylim(top=100)
ax.set_yticks([0.01, 0.1, 1, 10, 100], ['0.01', '0.1', '1', '10', '100'])
ax.set_xlabel('Threshold N (T/σ > N)')
ax.set_ylabel('Noise voxels above threshold [%]')
ax.set_title('CO(2-1) voxels above +Nσ that are noise, estimated from voxels below −Nσ', fontsize=10.5, loc='left')
ax.grid(alpha=0.25, lw=0.6)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT)
print('saved', OUT)
