"""SCIMES clouds on the 5-sigma noise-weighted dendrogram (snr5d2) with save_isol_leaves=False (left;
parallel_dendro_scimes.py snr5d2) vs True (right; parallel_dendro_scimes.py snr5d2 --isol).

Row 1: clouds outlined on the zeroth moment by their velocity-projected footprints; on the right, the
       isolated single-leaf trunks that save_isol_leaves=True adds are drawn in a second colour.
Row 2: mass vs radius.   Row 3: linewidth vs radius.

Cloud properties are intensity-weighted moments over each cloud's assigned voxels (no extrapolation to
T = 0, unlike CPROPS), cached in data/maps/<prefix>_scimes_isol_cloudprops.csv:
  R      = 1.91 sqrt(sig_maj sig_min), each axis beam-deconvolved in quadrature (Solomon+87, Rosolowsky+06);
           NaN when either axis is no larger than the beam (unresolved); R_obs_pc is the same without
           deconvolution, an upper limit on R, plotted for the unresolved clouds
  sig_v  = intensity-weighted velocity dispersion, channel width (dv / sqrt(2 pi)) removed in quadrature
  M      = alpha_CO(2-1) * L_CO, L_CO = sum(T dv) * pixel area; X_CO = 4e20 and R21 = 0.8 (Dobbs+2019)
           -> alpha_CO(1-0) = 6.4 and alpha_CO(2-1) = 8.0 Msun (K km/s pc^2)^-1, helium included
  D      = 859 kpc (as in ipynbs/mekuria_imara_pipeline.py)
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy.io import fits
from astropy.wcs import WCS

MAPS = '/Users/thummimm/Documents/research/data/maps/'
M33 = '/Users/thummimm/Documents/research/M33/'
CUBE = MAPS + 'm33_all_co21_8p0as.K.fits'
P = MAPS + 'm33_all_co21_8p0as_snr5d2_parallel_dendro'
PROPS = P + '_scimes_isol_cloudprops.csv'
CLUST, ISOL = '#1d4ed8', '#c2410c'
INK2, GRID = '#52514e', '#e6e5e1'
CHUNK = 20
D_PC = 859e3
ALPHA_21 = 6.4 / 0.8                                               # Msun (K km/s pc^2)^-1

hdr = fits.getheader(CUBE)
wcs = WCS(hdr).celestial
dv = abs(hdr['CDELT3']) / 1000.0                                   # km/s
pc_px = np.deg2rad(abs(hdr['CDELT2'])) * D_PC                      # pc per pixel
sig_beam = hdr['BMAJ'] / abs(hdr['CDELT2']) / np.sqrt(8 * np.log(2))   # px
mom0 = np.load(M33 + 'ipynbs/mom0_trunkmaps_r15_vs_snr.npz')['mom0']
sigma = fits.getdata(MAPS + 'm33_all_co21_8p0as_noise_map.fits')
ny, nx = mom0.shape

cat = pd.read_csv(P + '_catalog.csv')
off = pd.read_csv(P + '_scimes_clouds.csv')
on = pd.read_csv(P + '_scimes_isol_clouds.csv')
lookup_on = pd.read_csv(P + '_scimes_isol_lookup.csv').cloud_idx.values
isol = set(cat.idx[(cat.parent == -1) & cat.is_leaf])
added = set(on.cloud_idx) - set(off.cloud_idx)
assert set(off.cloud_idx) <= set(on.cloud_idx) and added == isol   # same clustering + every isolated leaf

# --- one pass over the index map: per-cloud footprints (row 1) and weighted moments (rows 2-3)
ids = np.array(sorted(on.cloud_idx))
slot = np.full(lookup_on.size, -1)
slot[lookup_on >= 0] = np.searchsorted(ids, lookup_on[lookup_on >= 0])   # structure -> cloud row
need_props = not os.path.exists(PROPS)
idx = fits.open(P + '_index_map.fits', memmap=True)[0].data
cube = fits.open(CUBE, memmap=True)[0].data
yy, xx = np.mgrid[0:ny, 0:nx]
n = len(ids)
S = {k: np.zeros(n) for k in ('w', 'x', 'y', 'v', 'xx', 'yy', 'xy', 'vv')}
codes = []
for i in range(0, idx.shape[0], CHUNK):
    ix = np.asarray(idx[i:i + CHUNK])
    m = ix >= 0
    c = slot[ix[m]]
    k = c >= 0
    c = c[k]
    z, y, x = (a[k] for a in np.nonzero(m))
    codes.append(np.unique(ids[c] * (ny * nx) + y * nx + x))
    if need_props:
        w = np.asarray(cube[i:i + CHUNK], dtype=np.float64)[m][k]
        v = (z + i) * dv
        for key, q in (('w', 1), ('x', x), ('y', y), ('v', v), ('xx', x * x), ('yy', y * y), ('xy', x * y), ('vv', v * v)):
            S[key] += np.bincount(c, weights=w * q, minlength=n)
codes = np.unique(np.concatenate(codes))
cid, pix = codes // (ny * nx), codes % (ny * nx)

if need_props:
    W = S['w']
    mx, my, mv = S['x'] / W, S['y'] / W, S['v'] / W
    cxx, cyy, cxy = S['xx'] / W - mx ** 2, S['yy'] / W - my ** 2, S['xy'] / W - mx * my
    tr, det = cxx + cyy, cxx * cyy - cxy ** 2
    lam1 = tr / 2 + np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))      # major / minor axis variances, px^2
    lam2 = tr / 2 - np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    with np.errstate(invalid='ignore'):
        dmaj = np.sqrt(lam1 - sig_beam ** 2)
        dmin = np.sqrt(lam2 - sig_beam ** 2)
        R = 1.91 * np.sqrt(dmaj * dmin) * pc_px
        R_obs = 1.91 * (lam1 * lam2) ** 0.25 * pc_px
        sv = np.sqrt(np.maximum(S['vv'] / W - mv ** 2, 0) - dv ** 2 / (2 * np.pi))
    L = W * dv * pc_px ** 2
    props = pd.DataFrame(dict(cloud_idx=ids, isolated_leaf=np.isin(ids, list(added)), R_pc=R, R_obs_pc=R_obs, sigma_v_kms=sv,
                              L_CO_Kkms_pc2=L, M_Msun=ALPHA_21 * L, x=mx, y=my, v_kms=mv + hdr['CRVAL3'] / 1000))
    props.to_csv(PROPS, index=False)
props = pd.read_csv(PROPS).set_index('cloud_idx')
peak = cat.set_index('idx').vmax                                   # peak S/N of each structure


def outlines(ax, keep, color, lw):
    """One contour per cloud in `keep` around its projected footprint."""
    sel = np.isin(cid, list(keep))
    c, p = cid[sel], pix[sel]
    _, start = np.unique(c, return_index=True)
    for s in np.split(p, start[1:]):
        y, x = s // nx, s % nx
        y0, x0 = y.min() - 1, x.min() - 1
        sub = np.zeros((y.max() - y0 + 2, x.max() - x0 + 2))
        sub[y - y0, x - x0] = 1
        ax.contour(np.arange(x0, x0 + sub.shape[1]), np.arange(y0, y0 + sub.shape[0]), sub,
                   levels=[0.5], colors=[color], linewidths=lw)


quiet = np.isfinite(mom0) & (sigma < 0.1)
vmax = np.nanpercentile(mom0[quiet], 99.5)
A, C = props.loc[sorted(added)], props.loc[sorted(off.cloud_idx)]
frac = A.L_CO_Kkms_pc2.sum() / props.L_CO_Kkms_pc2.sum()
print(f'isolated leaves: {len(A)}, median peak S/N {peak.loc[A.index].median():.1f} (max {peak.loc[A.index].max():.1f}), '
      f'{100 * frac:.1f}% of total L_CO; pixel {pc_px:.2f} pc, beam sigma {sig_beam:.2f} px, dv {dv:.3f} km/s')
for lab, t in (('clustered', C), ('isolated', A)):
    print(f'  {lab}: {len(t)} clouds, resolved {t.R_pc.notna().sum()}, median R {t.R_pc.median():.1f} pc, '
          f'median sig_v {t.sigma_v_kms.median():.2f} km/s, median M {t.M_Msun.median():.3g} Msun')

fig = plt.figure(figsize=(13, 22), constrained_layout=True)
gs = fig.add_gridspec(3, 2, height_ratios=[2.3, 1, 1])
maps = [fig.add_subplot(gs[0, j], projection=wcs) for j in (0, 1)]
for ax, title in zip(maps, (f'save_isol_leaves = False\n{len(off):,} clouds (clustered only)',
                            f'save_isol_leaves = True\n{len(off):,} clustered + {len(added):,} isolated leaves '
                            f'= {len(on):,} clouds')):
    ax.imshow(mom0, origin='lower', cmap='gray_r', vmin=0, vmax=vmax, interpolation='nearest')
    outlines(ax, set(off.cloud_idx), CLUST, 0.6)
    ax.set_title(title, fontsize=11)
    ax.coords[0].set_major_formatter('hh:mm:ss'); ax.coords[1].set_major_formatter('dd:mm')
    ax.coords[0].set_ticklabel(exclude_overlapping=True)
    ax.set_xlabel('RA (J2000)')
outlines(maps[1], added, ISOL, 0.6)
maps[0].set_ylabel('Dec (J2000)'); maps[1].set_ylabel(' '); maps[1].coords[1].set_ticklabel_visible(False)
maps[1].legend(handles=[Line2D([], [], color=CLUST, lw=1.5, label='SCIMES clustered cloud'),
                        Line2D([], [], color=ISOL, lw=1.5,
                               label=f'isolated leaf (peak S/N ≤ {peak.loc[A.index].max():.0f}; '
                                     f'{100 * frac:.1f}% of L_CO)')],
               loc='lower left', fontsize=9, framealpha=0.9)

# --- rows 2-3: scaling relations, shared limits across the two columns
rplot = props.R_pc.fillna(props.R_obs_pc)                         # R if resolved, else its upper limit
ok = props.sigma_v_kms > 0
rlim = (rplot.min() / 1.3, rplot.max() * 1.3)
mlim = (props.M_Msun.min() / 2, props.M_Msun.max() * 2)
vlim = (props.sigma_v_kms[ok].min() / 1.3, props.sigma_v_kms[ok].max() * 1.3)
rr = np.geomspace(*rlim, 50)


def scatter(ax, sets, ycol):
    for t, color, lab in sets:
        r, u = t[t.R_pc.notna()], t[t.R_pc.isna()]
        ax.scatter(u.R_obs_pc, u[ycol], s=16, marker='<', facecolors='none', edgecolors=color, linewidths=0.6,
                   alpha=0.6, label=f'{lab}, unresolved: R upper limit ({len(u):,})')
        ax.scatter(r.R_pc, r[ycol], s=14, color=color, edgecolors='white', linewidths=0.4, alpha=0.9,
                   label=f'{lab}, resolved ({len(r):,})')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(rlim)
    ax.grid(color=GRID, lw=0.5, which='both'); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_xlabel('R = 1.91 σ_r (pc)')


for j, sets in enumerate(([(C, CLUST, 'clustered')], [(C, CLUST, 'clustered'), (A, ISOL, 'isolated leaf')])):
    ax = fig.add_subplot(gs[1, j])
    scatter(ax, sets, 'M_Msun')
    for sig, ls in ((100, '--'), (10, ':')):
        ax.plot(rr, sig * np.pi * rr ** 2, color=INK2, lw=0.8, ls=ls, label=f'Σ = {sig} M$_\\odot$ pc$^{{-2}}$')
    ax.set_ylim(mlim); ax.set_ylabel('M = α_CO(2-1) L_CO (M$_\\odot$)' if j == 0 else '')
    ax.legend(loc='upper left', fontsize=9, frameon=False)
    ax = fig.add_subplot(gs[2, j])
    scatter(ax, sets, 'sigma_v_kms')
    ax.plot(rr, 0.72 * rr ** 0.5, color=INK2, lw=0.8, ls='--', label='σ_v = 0.72 R$^{0.5}$ (Solomon+87)')
    ax.set_ylim(vlim); ax.set_ylabel('σ_v (km s$^{-1}$)' if j == 0 else '')
    ax.legend(loc='upper left', fontsize=9, frameon=False)

fig.get_layout_engine().set(rect=(0, 0, 1, 0.972))
fig.text(0.5, 0.997, 'SCIMES on the 5σ noise-weighted CO(2-1) dendrogram (T/σ > 5, min_delta = 2σ, min_npix = 5): '
         'effect of save_isol_leaves\nrow 1: velocity-projected cloud footprints on the zeroth moment '
         f'(0–{vmax:.1f} K km/s); rows 2–3: moment-based properties (no extrapolation), D = 859 kpc,\n'
         'α_CO(2-1) = 8.0 M$_\\odot$ (K km/s pc$^2$)$^{-1}$; open triangles: unresolved (an axis no larger than the beam), '
         'plotted at the non-deconvolved R as an upper limit',
         ha='center', va='top', fontsize=11)
fig.savefig(M33 + 'm33_co21_snr5_save_isol_leaves.png', dpi=200)
print('saved', M33 + 'm33_co21_snr5_save_isol_leaves.png')
