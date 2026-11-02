"""CO(2-1) moment maps of the SCIMES clustered clouds, in the style of plot_moment_map_sets in
ipynbs/mekuria_imara_pipeline.py (the HI maps_sets_* PDFs), with moment 2 added:

    python co_cloud_map_sets.py [N]      N = dendrogram floor in sigma, default 4 (run snrNd2)

One panel triplet per set: Sigma_H2 (moment 0 x alpha_CO), velocity (moment 1, about the set's median) and
dispersion (moment 2). Sets are ordered by total mass, heaviest first; one multi-page PDF per set size.

Moments use only the voxels assigned to the set's clouds, so they are clipped at the dendrogram floor and are
consistent with the catalog's M, R and sigma_v. Moment 2 is the raw intensity-weighted dispersion per pixel,
with no channel-width correction.

Sets are maximal cliques, as in cluster_envelopes(criterion="min"): two clouds overlap if their centroids are
closer than the smaller R. Unlike the HI sets they must also lie within VSPAN km/s, so that clouds at
different velocities on one sightline are not blended into one moment map. Clouds can be in more than one
set. R is the deconvolved radius, or the upper limit R_obs for unresolved clouds. Cloud labels C1, C2, ... rank
the clouds by mass. Cutouts are in pixel axes, as in the HI sets, so +dx is West.
"""
import sys
import numpy as np, pandas as pd, networkx as nx
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle
from astropy.io import fits
from astropy.wcs import WCS

MAPS = '/Users/thummimm/Documents/research/data/maps/'
M33 = '/Users/thummimm/Documents/research/M33/'
CUBE = MAPS + 'm33_all_co21_8p0as.K.fits'
N_SIG = sys.argv[1] if len(sys.argv) > 1 else '4'
P = MAPS + f'm33_all_co21_8p0as_snr{N_SIG}d2_parallel_dendro'
FIGDIR = M33 + 'paper/figures/'
CHUNK = 20
D_PC = 859e3
ALPHA_21 = 6.4 / 0.8                                               # Msun (K km/s pc^2)^-1
M33_CENTER = (23.4621, 30.6600)                                    # deg, as in the pipeline
M33_PA_KIN, M33_INCL = 201.9, 54.0                                 # deg
VSPAN = 5.0                                                        # km/s, largest velocity offset within a set
VPLOT = 10.0                                                       # km/s, fixed half-range of the velocity panels
SIG_MAX = 6.0                                                      # km/s, fixed top of the dispersion panels
MARGIN_PC, MIN_HALF_PC = 40.0, 60.0
NROWS, SETS_PER_ROW = 5, 2
C_SET_MAP = ['#00e5ff', '#7cff00', '#ffffff', '#ff00d4', '#ffb000', '#9a9a9a']
NAN_BG = '#262626'

hdr = fits.getheader(CUBE)
wcs = WCS(hdr).celestial
dv = abs(hdr['CDELT3']) / 1000.0                                   # km/s
v_chan = lambda z: (hdr['CRVAL3'] + (z + 1 - hdr['CRPIX3']) * hdr['CDELT3']) / 1000.0
pc_px = np.deg2rad(abs(hdr['CDELT2'])) * D_PC                      # pc per pixel
sig_beam = hdr['BMAJ'] / abs(hdr['CDELT2']) / np.sqrt(8 * np.log(2))   # px
beam_fwhm_pc = hdr['BMAJ'] / abs(hdr['CDELT2']) * pc_px

props = pd.read_csv(P + '_scimes_isol_cloudprops.csv')
t = props[~props.isolated_leaf].sort_values('M_Msun', ascending=False).reset_index(drop=True)
t['rank'] = np.arange(1, len(t) + 1)
t['R_plot'] = t.R_pc.fillna(t.R_obs_pc)
t['resolved'] = t.R_pc.notna()
t['v'] = np.nan
ids = t.cloud_idx.values

# --- all voxels of the clustered clouds, from one pass over the index map
lookup = pd.read_csv(P + '_scimes_lookup.csv').cloud_idx.values
order = np.argsort(ids)
slot = np.full(lookup.size, -1)
hit = lookup >= 0
slot[hit] = order[np.searchsorted(ids[order], lookup[hit])]        # structure -> row of t
idx = fits.open(P + '_index_map.fits', memmap=True)[0].data
cube = fits.open(CUBE, memmap=True)[0].data
vox = []
for i in range(0, idx.shape[0], CHUNK):
    ix = np.asarray(idx[i:i + CHUNK])
    m = ix >= 0
    c = slot[ix[m]]
    k = c >= 0
    z, y, x = (a[k] for a in np.nonzero(m))
    vox.append((c[k], z + i, y, x, np.asarray(cube[i:i + CHUNK], dtype=np.float64)[m][k]))
c_v, z_v, y_v, x_v, T_v = (np.concatenate(a) for a in zip(*vox))
v_v = v_chan(z_v)
s = np.argsort(c_v, kind='stable')
c_v, v_v, y_v, x_v, T_v = c_v[s], v_v[s], y_v[s], x_v[s], T_v[s]
start = np.searchsorted(c_v, np.arange(len(t) + 1))


def voxels(k):
    a, b = start[k], start[k + 1]
    return v_v[a:b], y_v[a:b], x_v[a:b], T_v[a:b]


t['v'] = [np.average(voxels(k)[0], weights=voxels(k)[3]) for k in range(len(t))]
assert np.allclose(ALPHA_21 * np.array([voxels(k)[3].sum() for k in range(len(t))]) * dv * pc_px ** 2, t.M_Msun)

# --- galactocentric position
ra, dec = wcs.celestial.wcs_pix2world(t.x, t.y, 0)
dra = (ra - M33_CENTER[0]) * np.cos(np.deg2rad(M33_CENTER[1]))     # +ve towards East
ddec = dec - M33_CENTER[1]
pa = np.arctan2(dra, ddec) - np.deg2rad(M33_PA_KIN)
d_kpc = np.deg2rad(np.hypot(dra, ddec)) * D_PC / 1e3
t['Rgal'] = d_kpc * np.hypot(np.cos(pa), np.sin(pa) / np.cos(np.deg2rad(M33_INCL)))
t['phi'] = np.rad2deg(pa) % 360

# --- overlap sets
G = nx.Graph()
G.add_nodes_from(range(len(t)))
X = t[['x', 'y']].values * pc_px
for i in range(len(t)):
    d = np.hypot(*(X - X[i]).T)
    near = (d < np.minimum(t.R_plot, t.R_plot[i])) & (abs(t.v - t.v[i]) <= VSPAN)
    G.add_edges_from((i, j) for j in np.nonzero(near.values)[0] if j > i)
sets = sorted((sorted(q) for q in nx.find_cliques(G)), key=lambda q: -t.M_Msun[q].sum())


def set_maps(q):
    """Sigma_H2, velocity and dispersion over a square cutout holding every cloud in the set."""
    v, y, x, T = (np.concatenate(a) for a in zip(*(voxels(k) for k in q)))
    half = max(MIN_HALF_PC / pc_px, 0.5 * max(np.ptp(x), np.ptp(y)) + MARGIN_PC / pc_px)
    cx, cy = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
    x0, y0 = int(np.floor(cx - half)), int(np.floor(cy - half))
    n = int(np.ceil(2 * half)) + 1
    p = (y - y0) * n + (x - x0)
    W = np.bincount(p, T, n * n).reshape(n, n)
    Wv = np.bincount(p, T * v, n * n).reshape(n, n)
    Wvv = np.bincount(p, T * v * v, n * n).reshape(n, n)
    with np.errstate(invalid='ignore', divide='ignore'):
        m1 = Wv / W
        m2 = np.sqrt(np.maximum(Wvv / W - m1 ** 2, 0))
    W[W == 0] = np.nan
    extent = np.array([x0 - 0.5 - cx, x0 + n - 0.5 - cx, y0 - 0.5 - cy, y0 + n - 0.5 - cy]) * pc_px
    return ALPHA_21 * W * dv, m1, m2, (x0, y0, cx, cy, n), extent


spans, clipped = [], []


def label(ax, text, y, colour='white'):
    a = ax.annotate(text, xy=(0.03, y), xycoords='axes fraction', fontsize=6, color=colour)
    a.set_path_effects([pe.withStroke(linewidth=1.6, foreground='black')])


def draw_set(axs, q):
    sig, m1, m2, (x0, y0, cx, cy, n), extent = set_maps(q)
    vmid = float(np.nanmedian(m1))
    kw = dict(origin='lower', extent=extent, interpolation='nearest')
    axs[0].imshow(sig, cmap='magma', vmin=0, vmax=np.nanpercentile(sig, 99), **kw)
    axs[1].imshow(m1, cmap='RdBu', vmin=vmid - VPLOT, vmax=vmid + VPLOT, **kw)
    axs[2].imshow(m2, cmap='viridis', vmin=0, vmax=SIG_MAX, **kw)
    gx = (np.arange(x0, x0 + n) - cx) * pc_px
    gy = (np.arange(y0, y0 + n) - cy) * pc_px
    for j, k in enumerate(q):
        _, y, x, _ = voxels(k)
        foot = np.zeros((n, n))
        foot[y - y0, x - x0] = 1
        colour = C_SET_MAP[j % len(C_SET_MAP)]
        for ax in axs:
            cs = ax.contour(gx, gy, foot, levels=[0.5], colors=[colour], linewidths=0.8)
            cs.set_path_effects([pe.withStroke(linewidth=1.5, foreground='black')])
        row = t.iloc[k]
        lim = '' if row.resolved else '<'
        label(axs[0], rf'C{row["rank"]}  $M$={row.M_Msun:.2g}  $R${lim}{row.R_plot:.0f} pc', 0.93 - 0.075 * j, colour)
        label(axs[1], rf'$R_{{gal}}$={row.Rgal:.1f} kpc  $\phi$={row.phi:.0f}$^\circ$', 0.93 - 0.075 * j, colour)
    r0 = extent[0] + 0.75 * (extent[1] - extent[0])
    axs[0].add_patch(Circle((r0 + 0.13 * (extent[1] - extent[0]), extent[2] + 0.13 * (extent[3] - extent[2])),
                            beam_fwhm_pc / 2, fc='none', ec='white', lw=0.8))
    label(axs[0], rf'$\Sigma$ {np.nanmin(sig):.0f} to {np.nanmax(sig):.0f}', 0.04)
    clip = np.nanmax(abs(m1 - vmid)) > VPLOT
    spans.append(np.nanmax(m1) - np.nanmin(m1))
    clipped.append(clip)
    label(axs[1], rf'$v$ {np.nanmin(m1):.0f} to {np.nanmax(m1):.0f}' + (' (clipped)' if clip else ''), 0.04)
    label(axs[2], rf'$\sigma$ {np.nanmin(m2):.1f} to {np.nanmax(m2):.1f}', 0.04)
    for ax in axs:
        ax.set_facecolor(NAN_BG)
        ax.tick_params(labelsize=6)


for size in sorted({len(q) for q in sets}):
    group = [q for q in sets if len(q) == size]
    path = FIGDIR + f'co_maps_snr{N_SIG}_sets_size{size}.pdf'
    per_page = NROWS * SETS_PER_ROW
    npages = int(np.ceil(len(group) / per_page))
    with PdfPages(path) as pdf:
        for p0 in range(0, len(group), per_page):
            page = group[p0:p0 + per_page]
            fig, axes = plt.subplots(NROWS, 3 * SETS_PER_ROW, figsize=(2.45 * 3 * SETS_PER_ROW, 2.5 * NROWS))
            slots = [axes[i, 3 * j:3 * j + 3] for i in range(NROWS) for j in range(SETS_PER_ROW)]
            for axs, q in zip(slots, page):
                draw_set(axs, q)
            for axs in slots[len(page):]:
                for ax in axs:
                    ax.set_axis_off()
            fig.suptitle(rf'{N_SIG}$\sigma$ SCIMES clouds, CO(2-1), heaviest first: $\Sigma_{{H_2}}$ [M$_\odot$ pc$^{{-2}}$], '
                         rf'velocity ($\pm${VPLOT:.0f} km/s), dispersion (0-{SIG_MAX:.0f} km/s); sets of {size}; '
                         f'beam at lower right; page {p0 // per_page + 1} of {npages}', fontsize=10)
            fig.supxlabel(r'$\Delta x$ [pc]', fontsize=9)
            fig.supylabel(r'$\Delta y$ [pc]', fontsize=9)
            fig.tight_layout(rect=(0.01, 0.01, 1, 0.975))
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    print(f'{path}: {len(group)} sets, {npages} pages', flush=True)
print(f'{len(t)} clouds in {len(sets)} sets; {len(set(k for q in sets for k in q))} clouds shown')
print(f'moment-1 span within a cloud [km/s]: median {np.median(spans):.1f}, p90 {np.percentile(spans, 90):.1f}, '
      f'max {np.max(spans):.1f}; clipped at +/-{VPLOT:.0f} km/s: {sum(clipped)}')
top = np.array(spans[:30])
print(f'30 heaviest: median span {np.median(top):.1f} km/s, {sum(clipped[:30])} clipped')
