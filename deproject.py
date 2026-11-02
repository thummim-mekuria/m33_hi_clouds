"""Deproject the M33 VLA H I moment-0 map into the face-on disk plane."""
import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import matplotlib.patheffects as pe
from scipy.ndimage import map_coordinates

HERE = os.path.dirname(os.path.abspath(__file__))   # inputs/outputs relative to this file
CUBE = os.path.join(HERE, "data", "m33_C+D+tp_himidres_120pc.fits")
OUT  = HERE

# ---- disk geometry (Druard+14 / Corbelli+17 convention, same as the footprint figure)
D_KPC, PA, INC = 840.0, 22.5, 56.0
COSI = np.cos(np.radians(INC))
PC_PER_ARCSEC = D_KPC * 1e3 / 206265.0          # 4.072 pc/arcsec at 840 kpc

# ---------------------------------------------------------------- header
raw = open(CUBE, "rb").read(2880 * 12)
end = raw.find(b"END     ")
txt = raw[:end].decode("ascii", "replace")
CARDS = [txt[i:i + 80] for i in range(0, len(txt), 80)]   # FITS cards carry no newlines
def key(k, cast=float):
    for c in CARDS:
        if c[:8].strip() == k and c[8:10] == "= ":
            return cast(c[10:].split("/")[0].strip())
    raise KeyError(k)
N1, N2, N3 = int(key("NAXIS1")), int(key("NAXIS2")), int(key("NAXIS3"))
CRPIX1, CRPIX2 = key("CRPIX1"), key("CRPIX2")
CDELT1, CDELT2 = key("CDELT1"), key("CDELT2")
DV = abs(key("CDELT3")) / 1000.0                 # km/s per channel
BMAJ = key("BMAJ") * 3600.0                      # arcsec
hdr_blocks = (end // 2880) + 1
offset = hdr_blocks * 2880
print("cube %dx%dx%d  pix=%.2f\"  dv=%.2f km/s  beam=%.2f\"" %
      (N1, N2, N3, abs(CDELT2) * 3600, DV, BMAJ))

# ---------------------------------------------------------------- moment 0
cube = np.memmap(CUBE, dtype=">f4", mode="r", offset=offset, shape=(N3, N2, N1))

edge = np.concatenate([np.asarray(cube[:8, ::4, ::4], float),
                       np.asarray(cube[-8:, ::4, ::4], float)])
sigma = np.nanstd(edge[np.isfinite(edge)])
print("channel rms = %.4f K  (from 16 line-free channels)" % sigma)

mom0 = np.zeros((N2, N1))
for k in range(N3):
    pl = np.asarray(cube[k], dtype=float)
    np.nan_to_num(pl, copy=False)
    mom0 += np.where(pl > 3 * sigma, pl, 0.0)
mom0 *= DV                                        # K km/s
print("moment0: max %.1f K km/s, %.1f%% of pixels non-zero" %
      (mom0.max(), 100.0 * (mom0 > 0).mean()))

# ---------------------------------------------------------------- deprojection
# Work in sky-offset (xi, eta) degrees; SIN distortion over this 1.3 deg field
# is <0.1 pixel, so the linear offset plane is exact enough.
def sky_offsets_to_pix(xi_deg, eta_deg):
    return (CRPIX1 - 1 + xi_deg / CDELT1,
            CRPIX2 - 1 + eta_deg / CDELT2)

RMAX, STEP = 9.0, abs(CDELT2) * 3600 * PC_PER_ARCSEC / 1e3   # kpc per native pixel
gx = np.arange(-RMAX, RMAX + STEP, STEP)
X, Y = np.meshgrid(gx, gx)                        # face-on plane, kpc

phi = np.radians(PA)
xp = X * COSI                                     # squash back onto the sky
yp = Y
d_per_kpc = np.degrees(1.0 / D_KPC)               # deg per kpc
xi  = ( xp * np.cos(phi) + yp * np.sin(phi)) * d_per_kpc
eta = (-xp * np.sin(phi) + yp * np.cos(phi)) * d_per_kpc
px, py = sky_offsets_to_pix(xi, eta)
faceon = map_coordinates(mom0, [py, px], order=1, mode="constant", cval=np.nan)

# ---------------------------------------------------------------- verification
def _moments(w, Xg, Yg):
    w = np.nan_to_num(w, nan=0.0).copy()
    w[w < 0] = 0
    tot = w.sum()
    cx, cy = (w * Xg).sum() / tot, (w * Yg).sum() / tot
    dx, dy = Xg - cx, Yg - cy
    cov = np.array([[(w*dx*dx).sum()/tot, (w*dx*dy).sum()/tot],
                    [(w*dx*dy).sum()/tot, (w*dy*dy).sum()/tot]])
    ev, evec = np.linalg.eigh(cov)
    b, a = np.sqrt(np.abs(ev))
    return b / a, np.degrees(np.arctan2(evec[0, 1], evec[1, 1]))


def verify():
    """Flux-weighted axis ratio in matched physical apertures.

    A correct deprojection takes the sky-frame ratio (~cos i) to ~1 face-on.
    The aperture MUST be defined on deprojected radius in BOTH frames -- a
    circular cut on the sky truncates the major axis and rounds the answer.
    """
    ny, nx = mom0.shape
    sx = (np.arange(nx) - (CRPIX1 - 1)) * CDELT1 / d_per_kpc
    sy = (np.arange(ny) - (CRPIX2 - 1)) * CDELT2 / d_per_kpc
    SX, SY = np.meshgrid(sx, sy)
    ph = np.radians(PA)
    xmin =  SX * np.cos(ph) - SY * np.sin(ph)
    ymaj =  SX * np.sin(ph) + SY * np.cos(ph)
    Rgal = np.sqrt((xmin / COSI) ** 2 + ymaj ** 2)
    GX, GY = np.meshgrid(gx, gx)
    print("\n  aperture        sky b/a   sky PA    face-on b/a   (cos i = %.3f)" % COSI)
    for rcut in (4.0, 6.0, 8.0):
        m = mom0.copy(); m[Rgal > rcut] = 0
        f = faceon.copy(); f[GX**2 + GY**2 > rcut**2] = 0
        qs, ps = _moments(m, SX, SY)
        qd, _ = _moments(f, GX, GY)
        print("  R_gal < %.0f kpc    %.3f    %+5.1f deg     %.3f" % (rcut, qs, ps, qd))
    print("  (face-on PA is unconstrained once b/a ~ 1)\n")


verify()

# ---------------------------------------------------------------- plot
fig, axes = plt.subplots(1, 2, figsize=(16.5, 8.4))
vmax = np.nanpercentile(mom0[mom0 > 0], 99.5)
kw = dict(cmap="magma", vmin=0, vmax=vmax, origin="lower", interpolation="nearest")

# --- observed
ny, nx = mom0.shape
ext_x = ((np.array([0, nx]) - (CRPIX1 - 1)) * CDELT1 * 60)[::-1]   # arcmin, E left
ext_y = (np.array([0, ny]) - (CRPIX2 - 1)) * CDELT2 * 60
a0 = axes[0]
a0.imshow(mom0[:, ::-1], extent=[ext_x[0], ext_x[1], ext_y[0], ext_y[1]], **kw)
a0.set_xlim(40, -40); a0.set_ylim(-40, 40)
a0.set_xlabel("ΔRA  [arcmin]  (E ←)"); a0.set_ylabel("ΔDec  [arcmin]")
a0.set_title("Observed — on the sky\ninclined disk, circular 26.6″ beam", fontsize=12)
a0.add_patch(Circle((32, -33), BMAJ / 60 / 2, fc="w", ec="w"))
a0.text(32, -35.6, 'beam 26.6"', color="w", ha="center", fontsize=9,
        path_effects=[pe.withStroke(linewidth=2.4, foreground="k")])
for r_kpc in (2, 4, 6, 8):                        # deprojected radii -> ellipses on sky
    rr = r_kpc * d_per_kpc * 60
    a0.add_patch(Ellipse((0, 0), 2 * rr, 2 * rr * COSI, angle=90 + PA,
                         fc="none", ec="#7FE7FF", lw=1.0, ls=(0, (5, 4)), alpha=0.85))

# --- deprojected
a1 = axes[1]
a1.imshow(faceon[:, ::-1], extent=[RMAX, -RMAX, -RMAX, RMAX], **kw)
a1.set_xlabel("X  [kpc]"); a1.set_ylabel("Y  [kpc]  (along major axis)")
a1.set_title("Deprojected — face-on disk plane\nPA 22.5°, i 56°, stretch 1/cos i = 1.79",
             fontsize=12)
for r_kpc in (2, 4, 6, 8):
    a1.add_patch(Circle((0, 0), r_kpc, fc="none", ec="#7FE7FF", lw=1.0,
                        ls=(0, (5, 4)), alpha=0.85))
    a1.text(0, r_kpc, "%d kpc" % r_kpc, color="#7FE7FF", fontsize=8,
            ha="center", va="bottom",
            path_effects=[pe.withStroke(linewidth=2.2, foreground="k")])
# the beam deprojects into an ellipse
bmaj_kpc = BMAJ * PC_PER_ARCSEC / 1e3
a1.add_patch(Ellipse((7.0, -7.4), bmaj_kpc, bmaj_kpc / COSI, angle=0,
                     fc="w", ec="w"))
a1.text(7.0, -8.3, "beam\n%.0f × %.0f pc" % (bmaj_kpc * 1e3, bmaj_kpc / COSI * 1e3),
        color="w", ha="center", fontsize=8.5,
        path_effects=[pe.withStroke(linewidth=2.4, foreground="k")])

for a in axes:
    a.set_aspect("equal")
    a.tick_params(colors="0.25")
cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap="magma"),
                  ax=axes, fraction=0.028, pad=0.02)
cb.set_label("H I integrated intensity  [K km s⁻¹]")

fig.suptitle("M33 H I (VLA C+D+TP, Rosolowsky) — sky frame vs deprojected disk plane",
             fontsize=14, weight="bold", y=0.98)
fig.text(0.5, 0.015,
         "Deprojection rotates by PA and stretches the minor axis by 1/cos i = 1.79. "
         "Dashed rings are galactocentric radii: ellipses on the sky, circles face-on.\n"
         "Note the beam: circular 26.6″ (108 pc at 840 kpc) on the sky becomes 108 × 193 pc "
         "face-on. Deprojection stretches pixels, it does not add resolution.",
         ha="center", fontsize=9, color="0.35")
fig.subplots_adjust(left=0.06, right=0.90, top=0.88, bottom=0.13, wspace=0.18)
fig.savefig(os.path.join(OUT, "m33_hi_deprojected.png"), dpi=150, facecolor="white")
print("saved figure")
