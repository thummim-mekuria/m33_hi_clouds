"""M33 DSS2-Red image overlaid with the footprints of published molecular-line surveys.

Self-contained: does the TAN (gnomonic) projection with numpy only, so it does not
depend on astropy (the local astropy 7.1 / numpy 2.4 pairing is broken -- astropy
still references the removed np.in1d).

Reproducing the figure
----------------------
    /usr/local/bin/python3 footprints.py

NOT plain `python3` on this machine: that resolves to pyenv 3.5.10, which has no
numpy or matplotlib. Requires only numpy and matplotlib; the DSS cutout is fetched
from SkyView automatically if m33_dss.fits is not already next to this script.

Verified with: Python 3.11.1, numpy 2.4.6, matplotlib 3.10.0.
"""
import os
import sys

if sys.version_info < (3, 6):
    sys.exit(
        "footprints.py needs Python >= 3.6 (got %d.%d).\n"
        "On this machine use: /usr/local/bin/python3 footprints.py"
        % sys.version_info[:2]
    )

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Circle, Patch
import matplotlib.patheffects as pe
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

S = os.path.dirname(os.path.abspath(__file__))   # inputs/outputs sit next to this script
D2R = np.pi / 180.0

# ---------------------------------------------------------------- FITS + WCS
def read_fits(path):
    buf = open(path, "rb").read()
    hdr, off = {}, 0
    while True:
        block = buf[off:off + 2880].decode("ascii", "replace")
        off += 2880
        done = False
        for i in range(0, 2880, 80):
            card = block[i:i + 80]
            key = card[:8].strip()
            if key == "END":
                done = True
                break
            if card[8:10] == "= ":
                val = card[10:].split("/")[0].strip().strip("'").strip()
                try:
                    hdr[key] = float(val) if any(c in val for c in ".eE-+") or val.isdigit() else val
                except ValueError:
                    hdr[key] = val
        if done:
            break
    n1, n2 = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
    data = np.frombuffer(buf[off:off + n1 * n2 * 4], dtype=">f4").reshape(n2, n1).astype(float)
    return hdr, data

# Exact SkyView query behind m33_dss.fits: DSS2 Red, 1.9 deg field, 1400 px,
# centred on the SIMBAD position for M33 (23.4621, +30.6602).
FITS_PATH = os.path.join(S, "m33_dss.fits")
SKYVIEW_URL = (
    "https://skyview.gsfc.nasa.gov/current/cgi/pskcall"
    "?Survey=DSS2+Red&Position=23.4621,30.6602&Size=1.9&Pixels=1400&Return=FITS"
)

def fetch_cutout(url, dest):
    """Download the DSS cutout. Tries urllib with certifi (the python.org framework
    build ships no root certificates), then falls back to curl."""
    import ssl
    import urllib.request
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(url, context=ctx, timeout=180) as r, \
                open(dest, "wb") as out:
            out.write(r.read())
        return "urllib"
    except Exception as exc:
        sys.stderr.write("  urllib failed (%s); falling back to curl\n" % exc)
        import subprocess
        subprocess.run(["curl", "-sSL", "--max-time", "180", "-o", dest, url], check=True)
        return "curl"


if not os.path.exists(FITS_PATH):
    sys.stderr.write("m33_dss.fits not found -- fetching 7.9 MB cutout from SkyView...\n")
    how = fetch_cutout(SKYVIEW_URL, FITS_PATH)
    sys.stderr.write("wrote %s via %s\n" % (FITS_PATH, how))

HDR, IMG = read_fits(FITS_PATH)
A0, D0 = HDR["CRVAL1"] * D2R, HDR["CRVAL2"] * D2R
P1, P2 = HDR["CRPIX1"] - 1.0, HDR["CRPIX2"] - 1.0
C1, C2 = HDR["CDELT1"], HDR["CDELT2"]
ARCMIN_PIX = abs(C2) * 60.0                       # arcmin per pixel

def w2p(ra_deg, dec_deg):
    """Gnomonic world -> pixel (0-indexed)."""
    a, d = np.asarray(ra_deg) * D2R, np.asarray(dec_deg) * D2R
    cosc = np.sin(D0) * np.sin(d) + np.cos(D0) * np.cos(d) * np.cos(a - A0)
    xi = np.cos(d) * np.sin(a - A0) / cosc
    eta = (np.cos(D0) * np.sin(d) - np.sin(D0) * np.cos(d) * np.cos(a - A0)) / cosc
    return P1 + (xi / D2R) / C1, P2 + (eta / D2R) / C2

# ---------------------------------------------------------------- M33 geometry
D_KPC      = 840.0
ARCMIN_KPC = np.degrees(1.0 / D_KPC) * 60.0        # 4.093 arcmin per kpc
PA_DISK    = 22.5                                  # deg E of N  (Druard+14, Corbelli+17)
AXR        = np.cos(np.radians(56.0))              # i = 56 deg
CX, CY     = P1, P2                                # SkyView resolved the M33 centre

def px(arcmin):
    return arcmin / ARCMIN_PIX

def ellipse(a_arcmin, axr, pa=PA_DISK, **kw):
    return Ellipse((CX, CY), 2 * px(a_arcmin), 2 * px(a_arcmin * axr),
                   angle=90.0 + pa, **kw)

def rect(length_arcmin, width_arcmin, pa=PA_DISK, **kw):
    L, W = px(length_arcmin) / 2.0, px(width_arcmin) / 2.0
    t = np.radians(pa)
    ux, uy = -np.sin(t), np.cos(t)          # along PA   (N = +y, E = -x)
    vx, vy = -uy, ux                        # perpendicular
    pts = [(CX + s * L * ux + q * W * vx, CY + s * L * uy + q * W * vy)
           for s, q in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    return Polygon(pts, closed=True, **kw)

def sky_label(txt, r_arcmin, pa, color, dx=0, dy=0, ha="center", va="center", size=10.5):
    t = np.radians(pa)
    ax.text(CX - np.sin(t) * px(r_arcmin) + dx, CY + np.cos(t) * px(r_arcmin) + dy,
            txt, color=color, fontsize=size, ha=ha, va=va, weight="bold", zorder=14,
            path_effects=[pe.withStroke(linewidth=2.8, foreground="black")])

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(12.6, 12.6))

lo, hi = np.nanpercentile(IMG, 22), np.nanpercentile(IMG, 99.88)
x = np.clip((IMG - lo) / (hi - lo), 0, 1)
a = 0.05
ax.imshow(np.arcsinh(x / a) / np.arcsinh(1 / a), origin="lower", cmap="gray",
          vmin=0, vmax=1, interpolation="bilinear", zorder=0)

C = dict(iram="#4FA8DE", hera="#A8DCF5", bima="#FF6B4A", fcrao="#FFC145",
         nro="#C77DFF", ros="#5FD08A", aca="#FF4D8D", ref="#FFFFFF", lm="#FFE66D")

HERA_LEN = 2 * 8.5 * ARCMIN_KPC                    # 8.5 kpc radius along major axis
HERA_WID = 650.0 / HERA_LEN                        # 650 arcmin^2 total

# survey, beam (angular . physical), map pixel/grid, sky extent, reference.
# Values are each paper's own; "--" where the paper does not state a pixel grid.
LEG = [
    ("iram",  "IRAM 30 m",  "12\u2033 \u00b7 50 pc",        "3.0\u2033",  "R < 7 kpc",     "Druard 2014 / Corbelli 2017"),
    ("hera",  "IRAM HERA",  "12\u2033 \u00b7 50 pc",        "3.8\u2033",  "R < 8.5 kpc",   "Gratier 2010"),
    ("bima",  "BIMA",       "13\u2033 \u00b7 50 pc",        "4.0\u2033",  "H\u03b1 disk",      "Engargiola 2003"),
    ("fcrao", "FCRAO 14 m", "50\u2033 \u00b7 204 pc",      "22.1\u2033",  "r < 15\u2032",       "Heyer 2004"),
    ("nro",   "NRO MAGiC",  "19.3\u2033 \u00b7 81 pc",      "7.5\u2033",  "30\u2032 \u00d7 30\u2032",     "Tosaki 2011"),
    ("ros",   "BIMA+FCRAO", "13\u2033 \u00b7 53 pc",           "--",  "R < 5.5 kpc",   "Rosolowsky 2007"),
    ("aca",   "ALMA-ACA",   "7.3\u2033\u00d76.5\u2033 \u00b7 30 pc",     "--",  "1100\u2033 \u00d7 1180\u2033", "Muraoka 2023 / Konishi 2024"),
]
ROW = "%-11s %-18s %6s   %-14s %s"
HEADER = ROW % ("instrument", "beam", "pixel", "sky extent", "reference")

shapes = {
    "iram":  ellipse(7.0 * ARCMIN_KPC, AXR, fc="none", ec=C["iram"], lw=2.8, zorder=6),
    "hera":  rect(HERA_LEN, HERA_WID, fc="none", ec=C["hera"], lw=2.0, ls=(0, (7, 3)), zorder=6),
    "bima":  ellipse(5.8 * ARCMIN_KPC, np.cos(np.radians(51.0)), pa=21.0,
                     fc="none", ec=C["bima"], lw=2.5, zorder=7),
    "fcrao": Circle((CX, CY), px(15.0), fc="none", ec=C["fcrao"], lw=2.2, ls=(0, (9, 4)), zorder=6),
    "nro":   rect(30.0, 30.0, pa=0.0, fc="none", ec=C["nro"], lw=2.2, ls=(0, (6, 2, 1, 2)), zorder=7),
    "ros":   ellipse(5.5 * ARCMIN_KPC, AXR, fc="none", ec=C["ros"], lw=1.9, ls=(0, (1, 2)), zorder=7),
    "aca":   rect(1180 / 60.0, 1100 / 60.0, pa=0.0, fc="none", ec=C["aca"], lw=2.8, zorder=9),
}
entries = [(Line2D([], [], linestyle="none"), HEADER)] + \
          [(shapes[k], ROW % tuple(row)) for k, *row in LEG]

ax.add_patch(rect(1180 / 60.0, 1100 / 60.0, pa=0.0, fc=C["aca"], ec="none",
                  alpha=0.11, zorder=8))
ax.add_patch(rect(40.0, 2.0, fc="none", ec=C["hera"], lw=1.1, ls=(0, (2, 2)), zorder=6))
ax.add_patch(ellipse(30.8, AXR, fc="none", ec=C["ref"], lw=1.0, ls=(0, (3, 5)),
                     alpha=0.5, zorder=5))
for p, _ in entries:
    if isinstance(p, Patch):          # skip the invisible legend-header proxy
        ax.add_patch(p)

sky_label("IRAM 30 m — R < 7 kpc",   7.0 * ARCMIN_KPC, PA_DISK,     C["iram"], dy=px(1.5))
sky_label("HERA strip — R < 8.5 kpc", 8.5 * ARCMIN_KPC, PA_DISK + 180, C["hera"], dy=-px(1.7))
sky_label("BIMA all-disk",           5.8 * ARCMIN_KPC, PA_DISK + 90, C["bima"], dx=-px(1.2), ha="right")
sky_label("FCRAO  r < 15′",          15.0,             PA_DISK + 90, C["fcrao"], dx=px(1.2), ha="left")
sky_label("NRO MAGiC",               21.2,             150,          C["nro"])
sky_label("ALMA-ACA",                11.4,             0,            C["aca"])
sky_label("Rosolowsky 2007",         5.5 * ARCMIN_KPC, PA_DISK + 180, C["ros"], size=9.5, dy=-px(1.2))
sky_label("$R_{25}$ = 30.8′",        30.8,             PA_DISK - 90, C["ref"], dx=-px(1.0),
          ha="right", size=9)

for name, ra, dec in [("NGC 604", (1 + 34/60 + 32.5/3600) * 15, 30 + 47/60 + 6/3600),
                      ("NGC 595", (1 + 33/60 + 33.5/3600) * 15, 30 + 41/60 + 30/3600)]:
    qx, qy = w2p(ra, dec)
    ax.plot(qx, qy, "o", ms=10, mfc="none", mec=C["lm"], mew=1.8, zorder=11)
    ax.text(qx + px(1.0), qy, name, color=C["lm"], fontsize=9, va="center", ha="left",
            zorder=12, path_effects=[pe.withStroke(linewidth=2.6, foreground="black")])

# ---------------------------------------------------------------- coordinate grid
dec_ticks = np.arange(29.75, 31.76, 0.25)
ra_ticks = np.arange(22.5, 24.76, 0.5)
dec_line = np.linspace(HDR["CRVAL2"] - 1.15, HDR["CRVAL2"] + 1.15, 200)
ra_line = np.linspace(HDR["CRVAL1"] - 1.4, HDR["CRVAL1"] + 1.4, 200)
gk = dict(color="#8d9aa3", ls=":", lw=0.6, alpha=0.5, zorder=3)
for r in ra_ticks:
    ax.plot(*w2p(np.full_like(dec_line, r), dec_line), **gk)
for d in dec_ticks:
    ax.plot(*w2p(ra_line, np.full_like(ra_line, d)), **gk)

N = IMG.shape[0]
ax.set_xlim(0, N); ax.set_ylim(0, N)
xt = [w2p(r, HDR["CRVAL2"] - 0.92)[0] for r in ra_ticks]
xl = ["%dh%02dm" % (int(r / 15), round((r / 15 - int(r / 15)) * 60)) for r in ra_ticks]
yt = [w2p(HDR["CRVAL1"], d)[1] for d in dec_ticks]
yl = ["%d°%02d′" % (int(d), round((d - int(d)) * 60)) for d in dec_ticks]
keep = [i for i, v in enumerate(xt) if 0 < v < N]
ax.set_xticks([xt[i] for i in keep]); ax.set_xticklabels([xl[i] for i in keep])
keep = [i for i, v in enumerate(yt) if 0 < v < N]
ax.set_yticks([yt[i] for i in keep]); ax.set_yticklabels([yl[i] for i in keep])
ax.set_xlabel("Right Ascension (J2000)", fontsize=11)
ax.set_ylabel("Declination (J2000)", fontsize=11)
ax.tick_params(labelsize=9.5)

# scale bar + compass
bar = px(5.0 * ARCMIN_KPC)
bx, by = 0.055 * N, 0.05 * N
ax.plot([bx, bx + bar], [by, by], color="w", lw=3.5, solid_capstyle="butt", zorder=13)
ax.text(bx + bar / 2, by + px(1.1), "5 kpc  =  20.5′", color="w", ha="center", fontsize=10,
        zorder=13, path_effects=[pe.withStroke(linewidth=2.8, foreground="black")])
ax_, nx, ny = px(6.0), 0.935 * N, 0.06 * N
for dxa, dya, s, la, lb in [(0, ax_, "N", 0, ax_ + px(1.6)), (-ax_, 0, "E", -ax_ - px(1.8), 0)]:
    ax.annotate("", xy=(nx + dxa, ny + dya), xytext=(nx, ny),
                arrowprops=dict(arrowstyle="-|>", color="w", lw=1.8), zorder=13)
    ax.text(nx + la, ny + lb, s, color="w", ha="center", va="center", fontsize=11,
            weight="bold", zorder=13,
            path_effects=[pe.withStroke(linewidth=2.6, foreground="black")])

mono = FontProperties(family="monospace", size=8.6)
mono_hd = FontProperties(family="monospace", size=8.6, weight="bold")
leg = ax.legend([p for p, _ in entries], [t for _, t in entries], loc="upper left",
                framealpha=0.85, facecolor="#0b0f12", edgecolor="#3a4750",
                labelcolor=["#93a2ac"] + ["white"] * len(LEG),
                handlelength=3.0, handletextpad=1.1,
                borderpad=0.9, labelspacing=0.75, prop=mono)
leg.get_texts()[0].set_fontproperties(mono_hd)
leg.set_zorder(20)

ax.set_title("Molecular-line survey footprints on M33", fontsize=15.5, pad=13, weight="bold")
fig.text(0.5, 0.045,
         "DSS2 Red. Footprints drawn from each paper's stated coverage; adopted disk geometry "
         "PA = 22.5°, i = 56°, D = 840 kpc (BIMA at its own PA = 21°, i = 51°).\n"
         "Legend gives each map's beam and its pixel/grid spacing as quoted in the paper; “--” where the paper states none.\n"
         "Shapes are schematic representations of the quoted areas — the NRO and ALMA-ACA "
         "rectangles are drawn centred and axis-aligned. Single-GMC ALMA pointings\n"
         "(Tokuda 2020, Muraoka 2020, Kondo 2021, Sano 2021) and the CARMA outer-disk fields "
         "(Bigiel 2010) are ≲1′ across and are not drawn.",
         ha="center", fontsize=8.6, color="#3f4c55", linespacing=1.5)

fig.subplots_adjust(left=0.085, right=0.975, top=0.945, bottom=0.135)
fig.savefig(os.path.join(S, "m33_survey_footprints.png"), dpi=170, facecolor="white")
fig.savefig(os.path.join(S, "m33_survey_footprints.pdf"), facecolor="white")
print("saved:", ARCMIN_PIX * 60, "arcsec/pix ; HERA strip %.1f' x %.1f'" % (HERA_LEN, HERA_WID))
