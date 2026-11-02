#!/usr/bin/env python3
"""
Mekuria & Imara (2026) kinematic / angular-momentum pipeline applied to M33 GMCs.

Follow-up to "Kinematics of HI Envelopes Associated with Molecular Clouds"
(Mekuria & Imara 2026, AJ 171, 4). Instead of Solar-neighbourhood clouds vs.
Galactic HI4PI, this runs the identical HI-side analysis on the M33 GMC catalog
of Corbelli et al. (2017, Table 5) using the C+D+tp 120 pc HI cube.

For each GMC we execute, in order, the analyses of the paper (HI side only):

  1.  R_HI  -- accumulation-cylinder radius from the molecular mass (Sec. defn)
  2.  aperture -- project the cylinder onto the sky, build a circular mask
  3.  kinematic association -- Gaussian-fit the aperture spectrum -> v_bar, sigma;
      keep channels within v_bar +/- 2 sigma (paper's HI selection)
  4.  M_HI -- moment-0 over the associated window, optically-thin 21-cm
  5.  velocity field -- intensity-weighted first moment
  6.  Omega_HI, theta_HI -- least-squares plane fit v = v0 + Wx dx + Wy dy
  7.  rho_HI -- Pearson r of (distance along gradient) vs. v_LSR  (planarity test)
  8.  j_HI  -- specific angular momentum j = 0.4 R^2 Omega

Output mirrors Table 2 of the paper (HI columns only):
  Name, M_HI [1e4 Msun], R_HI [pc], Omega_HI [1e-2 km/s/pc],
  theta_HI [deg], rho_HI, j_HI [1e24 cm^2/s]

Figures (reproductions of Mekuria & Imara 2026 for the M33 sample):

    plot_field_map            Fig. 1   Sigma_HI field with the envelope apertures
    plot_envelope_spectrum    Fig. 2   aperture spectrum + association window
    plot_envelope_spectrum_grid Fig. 2 the same, as a grid over several clouds
    plot_cloud_maps           Fig. 3   Sigma_HI map and HI velocity field
    plot_velocity_profile     Fig. 5   <v_LSR> vs r_perp, planarity test
    plot_velocity_profile_grid Fig. 5  the same, as a grid over several clouds
    plot_pearson_summary      Fig. 6   rho distribution, theta vs rho
    plot_angular_momentum     Fig. 7   |j| distribution and direction
    plot_j_vs_size            Fig. 9   Omega and j vs R, with fits and 1-sigma bands
    plot_size_angular_momentum Fig. 10 size-j relation incl. literature data
    plot_position_angles      Fig. 11  theta vs galactocentric radius (adapted)

make_figures(tab, cube) renders the whole set to ../paper/figures; run() does
it too when called with figures=True (--figures on the command line).

Figures 4 and 8 of the paper need the molecular velocity gradient and are not
implemented: this project has no CO cube.

Usage:
    python3 mekuria_imara_pipeline.py [--nmax N] [--snr S] [--out FILE]
                                      [--figures] [--figdir DIR]
"""

import argparse
import warnings
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as path_effects
from scipy.ndimage import map_coordinates, uniform_filter
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import astropy.constants as const

warnings.simplefilter("ignore")

# ----------------------------------------------------------------------------
# Configuration -- edit paths / constants here
# ----------------------------------------------------------------------------
HI_CUBE   = "../data/m33_C+D+tp_himidres_120pc.fits" #"/Volumes/imara-lab/M33/HI/M33_A+B+C+D_HI_0p4kms_feathered_K_reimaged.fits"
CATALOG   = "corbelli_tab5.txt"
D_M33     = 859.0 * u.kpc          # distance used in the notebook (859000 pc)
FGND_CHAN = 60                     # first 60 channels = MW foreground (notebook [60:])
N_HI_WNM  = 0.57                   # cm^-3, warm-neutral-medium density (paper)
MU        = 1.36                   # mean molecular weight incl. He (paper mass defn)
C_MOI     = 0.4                    # moment-of-inertia coeff for uniform sphere (paper)
NHI_COEFF = 1.82e18               # cm^-2 / (K km/s), optically-thin 21-cm
FIGDIR    = "../paper/figures"     # where make_figures() writes (rel. to cwd)
M33_CENTER = (23.4621, 30.6600)   # deg, M33 centre (matches the cube CRVAL)
M33_PA_KIN = 201.9                # deg E of N, kinematic major axis, receding
                                  # side (Corbelli+2014, Kam+2017)
M33_INCL   = 54.0                 # deg, disk inclination, adopted from the
                                  # literature (Corbelli+2014; Kam+2017 give
                                  # 55). A direct fit to this HI cube's own
                                  # axis ratio gives 56 (see the 2026-08-25c
                                  # notebook entry); the literature value is
                                  # kept for comparability with the CO work.
                                  # The 2 deg spread moves 1/cos(i) by 3%.


# ----------------------------------------------------------------------------
# Analysis primitives (each maps to one step in the paper)
# ----------------------------------------------------------------------------
def R_HI_cylinder(M_H2_Msun, n_HI=N_HI_WNM, mu=MU):
    """[1] Accumulation-cylinder radius R_HI = (M_H2 / (2 pi n_HI mu m_p))**(1/3)."""
    M = M_H2_Msun * u.Msun
    den = 2 * np.pi * (n_HI * u.cm**-3) * mu * const.m_p
    return ((M / den) ** (1 / 3)).to(u.pc)


def gaussian(v, amp, mean, sigma):
    return amp * np.exp(-0.5 * ((v - mean) / sigma) ** 2)


def associate_velocity(spec, vax, snr_min=5.0):
    """[3] Gaussian-fit the aperture-summed spectrum -> (v_bar, sigma).

    Returns (None, None) if the profile is too weak or the fit fails.
    """
    good = np.isfinite(spec)
    if good.sum() < 5:
        return None, None
    v, s = vax[good], spec[good]
    # edge channels ~ noise. Use whichever end is quieter: depending on the cube's
    # velocity coverage the leading edge can still contain M33 emission.
    nedge = max(5, len(s) // 5)
    noise = min(np.nanstd(s[:nedge]), np.nanstd(s[-nedge:]))
    if noise <= 0 or np.nanmax(s) / noise < snr_min:
        return None, None
    p0 = [np.nanmax(s), v[np.nanargmax(s)], 8.0]
    try:
        popt, _ = curve_fit(gaussian, v, s, p0=p0,
                            bounds=([0, v.min(), 1.0], [np.inf, v.max(), 60.0]))
    except Exception:
        return None, None
    return popt[1], abs(popt[2])


def moment0(subcube, vchan):
    """Integrated intensity map [K km/s] over the associated channels."""
    return np.nansum(subcube, axis=0) * np.abs(vchan[1] - vchan[0])


def first_moment(subcube, vchan):
    """[5] Intensity-weighted velocity centroid map [km/s]."""
    num = np.nansum(subcube * vchan[:, None, None], axis=0)
    den = np.nansum(subcube, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def fit_plane(vfield, dx_pc, dy_pc, mask):
    """[6] Least-squares plane v = v0 + Wx dx + Wy dy over masked, finite pixels.

    Returns Wx, Wy [km/s/pc], theta [deg], and the (dx,dy,v) used for the
    planarity test. dx/dy are physical offsets from the aperture centre [pc].
    """
    sel = mask & np.isfinite(vfield)
    if sel.sum() < 6:
        return None
    x, y, v = dx_pc[sel], dy_pc[sel], vfield[sel]
    A = np.column_stack([np.ones_like(x), x, y])
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    _, Wx, Wy = coef
    theta = np.degrees(np.arctan2(Wy, Wx))
    return Wx, Wy, theta, x, y, v


def planarity_pearson(Wx, Wy, x, y, v):
    """[7] Pearson r between distance along the gradient axis and v_LSR."""
    norm = np.hypot(Wx, Wy)
    if norm == 0 or len(v) < 3:
        return np.nan
    proj = (x * Wx + y * Wy) / norm            # perpendicular distance from rot. axis
    return pearsonr(proj, v)[0]


def specific_angular_momentum(R_pc, Omega_kms_pc, c=C_MOI):
    """[8] j = c R^2 Omega  ->  cm^2 / s."""
    j = c * (R_pc * u.pc) ** 2 * (Omega_kms_pc * u.km / u.s / u.pc)
    return j.to(u.cm**2 / u.s).value


# ----------------------------------------------------------------------------
# Cube / catalog access
#
# The per-GMC analysis is factored out of run() so that the plotting helpers
# below (and notebook work) can get at the intermediate products -- aperture
# spectrum, moment maps, velocity field, plane fit -- instead of only the
# summary row that ends up in the output table.
# ----------------------------------------------------------------------------
Cube = namedtuple("Cube", "hdu hdr wcs vax pc_per_pix ny nx")

GMCResult = namedtuple("GMCResult", [
    "name", "gmc", "R_pc", "xc", "yc", "extent",
    "dx_pc", "dy_pc", "ap", "spec", "vax", "vbar", "sig", "nsigma",
    "vmask", "vchan", "m0", "N_HI", "sigma_HI", "M_HI", "vfield",
    "Wx", "Wy", "Omega", "theta", "theta_pix", "theta_PA", "theta_disk",
    "frame",
    "rho", "j", "proj", "pv",
])


def open_cube(path=None):
    """Open the HI cube (memmap) and precompute the axes the analysis needs.

    The velocity axis is returned already trimmed of the first FGND_CHAN
    channels of Milky Way foreground, so it lines up with hdu.data[FGND_CHAN:].
    """
    hdu = fits.open(path or HI_CUBE, memmap=True)[0]
    hdr = hdu.header
    vax = ((np.arange(hdr["NAXIS3"]) - hdr["CRPIX3"]) * hdr["CDELT3"]
          + hdr["CRVAL3"]) / 1e3                          # km/s
    return Cube(hdu=hdu, hdr=hdr, wcs=WCS(hdr).celestial, vax=vax[FGND_CHAN:],
                pc_per_pix=(np.deg2rad(abs(hdr["CDELT2"])) * D_M33).to(u.pc).value,
                ny=hdr["NAXIS2"], nx=hdr["NAXIS1"])


def load_catalog(path=None, nmax=None):
    """Corbelli et al. (2017) Table 5 as a DataFrame."""
    cat = pd.read_csv(path or CATALOG, sep=r"\s+")
    return cat.iloc[:nmax] if nmax else cat


def _disk_offsets(dx_pc, dy_pc, hdr, incl=M33_INCL, pa=M33_PA_KIN):
    """Sky offsets in pc -> offsets in the disk plane (major, minor) in pc."""
    i, P = np.deg2rad(incl), np.deg2rad(pa)
    E = np.sign(hdr["CDELT1"]) * dx_pc
    N = np.sign(hdr["CDELT2"]) * dy_pc
    return (E * np.sin(P) + N * np.cos(P),
            (E * np.cos(P) - N * np.sin(P)) / np.cos(i))


def analyze_gmc(gmc, cube, snr=5.0, nsigma=2.0, frame="disk"):
    """Steps [1]-[8] for a single catalog row -> GMCResult, or None if rejected.

    nsigma sets the width of the kinematic association window. The paper uses
    v_bar +/- 2 sigma by default and tightens to 1 sigma for sightlines with
    multiple blended HI peaks -- in M33 that is worth checking per cloud.

    frame chooses the geometry of the accumulation cylinder:
      "sky"            circular aperture on the sky, plane fitted in projected
                       pc. This is the paper's method, ported unchanged.
      "disk_aperture"  aperture circular in the disk plane (an ellipse on the
                       sky), plane still fitted in projected pc.
      "disk"           (default) aperture circular in the disk plane and the
                       plane fitted in in-disk coordinates, so Omega is per
                       in-disk pc and theta is measured from the major axis.
    The cylinder is a column through the disk, so its cross-section is circular
    in the disk plane, not on the sky; for a face-on sample the distinction
    does not arise, which is why the ported method does not make it.

    "disk" is the default here, a documented departure from the paper. It is a
    self-consistent geometry at M33's i = 54 deg, and it is what the data
    prefer: over the 566-cloud sample it raises the planarity (median rho
    0.94 -> 0.96) and tightens the alignment between the HI gradients and the
    disk major axis (median |dtheta| 35 -> 26 deg, fraction within 45 deg
    58% -> 68%). It lowers Omega by 18%, because in-disk separations along the
    minor axis are longer than their projections; the sky-frame Omega is
    correspondingly overestimated. Set frame="sky" to reproduce the paper.

    Whatever the frame, the result carries theta in three conventions:
    `theta` (the raw fit angle, in the frame that was fitted), `theta_pix`
    (pixel frame, for drawing on the sky) and `theta_disk` (in the disk plane,
    from the major axis) -- so downstream code never has to guess.
    """
    name = f"GMC-{int(gmc['N'])}"
    R_pc = R_HI_cylinder(gmc["MH2"]).value                     # [1]
    r_pix = R_pc / cube.pc_per_pix

    # [2] aperture: circular mask around the GMC centre
    xc, yc = cube.wcs.wcs_world2pix(gmc["RAdeg"], gmc["DEdeg"], 0)
    xc, yc = float(xc), float(yc)
    x0, x1 = max(0, int(xc - r_pix - 1)), min(cube.nx, int(xc + r_pix + 2))
    y0, y1 = max(0, int(yc - r_pix - 1)), min(cube.ny, int(yc + r_pix + 2))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None

    sub = np.asarray(cube.hdu.data[FGND_CHAN:, y0:y1, x0:x1], dtype=float)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx_pc = (xx - xc) * cube.pc_per_pix
    dy_pc = (yy - yc) * cube.pc_per_pix
    d_maj, d_min = _disk_offsets(dx_pc, dy_pc, cube.hdr)
    if frame == "sky":
        ap = (dx_pc**2 + dy_pc**2) <= R_pc**2                  # cylinder footprint
    else:
        ap = (d_maj**2 + d_min**2) <= R_pc**2
    fx, fy = (d_maj, d_min) if frame == "disk" else (dx_pc, dy_pc)

    # [3] kinematic association from aperture-summed spectrum
    spec = np.nansum(sub * ap[None, :, :], axis=(1, 2))
    vbar, sig = associate_velocity(spec, cube.vax, snr_min=snr)
    if vbar is None:
        return None
    vmask = np.abs(cube.vax - vbar) <= nsigma * sig
    if vmask.sum() < 2:
        return None

    subw = sub[vmask]
    vchan = cube.vax[vmask]

    # [4] HI mass over the associated window (optically thin)
    m0 = moment0(subw, vchan)                                  # K km/s
    N_HI = NHI_COEFF * m0                                      # cm^-2
    mp_Msun = const.m_p.to(u.Msun).value
    pix_area = (cube.pc_per_pix * u.pc).to(u.cm).value ** 2
    M_HI = MU * mp_Msun * np.nansum(N_HI[ap] * pix_area)       # Msun
    sigma_HI = MU * mp_Msun * N_HI * u.pc.to(u.cm) ** 2        # Msun / pc^2

    # [5] velocity field
    vfield = first_moment(subw, vchan)

    # [6] plane fit -> Omega, theta
    fit = fit_plane(vfield, fx, fy, ap)
    if fit is None:
        return None
    Wx, Wy, theta, px, py, pv = fit
    Omega = np.hypot(Wx, Wy)                                   # km/s/pc

    # the same direction in the three frames downstream code asks for
    if frame == "disk":
        du, dw = Wx, Wy                                # fitted in-disk already
        Wx_pix, Wy_pix = project_direction(du, dw, cube.hdr)
    else:
        Wx_pix, Wy_pix = Wx, Wy
        du, dw = deproject_direction(Wx, Wy, cube.hdr)
    theta_pix = np.degrees(np.arctan2(Wy_pix, Wx_pix)) % 360.0
    theta_disk = np.degrees(np.arctan2(dw, du)) % 360.0
    theta_PA = float(sky_position_angle(theta_pix, cube.hdr))

    # [7] planarity, and the projection the goodness-of-fit plot is drawn against
    rho = planarity_pearson(Wx, Wy, px, py, pv)
    proj = (px * Wx + py * Wy) / Omega if Omega else np.zeros_like(pv)

    # [8] specific angular momentum
    j = specific_angular_momentum(R_pc, Omega)

    return GMCResult(
        name=name, gmc=gmc, R_pc=R_pc, xc=xc, yc=yc,
        extent=(dx_pc.min(), dx_pc.max(), dy_pc.min(), dy_pc.max()),
        dx_pc=dx_pc, dy_pc=dy_pc, ap=ap, spec=spec, vax=cube.vax,
        vbar=vbar, sig=sig, nsigma=nsigma, vmask=vmask, vchan=vchan,
        m0=m0, N_HI=N_HI, sigma_HI=sigma_HI, M_HI=M_HI, vfield=vfield,
        Wx=Wx, Wy=Wy, Omega=Omega, theta=theta, theta_pix=theta_pix,
        theta_PA=theta_PA, theta_disk=theta_disk, frame=frame, rho=rho, j=j,
        proj=proj, pv=pv,
    )


def result_row(res):
    """The Table 2 row for one GMCResult."""
    return {
        "Name": res.name,
        "M_HI_1e4": res.M_HI / 1e4,
        "R_HI_pc": res.R_pc,
        "Omega_HI_1e-2": res.Omega * 1e2,
        "theta_HI_deg": res.theta_pix,
        "theta_PA_deg": res.theta_PA,
        "theta_disk_deg": res.theta_disk,
        "rho_HI": res.rho,
        "j_HI_1e24": res.j / 1e24,
        "frame": res.frame,
    }


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run(nmax=None, snr=5.0, out="hi_table2_m33.csv", nsigma=2.0,
        figures=False, figdir=FIGDIR, frame="disk"):
    cat = load_catalog(nmax=nmax)
    cube = open_cube()

    rows = []
    for _, gmc in cat.iterrows():
        res = analyze_gmc(gmc, cube, snr=snr, nsigma=nsigma, frame=frame)
        if res is None:
            continue
        rows.append(result_row(res))

    tab = pd.DataFrame(rows)
    tab.to_csv(out, index=False)
    if figures:
        make_figures(tab, cube=cube, cat=cat, outdir=figdir)
    return tab


# ----------------------------------------------------------------------------
# Figures
#
# Reproductions of the figures of Mekuria & Imara (2026) for the M33 sample.
# Colour convention follows the paper: red = HI, blue/cyan = molecular.
# Figures 4 and 8 of the paper compare the molecular velocity gradient with the
# HI one; they are not implemented because this project has no CO cube (the
# Corbelli catalog gives sigma_C and R_e but no CO velocity field).
# ----------------------------------------------------------------------------
C_HI    = "#c2453f"          # HI          (paper: red)
C_H2    = "#2f6fb3"          # molecular   (paper: blue/cyan)
C_INK   = "#4d4d4d"          # annotation / model lines
C_FIT   = "#a4308f"          # combined least-squares fit (paper: magenta)
C_GRID  = "#b8b8b8"
CMAP_N  = "magma"            # sequential  -- column / surface density
CMAP_V  = "RdBu"             # diverging   -- velocity about the line centre.
                             # Note this is NOT the usual receding=red
                             # convention: here high v is blue.
# categorical, for several clouds overlaid in one panel. Ordered so adjacent
# entries stay separable under protanopia/deuteranopia; every panel also
# carries a legend, so identity is never colour alone.
C_SET = ["#c2453f", "#0072b2", "#e69f00", "#009e73",
         "#cc79a7", "#56b4e9", "#4d4d4d"]
# overlays drawn ON a colormap need hues absent from both magma and RdBu_r,
# so the categorical set above (tuned for a white background) is not reused
C_SET_MAP = ["#00e5ff", "#7cff00", "#ffffff", "#ff00d4",
             "#ffb000", "#000000", "#9a9a9a"]
VSPAN_KMS = 5.0              # fixed half-range for per-set velocity panels.
                             # Chosen from the data: the median envelope spans
                             # 2*Omega*R = 4.4 km/s, so +/-5 gives typical sets
                             # good contrast while clipping only ~13%.



def _style(ax, grid=True):
    if grid:
        ax.grid(alpha=0.25, lw=0.6, color=C_GRID)
        ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def _label_sky_axes(ax, hdr, tick_arcmin=10.0):
    """RA/Dec axis labels and ticks, with the frame taken from the header.

    Falls back to pixel labels if `ax` is an ordinary Axes rather than a
    WCSAxes -- which is what happens when a caller passes its own ax=.
    """
    if not hasattr(ax, "coords"):
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")
        return ax
    frame = str(hdr.get("RADESYS", hdr.get("RADECSYS", "ICRS"))).strip()
    equinox = hdr.get("EQUINOX")
    tag = f"J{equinox:.0f}" if frame.upper() == "FK5" and equinox else frame
    ra, dec = ax.coords[0], ax.coords[1]
    ra.set_axislabel(f"RA ({tag})")
    dec.set_axislabel(f"Dec ({tag})")
    ra.set_major_formatter("hh:mm:ss")
    dec.set_major_formatter("dd:mm")
    ra.set_ticks(spacing=tick_arcmin * u.arcmin)
    dec.set_ticks(spacing=tick_arcmin * u.arcmin)
    ax.coords.grid(color=C_GRID, alpha=0.25, lw=0.5)
    return ax


def _loglog_fit(x, y):
    """Least-squares fit in log10 space.

    Returns (slope, slope_err, intercept, mean_log_x). The last item is the
    pivot the slope uncertainty rotates about -- a band drawn about the
    intercept instead would fan out to meaninglessness away from R = 1 pc.
    """
    lx, ly = np.log10(x), np.log10(y)
    ok = np.isfinite(lx) & np.isfinite(ly)
    if ok.sum() < 3:
        return None
    coef, cov = np.polyfit(lx[ok], ly[ok], 1, cov=True)
    return coef[0], float(np.sqrt(cov[0, 0])), coef[1], float(lx[ok].mean())


def field_surface_density(cube, chunk=64, step=1):
    """Sigma_HI [Msun/pc^2] over the whole field, integrated across all channels.

    The cube is ~2 GB, so the sum is accumulated in blocks of `chunk` channels;
    `step` decimates channels for a quick look (step=4 is ~4x faster and good
    enough for an overview map).
    """
    return field_moments(cube, chunk=chunk, step=step)[0]


def field_moments(cube, chunk=64, step=1, sigma_min=5.0):
    """Sigma_HI [Msun/pc^2] and the velocity field [km/s] over the whole field.

    One pass over the cube, accumulated in blocks of `chunk` channels. The
    first moment is weighted by intensity clipped at zero -- negative noise
    channels otherwise throw the centroid around wildly in the outer disk --
    and is blanked below sigma_min, where there is no signal to take a centroid
    of. Sigma itself keeps the unclipped sum so it stays an honest integral.
    """
    dv = abs(cube.vax[1] - cube.vax[0]) * step
    acc = np.zeros((cube.ny, cube.nx), dtype=float)     # unclipped -> Sigma
    wacc = np.zeros_like(acc)                           # clipped   -> moment 1
    wvacc = np.zeros_like(acc)
    nchan = cube.hdu.data.shape[0]
    for lo in range(FGND_CHAN, nchan, chunk):
        block = np.asarray(cube.hdu.data[lo:lo + chunk:step], dtype=float)
        v = cube.vax[lo - FGND_CHAN:lo - FGND_CHAN + chunk:step][:len(block)]
        w = np.clip(block, 0, None)
        acc += np.nansum(block, axis=0)
        wacc += np.nansum(w, axis=0)
        wvacc += np.nansum(w * v[:, None, None], axis=0)

    sigma = (MU * const.m_p.to(u.Msun).value * NHI_COEFF * acc * dv
             * u.pc.to(u.cm) ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        vfield = wvacc / wacc
    vfield[~np.isfinite(vfield) | (sigma < sigma_min)] = np.nan
    return sigma, vfield


def plot_field_map(cube, cat, sigma=None, ax=None, levels=None, step=4,
                   smooth=11, circles=True, deproject=False, rmax_kpc=8.0,
                   dr_kpc=1.0, incl=M33_INCL, pa=M33_PA_KIN,
                   cbar=True, vmax=None, guides=False, orient=None):
    """[Fig. 1] Sigma_HI map of M33 with the HI envelope apertures overlaid.

    Pass a precomputed `sigma` from field_surface_density() to avoid re-reading
    the cube. Contour levels default to the paper's 5-40 Msun/pc^2 in steps of 5
    plus 50/75/100.

    With deproject=True the map is resampled into the disk plane, so lines of
    constant galactocentric radius become circles; concentric rings every
    dr_kpc and the phi = 0 direction are drawn as guides. A circular sky
    aperture deprojects to an ellipse stretched by 1/cos(i) along the minor
    axis, so the apertures are drawn as ellipses in that frame.
    """
    if sigma is None:
        sigma = field_surface_density(cube, step=step)
    if levels is None:
        levels = list(range(5, 45, 5)) + [50, 75, 100]

    if deproject:
        rot_deg = _PHI0_ANGLE[PHI0_ORIENT if orient is None else orient]
        img, extent = deproject_image(sigma, cube, rmax_kpc=rmax_kpc,
                                      incl=incl, pa=pa, orient=orient)
        if ax is None:
            ax = plt.figure(figsize=(7.6, 7.0)).add_subplot()
        im = ax.imshow(img, origin="lower", extent=extent, cmap=CMAP_N,
                       vmin=0, vmax=(np.nanpercentile(img, 99.5)
                                     if vmax is None else vmax))
        gx = np.linspace(extent[0], extent[1], img.shape[1])
        gy = np.linspace(extent[2], extent[3], img.shape[0])
        csig = (uniform_filter(np.nan_to_num(img), size=smooth)
                if smooth else np.nan_to_num(img))
        ax.contour(gx, gy, csig, levels=levels, colors="white",
                   linewidths=0.4, alpha=0.5)
        if circles:
            stretch = 1.0 / np.cos(np.deg2rad(incl))
            for _, gmc in cat.iterrows():
                R = R_HI_cylinder(gmc["MH2"]).value / 1000.0
                cu, cw = _rotate_uw(*deproject_points(
                    gmc["RAdeg"], gmc["DEdeg"], incl=incl, pa=pa),
                    orient=orient)
                # width lies along canonical u, so the patch carries the same
                # rotation as the positions
                ax.add_patch(Ellipse((float(cu), float(cw)), 2 * R,
                                     2 * R * stretch, angle=rot_deg,
                                     fill=False, ec="cyan",
                                     lw=0.5, alpha=0.85))
        _disk_guides(ax, rmax_kpc=rmax_kpc, dr=dr_kpc, orient=orient)
    else:
        if ax is None:
            # WCSAxes: sky coordinates come straight off the cube's celestial WCS,
            # so the ticks are real RA/Dec rather than pixel indices
            fig = plt.figure(figsize=(7.6, 7.0))
            ax = fig.add_subplot(projection=cube.wcs)
        im = ax.imshow(sigma, origin="lower", cmap=CMAP_N,
                       vmin=0, vmax=(np.nanpercentile(sigma, 99.5)
                                     if vmax is None else vmax))
        # contour a smoothed copy: at the native grid the 5 Msun/pc^2 level traces
        # noise speckle across the whole outer disk
        csig = uniform_filter(np.nan_to_num(sigma), size=smooth) if smooth else sigma
        ax.contour(csig, levels=levels, colors="white", linewidths=0.4, alpha=0.5)
        if circles:
            for _, gmc in cat.iterrows():
                R_pc = R_HI_cylinder(gmc["MH2"]).value
                xc, yc = cube.wcs.wcs_world2pix(gmc["RAdeg"], gmc["DEdeg"], 0)
                ax.add_patch(Circle((float(xc), float(yc)),
                                    R_pc / cube.pc_per_pix, fill=False,
                                    ec="cyan", lw=0.5, alpha=0.85))
        if guides:
            _sky_guides(ax, cube, rmax_kpc=rmax_kpc, dr=dr_kpc,
                        incl=incl, pa=pa)
        _label_sky_axes(ax, cube.hdr)

    if cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label(r"$\Sigma_{\rm HI}$  [$M_\odot$ pc$^{-2}$]")
    ax.set_title(f"M33 HI envelopes  (n={len(cat)})"
                 + ("  -- deprojected" if deproject else ""))
    return ax


def plot_envelope_spectrum(res, ax=None, show_fit=True, show_co=False):
    """[Fig. 2] Aperture spectrum of the HI envelope with the association window.

    Vertical lines mark v_bar +/- nsigma*sigma, the channels kept as
    kinematically associated. show_co overlays a Gaussian *model* of the CO line
    built from the catalog's sigma_C -- it is not a measured spectrum, and is off
    by default for that reason.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.6))

    ax.step(res.vax, res.spec, where="mid", color=C_HI, lw=1.6, label="HI")
    if show_fit:
        # associate_velocity returns only (mean, sigma); recover the amplitude by
        # projecting the spectrum onto the fitted profile shape
        shape = gaussian(res.vax, 1.0, res.vbar, res.sig)
        amp = float(np.dot(np.nan_to_num(res.spec), shape) / np.dot(shape, shape))
        ax.plot(res.vax, amp * shape,
                color=C_INK, lw=1.2, ls=":", label="Gaussian fit")
    if show_co:
        sc = float(res.gmc["sigmaC"])
        ax.plot(res.vax, gaussian(res.vax, np.nanmax(res.spec), res.vbar, sc),
                color=C_H2, lw=1.2, ls="--", label=r"CO model ($\sigma_C$)")

    for edge in (res.vbar - res.nsigma * res.sig, res.vbar + res.nsigma * res.sig):
        ax.axvline(edge, color=C_INK, lw=1.0, ls="-", alpha=0.7)
    ax.axhline(0, color=C_GRID, lw=0.6)

    ax.set_xlabel(r"$v_{\rm LSR}$ [km s$^{-1}$]")
    ax.set_ylabel("aperture-summed $T_B$ [K]")
    ax.set_title(f"{res.name}   "
                 rf"$\bar v$={res.vbar:.1f}, $\sigma$={res.sig:.1f} km s$^{{-1}}$")
    ax.legend(frameon=False, fontsize=8)
    return _style(ax)


def plot_cloud_maps(res, fig=None):
    """[Fig. 3] Sigma_HI map with the GMC marked, and the HI velocity field.

    Two panels rather than the paper's three: the middle panel there is the
    molecular velocity field, which needs a CO cube.
    """
    if fig is None:
        fig = plt.figure(figsize=(10.5, 4.4))
    ax1, ax2 = fig.subplots(1, 2)

    sig_map = np.where(res.ap, res.sigma_HI, np.nan)
    im1 = ax1.imshow(sig_map, origin="lower", extent=res.extent, cmap=CMAP_N)
    ax1.add_patch(Circle((0, 0), res.R_pc, fill=False, ec="k", lw=1.2, ls="--"))
    ax1.add_patch(Circle((0, 0), float(res.gmc["re"]), fill=False,
                         ec=C_H2, lw=1.6))
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
    cb1.set_label(r"$\Sigma_{\rm HI}$  [$M_\odot$ pc$^{-2}$]")
    ax1.set_title(rf"{res.name}   $M_{{\rm HI}}$ = {res.M_HI/1e4:.1f}$\times10^4\,M_\odot$")

    vf = np.where(res.ap, res.vfield, np.nan)
    span = np.nanmax(np.abs(vf - res.vbar)) if np.isfinite(vf).any() else 1.0
    im2 = ax2.imshow(vf, origin="lower", extent=res.extent, cmap=CMAP_V,
                     vmin=res.vbar - span, vmax=res.vbar + span)
    ax2.add_patch(Circle((0, 0), res.R_pc, fill=False, ec="k", lw=1.2, ls="--"))
    arrow = 0.6 * res.R_pc
    ax2.arrow(0, 0, arrow * res.Wx / res.Omega, arrow * res.Wy / res.Omega,
              color=C_HI, width=0.012 * res.R_pc, length_includes_head=True)
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    cb2.set_label(r"$\langle v_{\rm LSR}\rangle$ [km s$^{-1}$]")
    ax2.set_title(rf"$|\Omega|$ = {res.Omega*1e2:.2f}$\times10^{{-2}}$ km s$^{{-1}}$ pc$^{{-1}}$,"
                  rf"  $\theta$ = {res.theta:.0f}$^\circ$")

    for ax in (ax1, ax2):
        ax.set_xlabel(r"$\Delta x$ [pc]")
        ax.set_ylabel(r"$\Delta y$ [pc]")
    fig.tight_layout()
    return fig


def plot_velocity_profile(res, ax=None, nbins=10):
    """[Fig. 5] <v_LSR> vs perpendicular displacement from the rotation axis.

    Dashed line is the planar model; the shaded bands are the +/-1 and +/-2 sigma
    scatter of the velocity field within each radial bin, as in the paper.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 4.2))

    r, v = res.proj, res.pv
    ax.scatter(r, v, s=10, color=C_HI, alpha=0.45, lw=0, label="pixels")

    # least-squares plane passes through the centroid, so v0 follows from it
    v0 = np.mean(v) - res.Omega * np.mean(r)
    grid = np.linspace(r.min(), r.max(), 50)
    ax.plot(grid, v0 + res.Omega * grid, color=C_INK, lw=1.6, ls="--",
            label="planar model")

    edges = np.linspace(r.min(), r.max(), nbins + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, nbins - 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    mean = np.array([v[idx == b].mean() if (idx == b).any() else np.nan
                     for b in range(nbins)])
    sd = np.array([v[idx == b].std() if (idx == b).sum() > 1 else np.nan
                   for b in range(nbins)])
    for k, alpha in ((2, 0.12), (1, 0.22)):
        ax.fill_between(ctr, mean - k * sd, mean + k * sd,
                        color=C_HI, alpha=alpha, lw=0,
                        label=rf"$\pm{k}\sigma$ scatter")

    ax.annotate(rf"$\rho$ = {res.rho:.2f}", xy=(0.04, 0.93),
                xycoords="axes fraction", fontsize=11, color=C_INK)
    ax.set_xlabel(r"$r_\perp$ from rotation axis [pc]")
    ax.set_ylabel(r"$\langle v_{\rm LSR}\rangle$ [km s$^{-1}$]")
    ax.set_title(res.name)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _style(ax)


# ----------------------------------------------------------------------------
# Panel selection for the Figure 2 / Figure 5 grids
# ----------------------------------------------------------------------------
# The two grids share one set of six clouds, chosen so the layout itself is the
# argument: rows separate planar from non-planar envelopes, columns separate
# isolated from crowded ones. Reading down a column asks whether planarity
# survives at fixed crowding; reading across a row asks whether crowding is
# what breaks it.
OVERLAP_BANDS = {                # label: (lo, hi, target) on the covered fraction
    "isolated": (0.00, 0.02, 0.00),
    "moderate": (0.15, 0.50, 0.35),
    "crowded":  (0.75, 1.01, 0.90),
}


def envelope_overlap(tab, cat, cube, nsamp=400, seed=0):
    """Fraction of each envelope's aperture that other envelopes also cover.

    Measured by sampling points uniformly inside the aperture and asking how
    many fall inside at least one neighbour, which handles several overlapping
    neighbours without the double counting that summing pairwise lens areas
    would introduce. Returns a float per row of the merged table, in [0, 1].
    """
    merged = tab.merge(
        cat.assign(Name=[f"GMC-{int(n)}" for n in cat["N"]]), on="Name")
    x, y = cube.wcs.wcs_world2pix(merged["RAdeg"].to_numpy(),
                                  merged["DEdeg"].to_numpy(), 0)
    xy = np.column_stack([x, y]) * cube.pc_per_pix
    R = merged["R_HI_pc"].to_numpy()

    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, nsamp)
    rad = np.sqrt(rng.uniform(0, 1, nsamp))              # uniform over the disc
    tree = cKDTree(xy)

    frac = np.zeros(len(R))
    for i in range(len(R)):
        nb = [j for j in tree.query_ball_point(xy[i], R[i] + R.max())
              if j != i]
        if not nb:
            continue
        px = xy[i, 0] + R[i] * rad * np.cos(ang)
        py = xy[i, 1] + R[i] * rad * np.sin(ang)
        d = np.hypot(px[:, None] - xy[nb, 0], py[:, None] - xy[nb, 1])
        frac[i] = (d < R[nb]).any(axis=1).mean()
    return pd.Series(frac, index=merged.index), merged


def select_grid_clouds(tab, cat, cube, rho_hi=0.9, rho_lo=0.5, bands=None,
                       overlap=None):
    """Six clouds for the 2x3 grids: (rho band) x (crowding band), row-major.

    Row 1 is rho > rho_hi, row 2 is rho < rho_lo; the columns run isolated ->
    moderate -> crowded in the fraction of the aperture covered by neighbours.
    Within a cell the cloud whose overlap sits closest to the band's target is
    taken, with rho as the tiebreak (highest in the top row, lowest in the
    bottom). A cell with no cloud in its band falls back to the closest match
    in that rho row, so the grid always fills; the returned records carry the
    numbers so a caller can label what it actually got.

    Returns a list of six dicts with keys name, rho, overlap, row, col.
    """
    bands = OVERLAP_BANDS if bands is None else bands
    if overlap is None:
        overlap, merged = envelope_overlap(tab, cat, cube)
    else:
        merged = tab.merge(
            cat.assign(Name=[f"GMC-{int(n)}" for n in cat["N"]]), on="Name")
    m = merged.assign(f_overlap=np.asarray(overlap))

    picks = []
    for row, rows in enumerate(((m["rho_HI"] > rho_hi, False),
                                (m["rho_HI"] < rho_lo, True))):
        sel, low = rows
        for col, (label, (lo, hi, target)) in enumerate(bands.items()):
            band = m[sel & (m["f_overlap"] >= lo) & (m["f_overlap"] < hi)]
            pool = band if len(band) else m[sel]
            if not len(pool):
                continue
            key = (pool["f_overlap"] - target).abs()
            tie = pool["rho_HI"] if low else -pool["rho_HI"]
            best = pool.loc[(key + 1e-3 * tie).idxmin()]
            picks.append(dict(name=best["Name"], rho=float(best["rho_HI"]),
                              overlap=float(best["f_overlap"]),
                              row=row, col=col, band=label,
                              in_band=bool(len(band))))
    return picks


def _cloud_grid(results, draw, ncols=3, nrows=2, panel=(4.6, 3.6), fig=None,
                col_titles=None, row_titles=None, notes=None):
    """Lay one per-cloud panel out per cell, via `draw(res, ax)`.

    Shared by the Figure 2 and Figure 5 grids. Every panel carries the same
    legend entries, so the per-axes legends are dropped and one is put under
    the grid; axis labels survive on the outer panels only. Unused cells (5
    clouds in a 2x3 grid) are removed rather than left as empty frames.

    col_titles / row_titles caption the axes of the layout itself -- what the
    columns and rows were selected to vary -- above the top row and to the left
    of the first column. `notes` adds a per-panel line in the upper left, for
    the numbers that put each cloud in its cell.
    """
    if fig is None:
        fig = plt.figure(figsize=(panel[0] * ncols, panel[1] * nrows))
    axes = np.atleast_2d(fig.subplots(nrows, ncols, squeeze=False))

    for k, ax in enumerate(axes.ravel()):
        if k >= len(results):
            ax.remove()
            continue
        draw(results[k], ax)
        if ax.legend_ is not None:
            ax.legend_.remove()
        if k % ncols:                       # y label on the left column only
            ax.set_ylabel("")
        if k < len(results) - ncols:        # x label on the bottom row only
            ax.set_xlabel("")
        if notes is not None:
            ax.annotate(notes[k], xy=(0.04, 0.84), xycoords="axes fraction",
                        fontsize=9, color=C_INK)
        if col_titles is not None and k < ncols:
            ax.annotate(col_titles[k], xy=(0.5, 1.20), xycoords="axes fraction",
                        ha="center", va="bottom", fontsize=12, weight="bold")
        if row_titles is not None and k % ncols == 0:
            ax.annotate(row_titles[k // ncols], xy=(-0.26, 0.5),
                        xycoords="axes fraction", ha="center", va="center",
                        rotation=90, fontsize=12, weight="bold")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.tight_layout()
    empty = nrows * ncols - len(results)
    if empty:
        # park the legend in the cells the grid does not use, rather than
        # under the last row where it lands on an axis label
        x0 = (len(results) % ncols) / ncols
        fig.legend(handles, labels, frameon=False, fontsize=10, loc="center",
                   bbox_to_anchor=(x0 + 0.5 * empty / ncols, 0.5 / nrows))
    else:
        # anchored by its top edge at the figure bottom, so it clears the
        # x label of the centre panel instead of landing on it
        fig.legend(handles, labels, frameon=False, fontsize=9,
                   ncol=len(labels), loc="upper center",
                   bbox_to_anchor=(0.5, 0.0))
    return fig


def plot_envelope_spectrum_grid(results, fig=None, ncols=3, nrows=2,
                                col_titles=None, row_titles=None, notes=None,
                                **kw):
    """[Fig. 2] A grid of aperture spectra, one panel per cloud.

    Replaces the separate single-cloud spectra. Each panel keeps its own axes:
    the envelopes sit at different systemic velocities and differ by an order
    of magnitude in brightness, so a common scale would push most of them into
    a corner. Each title carries its own v_bar and sigma.
    """
    return _cloud_grid(results,
                       lambda res, ax: plot_envelope_spectrum(res, ax=ax, **kw),
                       ncols=ncols, nrows=nrows, panel=(5.0, 3.4), fig=fig,
                       col_titles=col_titles, row_titles=row_titles,
                       notes=notes)


def plot_velocity_profile_grid(results, fig=None, ncols=3, nrows=2, nbins=10,
                               col_titles=None, row_titles=None, notes=None):
    """[Fig. 5] A grid of velocity profiles, one panel per cloud.

    Replaces the separate single-cloud figures. Panels keep their own axis
    limits -- the velocity ranges differ by tens of km/s from cloud to cloud,
    so a shared scale would flatten every profile into a line -- and it is the
    shape of the run, not its absolute level, that is being compared.
    """
    return _cloud_grid(results,
                       lambda res, ax: plot_velocity_profile(res, ax=ax,
                                                             nbins=nbins),
                       ncols=ncols, nrows=nrows, fig=fig,
                       col_titles=col_titles, row_titles=row_titles,
                       notes=notes)



def summary_stats(tab, fig=None, rho_cut=0.75):
    """[Fig. 6] Pearson coefficients, then theta and j against rho.

    The third panel asks whether the clouds with poorly-fitting planes are also
    the ones with anomalous angular momentum -- if j were correlated with rho,
    the rho cut applied in Figure 9 would be biasing that fit.
    """
    if fig is None:
        fig = plt.figure(figsize=(6.0, 9.4))
    ax1, ax2, ax3 = fig.subplots(3, 1)

    rho = tab["rho_HI"].to_numpy()
    ax1.hist(rho, bins=np.linspace(0, 1, 21), color=C_HI, alpha=0.85,
             label="HI envelopes")
    med = np.nanmedian(rho)
    ax1.axvline(med, color=C_INK, ls="--", lw=1.4)
    ax1.annotate(rf"median $\rho$ = {med:.2f}", xy=(0.04, 0.88),
                 xycoords="axes fraction", fontsize=10, color=C_INK)
    frac = np.mean(rho >= rho_cut)
    ax1.annotate(rf"$\rho\geq{rho_cut}$: {frac*100:.0f}%", xy=(0.04, 0.78),
                 xycoords="axes fraction", fontsize=10, color=C_INK)
    ax1.set_xlabel(r"Pearson $\rho$")
    ax1.set_ylabel("N")
    ax1.legend(frameon=False, fontsize=9)

    ax2.scatter(rho, tab["theta_HI_deg"], s=16, color=C_HI, alpha=0.7, lw=0)
    ax2.axvline(rho_cut, color=C_INK, ls=":", lw=1.2)
    ax2.set_xlabel(r"Pearson $\rho$")
    ax2.set_ylabel(r"$\theta_{\rm HI}$ [deg]")

    j = tab["j_HI_1e24"].to_numpy() / 10.0            # -> 10^25 cm^2 / s
    ax3.scatter(rho, j, s=16, color=C_HI, alpha=0.7, lw=0)
    ax3.axvline(rho_cut, color=C_INK, ls=":", lw=1.2)
    ax3.set_yscale("log")
    ax3.set_xlabel(r"Pearson $\rho$")
    ax3.set_ylabel(r"$j_{\rm HI}$ [$10^{25}$ cm$^2$ s$^{-1}$]")

    _style(ax1); _style(ax2); _style(ax3)
    fig.tight_layout()
    return fig


def plot_angular_momentum(tab, fig=None, nbins_j=60, nbins_dtheta=36,
                          hdr=None, pa_gal=M33_PA_KIN):
    """[Fig. 7] Magnitude of j, and the HI gradient direction relative to M33.

    Two panels: the distribution of j, and
    delta_theta = PA_HI - pa_gal over (-180, 180], in the style of the paper's
    Figure 4, where a separation under 45 deg counts as co-rotation.

    theta_HI_deg is converted to a sky position angle first (see
    sky_position_angle); pass the cube header as `hdr` if its CDELT signs differ
    from the usual orientation. pa_gal defaults to M33's kinematic major axis.
    """
    if fig is None:
        fig = plt.figure(figsize=(10.0, 4.2))
    ax1, ax2 = fig.subplots(1, 2)

    j = tab["j_HI_1e24"].to_numpy() / 10.0            # -> 10^25 cm^2 / s
    ax1.hist(j, bins=np.logspace(np.log10(max(j.min(), 1e-4)),
                                 np.log10(j.max()), nbins_j + 1),
             color=C_HI, alpha=0.85)
    ax1.set_xscale("log")
    ax1.axvline(np.nanmedian(j), color=C_INK, ls="--", lw=1.4)
    ax1.annotate(rf"median = {np.nanmedian(j):.2f}", xy=(0.04, 0.9),
                 xycoords="axes fraction", fontsize=10, color=C_INK)
    ax1.set_xlabel(r"$j_{\rm HI}$ [$10^{25}$ cm$^2$ s$^{-1}$]")
    ax1.set_ylabel("N")
    _style(ax1)

    # theta_disk_deg is already referenced to the major axis, so the offset
    # from galactic rotation is just that angle wrapped; fall back to the sky
    # PA for tables written before the column existed.
    if "theta_disk_deg" in tab:
        dth = _wrap180(tab["theta_disk_deg"].to_numpy())
        pa_lab = "disk major axis"
    else:
        dth = _wrap180(sky_position_angle(tab["theta_HI_deg"].to_numpy(), hdr)
                       - pa_gal)
        pa_lab = rf"{pa_gal:.1f}$^\circ$"

    ax2.hist(dth, bins=np.linspace(-180, 180, nbins_dtheta + 1),
             color=C_HI, alpha=0.85)
    for edge in (-45, 45):
        ax2.axvline(edge, color=C_INK, ls=":", lw=1.4)
    ax2.annotate(rf"$|\Delta\theta|<45^\circ$: {np.mean(np.abs(dth) < 45)*100:.0f}%"
                 "\n" rf"median $|\Delta\theta|$ = {np.median(np.abs(dth)):.0f}$^\circ$",
                 xy=(0.02, 0.86), xycoords="axes fraction", fontsize=10,
                 color=C_INK)
    ax2.set_xlim(-180, 180)
    ax2.set_xticks(np.arange(-180, 181, 45))
    ax2.set_xlabel(rf"$\theta_{{\rm HI}}-\theta_{{\rm Gal}}$ [deg]   "
                   rf"($\theta_{{\rm Gal}}$ = {pa_lab})")
    ax2.set_ylabel("N")
    _style(ax2)

    fig.tight_layout()
    return fig


def _size_relation(ax, R, y, keep, ylabel, sym, rho_cut=0.75, legend=True):
    """One log-log panel of `y` against R_HI, with the fit and its two bands.

    Shared by the Omega and j panels of Fig. 9 so both are fitted and drawn
    identically; `sym` is the symbol used in the slope annotation.
    """
    ax.scatter(R[keep], y[keep], s=22, color=C_HI, lw=0, alpha=0.8, zorder=2,
               label=rf"$\rho>{rho_cut}$  (fitted)")
    ax.scatter(R[~keep], y[~keep], s=22, facecolors="none", edgecolors=C_HI,
               lw=1.0, alpha=0.8, zorder=2,
               label=rf"$\rho\leq{rho_cut}$  (excluded)")

    fit = _loglog_fit(R[keep], y[keep])
    if fit:
        slope, err, icept, pivot = fit
        grid = np.logspace(np.log10(R.min()), np.log10(R.max()), 50)
        lg = np.log10(grid)
        model = icept + slope * lg

        # two different things, drawn differently on purpose: the wide red band
        # is where an individual cloud falls (rms residual), the narrow black
        # band is how well the mean relation itself is pinned down. With ~500
        # clouds the second is ~8x thinner than the first.
        lx, ly = np.log10(R[keep]), np.log10(y[keep])
        rms = float((ly - (icept + slope * lx)).std(ddof=2))
        ax.fill_between(grid, 10 ** (model - rms), 10 ** (model + rms),
                        color=C_HI, alpha=0.4, lw=0, zorder=1,
                        label=rf"$\pm1\sigma$ scatter ({rms:.2f} dex)")

        band = err * np.abs(lg - pivot)
        ax.fill_between(grid, 10 ** (model - band), 10 ** (model + band),
                        color="black", alpha=1.0, lw=0, zorder=3,
                        label=r"$\pm1\sigma$ fit uncertainty")
        ax.plot(grid, 10 ** model, color=C_GRID, ls="--", lw=0.7, zorder=4)

        ax.annotate(rf"${sym} \propto R^{{{slope:.2f}\pm{err:.2f}}}$",
                    xy=(0.04, 0.9), xycoords="axes fraction",
                    fontsize=11, color=C_INK)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$R_{\rm HI}$ [pc]")
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    return _style(ax)


def plot_j_vs_size(tab, ax=None, rho_cut=0.75):
    """[Fig. 9] Omega_HI and j_HI against R_HI, log-log, with fits and bands.

    Following the paper, systems with rho <= rho_cut are drawn as open markers
    and are excluded from the fits.

    Called with ax=None this returns a two-panel figure: Omega on the left, j on
    the right. Since j = 0.4 R^2 Omega by construction, the left slope should be
    the right one minus 2 -- the pair is a consistency check on the fit as much
    as a result. Passing a single `ax` draws the j panel alone, which is what
    the other figures that embed this relation want.
    """
    R = tab["R_HI_pc"].to_numpy()
    j = tab["j_HI_1e24"].to_numpy() / 10.0            # -> 10^25 cm^2 / s
    Om = tab["Omega_HI_1e-2"].to_numpy()              # 10^-2 km/s/pc
    keep = tab["rho_HI"].to_numpy() > rho_cut

    j_lab = r"$j_{\rm HI}$ [$10^{25}$ cm$^2$ s$^{-1}$]"
    om_lab = r"$\Omega_{\rm HI}$ [$10^{-2}$ km s$^{-1}$ pc$^{-1}$]"

    if ax is not None:
        return _size_relation(ax, R, j, keep, j_lab, "j", rho_cut)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.6))
    _size_relation(ax0, R, Om, keep, om_lab, r"\Omega", rho_cut, legend=False)
    _size_relation(ax1, R, j, keep, j_lab, "j", rho_cut)
    fig.tight_layout()
    return fig


def plot_size_angular_momentum(tab, ax=None, extra=None, rho_cut=0.75):
    """[Fig. 10] Size-angular momentum relation with literature data.

    `extra` is {label: (R_pc, j_1e24)} for comparison samples -- e.g. the Solar
    Neighbourhood clouds of Mekuria & Imara. Those are drawn filled; the M33
    points are drawn open, following the paper's convention that extragalactic
    measurements are unfilled. The magenta line fits everything together.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 5.0))

    R = tab["R_HI_pc"].to_numpy()
    j = tab["j_HI_1e24"].to_numpy()
    keep = tab["rho_HI"].to_numpy() > rho_cut
    ax.scatter(R[keep], j[keep], s=20, facecolors="none", edgecolors=C_HI,
               lw=1.0, label="M33 HI envelopes")

    allR, allj = [R[keep]], [j[keep]]
    for i, (label, (xr, xj)) in enumerate(sorted((extra or {}).items())):
        ax.scatter(xr, xj, s=20, color=C_H2 if i == 0 else C_INK,
                   lw=0, alpha=0.8, label=label)
        allR.append(np.asarray(xr, float)); allj.append(np.asarray(xj, float))

    allR, allj = np.concatenate(allR), np.concatenate(allj)
    fit = _loglog_fit(allR, allj)
    if fit:
        slope, err, icept, _ = fit
        grid = np.logspace(np.log10(allR.min()), np.log10(allR.max()), 50)
        ax.plot(grid, 10 ** icept * grid ** slope, color=C_FIT, lw=1.8,
                label=rf"$j\propto R^{{{slope:.2f}\pm{err:.2f}}}$")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$R$ [pc]")
    ax.set_ylabel(r"$j$ [$10^{24}$ cm$^2$ s$^{-1}$]")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    return _style(ax)


def plot_position_angles(tab, cat, cube, vfield=None, sigma=None, ax=None,
                         arrow_pc=400.0, scale_by=None, step=4, zoom=0.08,
                         arrow_color="black", vsys=None, levels=None,
                         smooth=11, deproject=False, rmax_kpc=8.0,
                         dr_kpc=1.0, incl=M33_INCL, pa=M33_PA_KIN,
                         cbar=True, span=None, guides=False, orient=None):
    """[Fig. 11] Gradient directions as arrows over M33's own velocity field.

    Replaces the paper's Galactic-latitude ordering, which has no M33
    analogue, with a sharper version of the same diagnostic: an arrow that
    simply follows the local colour of the background is tracking galactic
    rotation, not cloud-scale rotation.

    Arrows are fixed length (arrow_pc) unless scale_by is "Omega" or "j", in
    which case length runs linearly with that column, up to arrow_pc. `zoom`
    crops to the clouds' bounding box plus that fractional margin -- over the
    full field the 566 arrows collapse into an unreadable smear. Pass zoom=None
    to keep the whole map. vsys defaults to the median of the field, which
    centres the diverging colour scale on M33's systemic velocity. Sigma_HI is
    overlaid as contours at the same levels as Figure 1.

    With deproject=True everything is mapped into the disk plane: the velocity
    field, the contours, the arrow positions and the arrow directions, which
    are transformed by the same rotation and stretch as positions rather than
    simply carried over.
    """
    if vfield is None or sigma is None:
        _sigma, _vfield = field_moments(cube, step=step)
        sigma = _sigma if sigma is None else sigma
        vfield = _vfield if vfield is None else vfield
    if levels is None:
        levels = list(range(5, 45, 5)) + [50, 75, 100]

    merged = tab.merge(
        cat.assign(Name=[f"GMC-{int(n)}" for n in cat["N"]]), on="Name")
    th = np.deg2rad(merged["theta_HI_deg"].to_numpy())   # pixel frame

    if deproject:
        vmap, extent = deproject_image(vfield, cube, rmax_kpc=rmax_kpc,
                                       incl=incl, pa=pa, orient=orient)
        smap, _ = deproject_image(sigma, cube, rmax_kpc=rmax_kpc,
                                  incl=incl, pa=pa, orient=orient)
        if ax is None:
            ax = plt.figure(figsize=(7.6, 7.0)).add_subplot()
        xs, ys = _rotate_uw(*deproject_points(merged["RAdeg"].to_numpy(),
                                              merged["DEdeg"].to_numpy(),
                                              incl=incl, pa=pa),
                            orient=orient)
        dx, dy = _rotate_uw(*deproject_direction(np.cos(th), np.sin(th),
                                                 cube.hdr, incl=incl, pa=pa),
                            orient=orient)
        scale = arrow_pc / 1000.0
    else:
        vmap, smap, extent = vfield, sigma, None
        if ax is None:
            ax = plt.figure(figsize=(7.6, 7.0)).add_subplot(
                projection=cube.wcs)
        xs, ys = cube.wcs.wcs_world2pix(merged["RAdeg"].to_numpy(),
                                        merged["DEdeg"].to_numpy(), 0)
        dx, dy = np.cos(th), np.sin(th)
        scale = arrow_pc / cube.pc_per_pix

    if vsys is None:
        vsys = float(np.nanmedian(vmap))
    if span is None:
        span = np.nanpercentile(np.abs(vmap - vsys), 99)
    im = ax.imshow(vmap, origin="lower", cmap=CMAP_V, extent=extent,
                   vmin=vsys - span, vmax=vsys + span)
    if cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label(r"$\langle v_{\rm LSR}\rangle$ [km s$^{-1}$]")

    csig = uniform_filter(np.nan_to_num(smap), size=smooth) if smooth else smap
    if extent is None:
        ax.contour(csig, levels=levels, colors=C_INK, linewidths=0.4,
                   alpha=0.55)
    else:
        ax.contour(np.linspace(extent[0], extent[1], csig.shape[1]),
                   np.linspace(extent[2], extent[3], csig.shape[0]),
                   csig, levels=levels, colors=C_INK, linewidths=0.4,
                   alpha=0.55)

    length = np.full(len(merged), scale)
    if scale_by:
        col = {"Omega": "Omega_HI_1e-2", "j": "j_HI_1e24"}[scale_by]
        w = merged[col].to_numpy()
        length *= w / np.nanmax(w)

    ax.quiver(xs, ys, length * dx, length * dy, angles="xy",
              scale_units="xy", scale=1.0, color=arrow_color, width=0.0030,
              headwidth=4, headlength=5, alpha=0.9)

    if deproject:
        _disk_guides(ax, rmax_kpc=rmax_kpc, dr=dr_kpc, colour="black",
                     orient=orient)
        ax.set_xlim(-rmax_kpc, rmax_kpc)
        ax.set_ylim(-rmax_kpc, rmax_kpc)
    else:
        if guides:
            _sky_guides(ax, cube, rmax_kpc=rmax_kpc, dr=dr_kpc, colour="black",
                        incl=incl, pa=pa, crop=zoom is None)
        if zoom is not None:
            pad = zoom * max(np.ptp(xs), np.ptp(ys))
            ax.set_xlim(xs.min() - pad, xs.max() + pad)
            ax.set_ylim(ys.min() - pad, ys.max() + pad)
        _label_sky_axes(ax, cube.hdr)

    ax.set_title(rf"M33: $\theta_{{\rm HI}}$ over the HI velocity field  "
                 rf"(n={len(merged)})"
                 + ("  -- deprojected" if deproject else ""))
    return ax


def sky_position_angle(theta_deg, hdr=None):
    """Pixel-frame gradient angle -> sky position angle, degrees East of North.

    fit_plane returns atan2(Wy, Wx) measured from the +x pixel axis, which is
    not a position angle: with CDELT1 < 0 the +x axis points West. East and
    North in pixel space are sign(CDELT1)*x and sign(CDELT2)*y, so the mapping
    follows from the header. Without a header the usual orientation
    (CDELT1 < 0, CDELT2 > 0) is assumed, which for this cube is PA = theta - 90.
    """
    s1 = np.sign(hdr["CDELT1"]) if hdr is not None else -1.0
    s2 = np.sign(hdr["CDELT2"]) if hdr is not None else 1.0
    th = np.deg2rad(np.asarray(theta_deg, dtype=float))
    pa = np.degrees(np.arctan2(s1 * np.cos(th), s2 * np.sin(th)))
    return np.round(pa, 10) % 360   # round first: -1e-15 would wrap to 360.0


def _wrap180(d):
    """Wrap an angle difference onto (-180, 180]."""
    return (np.asarray(d, dtype=float) + 180.0) % 360.0 - 180.0


OVERLAP_CRITERIA = {
    "touch":   lambda d, Ri, Rj: d < Ri + Rj,
    "half":    lambda d, Ri, Rj: d < 0.5 * (Ri + Rj),
    "min":     lambda d, Ri, Rj: d < np.minimum(Ri, Rj),
    "quarter": lambda d, Ri, Rj: d < 0.25 * (Ri + Rj),
}


def cluster_envelopes(tab, cat, cube, criterion="min"):
    """Group overlapping HI envelopes into maximal cliques.

    Returns a list of tuples of cloud names, shortest first. Every cloud
    appears at least once; clouds with no qualifying overlap come back as
    singletons. Unlike connected components these sets can share members -- a
    cloud that overlaps two neighbours which do not overlap each other belongs
    to both sets, which single-linkage grouping cannot express and which
    otherwise percolates the whole inner disk into one blob.

    criterion picks how strict the overlap test is; see OVERLAP_CRITERIA.
    """
    merged = tab.merge(
        cat.assign(Name=[f"GMC-{int(n)}" for n in cat["N"]]), on="Name")
    x, y = cube.wcs.wcs_world2pix(merged["RAdeg"].to_numpy(),
                                  merged["DEdeg"].to_numpy(), 0)
    xy = np.column_stack([x, y]) * cube.pc_per_pix
    R = merged["R_HI_pc"].to_numpy()
    names = merged["Name"].tolist()
    n = len(names)

    pairs = np.array(sorted(cKDTree(xy).query_pairs(r=2 * R.max())))
    d = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
    keep = OVERLAP_CRITERIA[criterion](d, R[pairs[:, 0]], R[pairs[:, 1]])

    adj = {i: set() for i in range(n)}
    for i, j in pairs[keep]:
        adj[i].add(j)
        adj[j].add(i)

    cliques = []

    def expand(clique, cand, done):                    # Bron-Kerbosch, pivoted
        if not cand and not done:
            cliques.append(clique)
            return
        pivot = max(cand | done, key=lambda v: len(adj[v]))
        for v in list(cand - adj[pivot]):
            expand(clique | {v}, cand & adj[v], done & adj[v])
            cand.remove(v)
            done.add(v)

    expand(set(), set(range(n)), set())
    out = [tuple(sorted((names[i] for i in c),
                        key=lambda s: int(s.split("-")[1]))) for c in cliques]
    return sorted(out, key=lambda t: (len(t), int(t[0].split("-")[1])))


def _draw_set_panel(ax, results, per_pixel=True, normalize=False):
    """One panel: the spectra of a set overlaid, each with its Gaussian fit.

    per_pixel divides the aperture-summed spectrum by the number of pixels
    summed. Without it the amplitude carries the aperture area rather than the
    gas: R_HI is derived from M_H2, so two heavily overlapping envelopes of
    different mass sum over areas differing as R^2 and their profiles differ by
    that factor even though the mean brightness agrees to ~10%. Dividing gives
    the mean T_B in the aperture, which is comparable between clouds.
    """
    lo, hi = [], []
    for k, res in enumerate(results):
        colour = C_SET[k % len(C_SET)]
        spec = res.spec / res.ap.sum() if per_pixel else res.spec
        shape = gaussian(res.vax, 1.0, res.vbar, res.sig)
        amp = float(np.dot(np.nan_to_num(spec), shape) / np.dot(shape, shape))
        if normalize and amp:
            spec, amp = spec / amp, 1.0
        ax.step(res.vax, spec, where="mid", color=colour, lw=1.1, alpha=0.85,
                label=rf"{res.name}  ($\sigma$={res.sig:.0f})")
        ax.plot(res.vax, amp * shape, color=colour, lw=1.0, ls=":")
        for edge in (-1, 1):
            ax.axvline(res.vbar + edge * res.nsigma * res.sig,
                       color=colour, lw=0.6, alpha=0.35)
        lo.append(res.vbar - 4 * res.sig)
        hi.append(res.vbar + 4 * res.sig)

    ax.axhline(0, color=C_GRID, lw=0.5)
    ax.set_xlim(min(lo), max(hi))
    ax.legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.2)
    ax.tick_params(labelsize=7)
    return _style(ax)


def plot_spectrum_sets(tab, cat, cube, outdir=FIGDIR, criterion="min",
                       nrows=5, ncols=4, snr=5.0, per_pixel=True,
                       normalize=False, prefix="spectra_sets"):
    """Appendix-style spectrum pages, one panel per overlap set.

    Sets of one cloud get a panel each; sets of N clouds get all N spectra
    overlaid in one panel, each with its own Gaussian fit and association
    window. One multi-page PDF per set size. Returns the paths written.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    sets = cluster_envelopes(tab, cat, cube, criterion=criterion)

    rows = {f"GMC-{int(r['N'])}": r for _, r in cat.iterrows()}
    cache = {}

    def result(name):
        if name not in cache:
            cache[name] = analyze_gmc(rows[name], cube, snr=snr)
        return cache[name]

    written = []
    for size in sorted({len(t) for t in sets}):
        group = [t for t in sets if len(t) == size]
        path = out / f"{prefix}_{criterion}_size{size}.pdf"
        per_page = nrows * ncols
        npages = int(np.ceil(len(group) / per_page))
        with PdfPages(path) as pdf:
            for start in range(0, len(group), per_page):
                page = group[start:start + per_page]
                fig, axes = plt.subplots(nrows, ncols,
                                         figsize=(3.2 * ncols, 2.4 * nrows))
                for ax, names in zip(np.ravel(axes), page):
                    res = [r for r in (result(n) for n in names) if r is not None]
                    if res:
                        _draw_set_panel(ax, res, per_pixel=per_pixel,
                                        normalize=normalize)
                    else:
                        ax.set_axis_off()
                for ax in np.ravel(axes)[len(page):]:
                    ax.set_axis_off()
                fig.suptitle(f"HI envelope spectra -- sets of {size} "
                             f"({criterion}); page {start // per_page + 1}"
                             f" of {npages}", fontsize=10)
                fig.supxlabel(r"$v_{\rm LSR}$ [km s$^{-1}$]", fontsize=9)
                fig.supylabel("mean $T_B$ in aperture [K]" if per_pixel
                              else "aperture-summed $T_B$ [K]", fontsize=9)
                fig.tight_layout(rect=(0.01, 0.01, 1, 0.97))
                pdf.savefig(fig, dpi=150)
                plt.close(fig)
        written.append(path)
    return written


# ----------------------------------------------------------------------------
# Deprojection into M33's disk plane
#
# The sky map is an inclined view of a disk. Resampling it onto the disk plane
# turns lines of constant galactocentric radius into circles, which is what
# makes a radial/azimuthal reading of the map possible at all.
# ----------------------------------------------------------------------------
def _disk_basis(incl=M33_INCL, pa=M33_PA_KIN):
    return np.deg2rad(incl), np.deg2rad(pa)


# Where phi = 0 -- the receding major axis -- is made to point in a deprojected
# panel. This is presentation only: deproject_points/_direction keep returning
# canonical (u, w) with u along the major axis, so phi, galactocentric_coords
# and everything measured off them are untouched. The rotation is applied by
# the plotting layer alone, to the image grid, the aperture positions and the
# arrow directions alike, so a panel stays internally consistent.
PHI0_ORIENT = "down"
_PHI0_ANGLE = {"right": 0.0, "up": 90.0, "left": 180.0, "down": -90.0}
_PHI0_LABELS = {                         # (x label, y label)
    "right": (r"$u$ along major axis [kpc]", r"$w$ along minor axis [kpc]"),
    "left":  (r"$-u$ along major axis [kpc]", r"$-w$ along minor axis [kpc]"),
    "up":    (r"$-w$ along minor axis [kpc]", r"$u$ along major axis [kpc]"),
    "down":  (r"$w$ along minor axis [kpc]", r"$-u$ along major axis [kpc]"),
}


def _rotate_uw(du, dw, orient=None, inverse=False):
    """Canonical in-disk (u, w) -> the displayed frame (or back)."""
    a = np.deg2rad(_PHI0_ANGLE[PHI0_ORIENT if orient is None else orient])
    if inverse:
        a = -a
    c, sn = np.cos(a), np.sin(a)
    du, dw = np.asarray(du, float), np.asarray(dw, float)
    return c * du - sn * dw, sn * du + c * dw


def deproject_points(ra_deg, dec_deg, incl=M33_INCL, pa=M33_PA_KIN):
    """Sky positions -> in-disk (u, w) in kpc, u along the major axis.

    u is unforeshortened; w is the minor-axis offset stretched by 1/cos(i).
    R = hypot(u, w) and phi = atan2(w, u) then match galactocentric_coords()
    in the i -> 0 limit.
    """
    i, P = _disk_basis(incl, pa)
    dra = (np.asarray(ra_deg, float) - M33_CENTER[0]) * np.cos(
        np.deg2rad(M33_CENTER[1]))
    ddec = np.asarray(dec_deg, float) - M33_CENTER[1]
    D = D_M33.to(u.kpc).value
    E = np.deg2rad(dra) * D                              # kpc, +ve East
    N = np.deg2rad(ddec) * D                             # kpc, +ve North
    x_maj = E * np.sin(P) + N * np.cos(P)
    x_min = E * np.cos(P) - N * np.sin(P)
    return x_maj, x_min / np.cos(i)


def deproject_direction(Wx, Wy, hdr, incl=M33_INCL, pa=M33_PA_KIN):
    """A pixel-frame direction -> its direction in the disk plane (unit vector).

    Deprojection is an affine map, so a direction transforms by the same
    rotation and minor-axis stretch as a position -- but it is not simply the
    deprojected position of the arrow tip, which is why this is separate.
    """
    i, P = _disk_basis(incl, pa)
    E = np.sign(hdr["CDELT1"]) * np.asarray(Wx, float)
    N = np.sign(hdr["CDELT2"]) * np.asarray(Wy, float)
    du = E * np.sin(P) + N * np.cos(P)
    dw = (E * np.cos(P) - N * np.sin(P)) / np.cos(i)
    norm = np.hypot(du, dw)
    with np.errstate(invalid="ignore", divide="ignore"):
        return du / norm, dw / norm


def project_direction(du, dw, hdr, incl=M33_INCL, pa=M33_PA_KIN):
    """An in-disk direction -> its direction in the pixel frame (unit vector).

    The inverse of deproject_direction: undo the minor-axis stretch, rotate
    back onto (East, North), then onto the pixel axes via the CDELT signs.
    Needed because a fit done in in-disk coordinates still has to be drawn on
    the sky.
    """
    i, P = _disk_basis(incl, pa)
    du = np.asarray(du, float)
    dw = np.asarray(dw, float) * np.cos(i)
    E = du * np.sin(P) + dw * np.cos(P)
    N = du * np.cos(P) - dw * np.sin(P)
    Wx = np.sign(hdr["CDELT1"]) * E
    Wy = np.sign(hdr["CDELT2"]) * N
    norm = np.hypot(Wx, Wy)
    with np.errstate(invalid="ignore", divide="ignore"):
        return Wx / norm, Wy / norm


def deproject_image(image, cube, rmax_kpc=8.0, npix=601, incl=M33_INCL,
                    pa=M33_PA_KIN, orient=None):
    """Resample a sky map onto the disk plane -> (face-on image, extent kpc).

    Blank pixels are carried through by sampling a validity mask alongside the
    data, so NaNs do not bleed outwards through the interpolation.
    """
    i, P = _disk_basis(incl, pa)
    xc, yc = cube.wcs.wcs_world2pix(M33_CENTER[0], M33_CENTER[1], 0)
    kpc_pix = cube.pc_per_pix / 1000.0

    g = np.linspace(-rmax_kpc, rmax_kpc, npix)
    # the grid is laid out in the *displayed* frame, then rotated back into
    # canonical (u, w) before being squashed onto the sky
    uu, ww = _rotate_uw(*np.meshgrid(g, g), orient=orient, inverse=True)
    x_maj, x_min = uu, ww * np.cos(i)                    # squash back onto sky
    E = x_maj * np.sin(P) + x_min * np.cos(P)
    N = x_maj * np.cos(P) - x_min * np.sin(P)
    px = float(xc) + np.sign(cube.hdr["CDELT1"]) * E / kpc_pix
    py = float(yc) + np.sign(cube.hdr["CDELT2"]) * N / kpc_pix

    good = np.isfinite(image)
    num = map_coordinates(np.where(good, image, 0.0), [py, px], order=1,
                          mode="constant", cval=0.0)
    den = map_coordinates(good.astype(float), [py, px], order=1,
                          mode="constant", cval=0.0)
    out = np.where(den > 0.5, num / np.where(den > 0, den, 1.0), np.nan)
    return out, (-rmax_kpc, rmax_kpc, -rmax_kpc, rmax_kpc)


def _disk_guides(ax, rmax_kpc=8.0, dr=1.0, colour="white", ray=True,
                 orient=None):
    """Concentric circles every dr kpc, plus the phi = 0 direction."""
    stroke = "black" if colour in ("white", "cyan") else "white"
    for R in np.arange(dr, rmax_kpc + 0.5 * dr, dr):
        ax.add_patch(Circle((0, 0), R, fill=False, ec=colour, lw=0.5,
                            alpha=0.55, zorder=4))
        ax.annotate(f"{R:.0f}", xy=(R * 0.7071, R * 0.7071), fontsize=5,
                    color=colour, alpha=0.8, zorder=4,
                    ha="center", va="center").set_path_effects(
                        [path_effects.withStroke(linewidth=1.2,
                                                 foreground=stroke)])
    if ray:
        ex, ey = _rotate_uw(rmax_kpc, 0.0, orient=orient)
        a = ax.annotate(r"$\phi=0$", xy=(float(ex) * 0.98, float(ey) * 0.98),
                        fontsize=7, color=colour, zorder=5,
                        ha="right" if ex > 0 else ("left" if ex < 0 else "center"),
                        va="top" if ey > 0 else "bottom")   # keep it inside
        a.set_path_effects([path_effects.withStroke(linewidth=1.4,
                                                    foreground=stroke)])
        ax.plot([0, float(ex)], [0, float(ey)], color=colour, lw=1.1, ls="--",
                alpha=0.9, zorder=4)
    ax.set_aspect("equal")
    xlab, ylab = _PHI0_LABELS[PHI0_ORIENT if orient is None else orient]
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    return ax


def _sky_ring(R_kpc, cube, incl=M33_INCL, pa=M33_PA_KIN, n=361):
    """One galactocentric circle of radius R_kpc, in sky pixel coordinates.

    The inverse of deproject_image's mapping: squash by cos i along the minor
    axis, rotate by the position angle, scale to pixels. A circle in the disk
    plane comes back as an ellipse on the sky, which is the point of drawing it.
    """
    i, P = _disk_basis(incl, pa)
    xc, yc = cube.wcs.wcs_world2pix(M33_CENTER[0], M33_CENTER[1], 0)
    kpc_pix = cube.pc_per_pix / 1000.0
    phi = np.linspace(0, 2 * np.pi, n)
    x_maj, x_min = R_kpc * np.cos(phi), R_kpc * np.sin(phi) * np.cos(i)
    E = x_maj * np.sin(P) + x_min * np.cos(P)
    N = x_maj * np.cos(P) - x_min * np.sin(P)
    return (float(xc) + np.sign(cube.hdr["CDELT1"]) * E / kpc_pix,
            float(yc) + np.sign(cube.hdr["CDELT2"]) * N / kpc_pix)


def _sky_guides(ax, cube, rmax_kpc=8.0, dr=1.0, colour="white",
                incl=M33_INCL, pa=M33_PA_KIN, ray=True, crop=True):
    """_disk_guides' counterpart for the sky frame: the same rings as ellipses.

    Drawn on the observed panel of a sky/deprojected pair so the reader can
    match ring for ring across the two. `crop` limits the axes to the outer
    ring's bounding box, which is the sky footprint of the deprojected panel.
    """
    stroke = "black" if colour in ("white", "cyan") else "white"
    for R in np.arange(dr, rmax_kpc + 0.5 * dr, dr):
        px, py = _sky_ring(R, cube, incl=incl, pa=pa)
        ax.plot(px, py, color=colour, lw=0.5, alpha=0.55, zorder=4)
    if ray:
        # phi = 0 is the receding major axis: the disk-plane +u direction
        px, py = _sky_ring(rmax_kpc, cube, incl=incl, pa=pa, n=361)
        cx, cy = cube.wcs.wcs_world2pix(M33_CENTER[0], M33_CENTER[1], 0)
        ax.plot([float(cx), px[0]], [float(cy), py[0]], color=colour, lw=1.1,
                ls="--", alpha=0.9, zorder=4)
        a = ax.annotate(r"$\phi=0$", xy=(px[0], py[0]), fontsize=7,
                        color=colour, ha="center", va="bottom", zorder=5)
        a.set_path_effects([path_effects.withStroke(linewidth=1.4,
                                                    foreground=stroke)])
    if crop:
        px, py = _sky_ring(rmax_kpc, cube, incl=incl, pa=pa)
        mx, my = 0.04 * np.ptp(px), 0.04 * np.ptp(py)
        ax.set_xlim(px.min() - mx, px.max() + mx)
        ax.set_ylim(py.min() - my, py.max() + my)
    return ax


def galactocentric_coords(ra_deg, dec_deg, pa_ref=M33_PA_KIN):
    """(R_phi) position of a cloud in M33's frame -> phi in degrees.

    phi is the sky position angle of the cloud about the M33 centre, measured
    East of North and then referenced to `pa_ref`, so phi = 0 lies along the
    receding kinematic major axis and phi grows towards the East. This is a
    sky-plane azimuth: deprojecting it into the disk would need the
    inclination, which this project does not carry.

    The galactocentric radius is taken from the catalogue's own R column
    rather than recomputed here.
    """
    dra = (np.asarray(ra_deg, float) - M33_CENTER[0]) * np.cos(
        np.deg2rad(M33_CENTER[1]))                       # +ve towards East
    ddec = np.asarray(dec_deg, float) - M33_CENTER[1]    # +ve towards North
    pa = np.degrees(np.arctan2(dra, ddec)) % 360
    return (pa - pa_ref) % 360


def _set_cutout(results, cube, margin_pc=40.0, sigma_min=5.0):
    """Cutout holding every envelope of a set entire, plus its moment maps.

    The box spans all the apertures (not just their centres) so that for sets
    of two or more nothing is clipped, and the velocity window is the union of
    the members' association windows.
    """
    xs = np.array([r.xc for r in results])
    ys = np.array([r.yc for r in results])
    rs = np.array([r.R_pc for r in results]) / cube.pc_per_pix
    m = margin_pc / cube.pc_per_pix
    x0 = int(max(0, np.floor((xs - rs).min() - m)))
    x1 = int(min(cube.nx, np.ceil((xs + rs).max() + m)))
    y0 = int(max(0, np.floor((ys - rs).min() - m)))
    y1 = int(min(cube.ny, np.ceil((ys + rs).max() + m)))

    vmask = np.zeros(len(cube.vax), dtype=bool)
    for r in results:
        vmask |= np.abs(cube.vax - r.vbar) <= r.nsigma * r.sig

    sub = np.asarray(cube.hdu.data[FGND_CHAN:, y0:y1, x0:x1], dtype=float)[vmask]
    vchan = cube.vax[vmask]
    m0 = moment0(sub, vchan)
    sigma = (MU * const.m_p.to(u.Msun).value * NHI_COEFF * m0
             * u.pc.to(u.cm) ** 2)
    v1 = first_moment(sub, vchan)
    v1 = np.where(sigma >= sigma_min, v1, np.nan)
    return (x0, x1, y0, y1), sigma, v1


def _draw_set_maps(ax0, ax1, results, cube, box, sigma, v1, vspan=VSPAN_KMS):
    """Sigma_HI and velocity panels for one set, with the apertures marked.

    Solid circle = HI envelope (R_HI), dashed = the GMC itself (catalogue R_e),
    one colour per cloud. Ranges are annotated in the corners rather than given
    colourbars, which would not fit a grid of panels.

    vspan fixes the velocity half-range in km/s so that colour means the same
    thing on every page: scaling each panel to its own extremes made a 1 km/s
    noise gradient and a 20 km/s real one look identical. Pass vspan=None for
    the per-panel version, which is then set by the 95th percentile rather than
    by a single outlier pixel. Panels that saturate say so in the annotation.
    """
    x0, x1, y0, y1 = box
    pc = cube.pc_per_pix
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    extent = ((x0 - cx) * pc, (x1 - cx) * pc, (y0 - cy) * pc, (y1 - cy) * pc)

    # union of the apertures, so the annotated ranges describe the envelopes
    # rather than the cutout box and its margin
    yy, xx = np.mgrid[y0:y1, x0:x1]
    ap = np.zeros(sigma.shape, dtype=bool)
    for r in results:
        ap |= ((xx - r.xc) ** 2 + (yy - r.yc) ** 2) <= (r.R_pc / pc) ** 2

    # filled contours on pixel-centre coordinates, so the maps line up with the
    # circles rather than sitting half a pixel off as an imshow extent would
    gx = (np.arange(x0, x1) - cx) * pc
    gy = (np.arange(y0, y1) - cy) * pc

    def _filled(ax, z, **kw):
        """contourf, with the antialiasing seams between bands suppressed.

        Adjacent filled polygons otherwise leave hairline gaps in vector
        output, which print as white striping across the map.
        """
        cs = ax.contourf(gx, gy, z, extend="both", **kw)
        cs.set_edgecolor("face")
        return cs

    s_ap = sigma[ap]
    _filled(ax0, sigma, cmap=CMAP_N,
            levels=np.linspace(np.nanpercentile(s_ap, 1),
                               np.nanpercentile(s_ap, 99), 21))

    v_ap = v1[ap]
    saturated = False
    if np.isfinite(v_ap).any():
        vmid = float(np.nanmedian(v_ap))
        if vspan is None:
            half = float(np.nanpercentile(np.abs(v_ap - vmid), 95))
        else:
            half = float(vspan)
            saturated = np.nanmax(np.abs(v_ap - vmid)) > half
        _filled(ax1, v1, cmap=CMAP_V,
                levels=np.linspace(vmid - half, vmid + half, 21))
    for ax in (ax0, ax1):
        ax.set_aspect("equal")
        ax.set_xlim(gx[0], gx[-1])
        ax.set_ylim(gy[0], gy[-1])

    for k, r in enumerate(results):
        colour = C_SET_MAP[k % len(C_SET_MAP)]
        dx, dy = (r.xc - cx) * pc, (r.yc - cy) * pc
        for ax in (ax0, ax1):
            ax.add_patch(Circle((dx, dy), r.R_pc, fill=False, ec=colour,
                                lw=1.3))
            ax.add_patch(Circle((dx, dy), float(r.gmc["re"]), fill=False,
                                ec=colour, lw=1.1, ls="--"))
        # fitted gradient direction, on the velocity panel only: the arrow is
        # what fit_plane derived, so it can be checked against the colours it
        # was derived from
        if r.Omega:
            arrow = 0.6 * r.R_pc
            a = ax1.arrow(dx, dy, arrow * r.Wx / r.Omega, arrow * r.Wy / r.Omega,
                          color=colour, width=0.022 * r.R_pc,
                          length_includes_head=True, zorder=5)
            a.set_path_effects([path_effects.withStroke(
                linewidth=1.2, foreground="black")])

    def label(ax, text, y, colour="white"):
        t = ax.annotate(text, xy=(0.03, y), xycoords="axes fraction",
                        fontsize=6, color=colour)
        t.set_path_effects([path_effects.withStroke(
            linewidth=1.6, foreground="black")])

    for k, r in enumerate(results):
        phi = float(galactocentric_coords(r.gmc["RAdeg"], r.gmc["DEdeg"]))
        label(ax0, rf"{r.name}  $R$={float(r.gmc['R']):.1f} kpc  "
                   rf"$\phi$={phi:.0f}$^\circ$",
              0.94 - 0.07 * k, colour=C_SET_MAP[k % len(C_SET_MAP)])
    label(ax0, rf"$\Sigma$ {np.nanmin(s_ap):.0f} to {np.nanmax(s_ap):.0f}", 0.04)
    if np.isfinite(v_ap).any():
        note = " (clipped)" if saturated else ""
        label(ax1, rf"$v$ {np.nanmin(v_ap):.0f} to {np.nanmax(v_ap):.0f}{note}",
              0.04)
    for ax in (ax0, ax1):
        ax.tick_params(labelsize=6)
    return ax0, ax1


def plot_moment_map_sets(tab, cat, cube, outdir=FIGDIR, criterion="min",
                         nrows=4, sets_per_row=2, snr=5.0, margin_pc=40.0,
                         prefix="maps_sets", vspan=VSPAN_KMS):
    """Appendix-style moment-map pages, one Sigma_HI/velocity pair per set.

    Companion to plot_spectrum_sets: same sets, same page structure, but each
    set gets two panels -- integrated intensity and velocity field -- over a
    cutout that contains all of its envelopes. One multi-page PDF per set size.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    sets = cluster_envelopes(tab, cat, cube, criterion=criterion)

    rows = {f"GMC-{int(r['N'])}": r for _, r in cat.iterrows()}
    cache = {}

    def result(name):
        if name not in cache:
            cache[name] = analyze_gmc(rows[name], cube, snr=snr)
        return cache[name]

    written = []
    for size in sorted({len(t) for t in sets}):
        group = [t for t in sets if len(t) == size]
        path = out / f"{prefix}_{criterion}_size{size}.pdf"
        per_page = nrows * sets_per_row
        npages = int(np.ceil(len(group) / per_page))
        with PdfPages(path) as pdf:
            for start in range(0, len(group), per_page):
                page = group[start:start + per_page]
                fig, axes = plt.subplots(
                    nrows, 2 * sets_per_row,
                    figsize=(2.9 * 2 * sets_per_row, 2.9 * nrows))
                axes = np.atleast_2d(axes)
                slots = [(axes[i, 2 * s], axes[i, 2 * s + 1])
                         for i in range(nrows) for s in range(sets_per_row)]
                for (ax0, ax1), names in zip(slots, page):
                    res = [r for r in (result(n) for n in names) if r is not None]
                    if not res:
                        ax0.set_axis_off(); ax1.set_axis_off()
                        continue
                    box, sigma, v1 = _set_cutout(res, cube, margin_pc=margin_pc)
                    _draw_set_maps(ax0, ax1, res, cube, box, sigma, v1,
                                   vspan=vspan)
                for ax0, ax1 in slots[len(page):]:
                    ax0.set_axis_off(); ax1.set_axis_off()
                scale = (f"velocity scale fixed at +/-{vspan:.0f} km/s"
                         if vspan is not None else "velocity scale per panel")
                fig.suptitle(f"HI envelopes -- moment 0 (left) and moment 1 "
                             f"(right); sets of {size} ({criterion}); "
                             f"{scale}; page {start // per_page + 1} of {npages}",
                             fontsize=10)
                fig.supxlabel(r"$\Delta x$ [pc]", fontsize=9)
                fig.supylabel(r"$\Delta y$ [pc]", fontsize=9)
                fig.tight_layout(rect=(0.01, 0.01, 1, 0.97))
                pdf.savefig(fig, dpi=150)
                plt.close(fig)
        written.append(path)
    return written


# ----------------------------------------------------------------------------
# Sky / disk-plane pairs -- the two-panel forms of Figures 1 and 11
# ----------------------------------------------------------------------------
def _pair_axes(cube, figsize=(13.4, 6.2)):
    """A figure with a WCSAxes on the left and a plain kpc axes on the right.

    The projection has to be fixed when the axes is created, which is why the
    pair figures build their own axes and hand them down rather than letting
    each plotter make its own.
    """
    fig = plt.figure(figsize=figsize)
    ax_sky = fig.add_subplot(1, 2, 1, projection=cube.wcs)
    ax_dep = fig.add_subplot(1, 2, 2)
    return fig, ax_sky, ax_dep


def plot_field_map_pair(cube, cat, sigma=None, step=4, rmax_kpc=8.0,
                        dr_kpc=1.0, incl=M33_INCL, pa=M33_PA_KIN,
                        orient=None, **kw):
    """[Fig. 1] Sigma_HI as observed (left) and in the disk plane (right).

    Both panels share one stretch, taken from the sky map, so the colour of a
    feature means the same thing on either side; only the right-hand panel
    carries the colorbar. The rings are the same galactocentric radii in both,
    ellipses on the sky and circles face-on.
    """
    if sigma is None:
        sigma = field_surface_density(cube, step=step)
    vmax = float(np.nanpercentile(sigma, 99.5))
    fig, ax_sky, ax_dep = _pair_axes(cube)

    plot_field_map(cube, cat, sigma=sigma, ax=ax_sky, deproject=False,
                   rmax_kpc=rmax_kpc, dr_kpc=dr_kpc, incl=incl, pa=pa,
                   cbar=False, vmax=vmax, guides=True, orient=orient, **kw)
    plot_field_map(cube, cat, sigma=sigma, ax=ax_dep, deproject=True,
                   rmax_kpc=rmax_kpc, dr_kpc=dr_kpc, incl=incl, pa=pa,
                   cbar=True, vmax=vmax, orient=orient, **kw)

    ax_sky.set_title("Observed -- on the sky")
    ax_dep.set_title(rf"Deprojected -- face-on  "
                     rf"(PA {pa:.1f}$^\circ$, $i$ {incl:.0f}$^\circ$, "
                     rf"stretch $1/\cos i$ = {1/np.cos(np.deg2rad(incl)):.2f})")
    fig.suptitle(f"M33 HI envelopes  (n={len(cat)})", y=0.97)
    fig.tight_layout()
    return fig


def plot_position_angles_pair(tab, cat, cube, vfield=None, sigma=None, step=4,
                              rmax_kpc=8.0, dr_kpc=1.0, incl=M33_INCL,
                              pa=M33_PA_KIN, orient=None, **kw):
    """[Fig. 11] theta_HI over the velocity field, observed (left) vs face-on.

    As with the Figure 1 pair, one velocity stretch is shared across both
    panels -- computed from the sky map and passed to the disk-plane one -- so
    a colour is the same v_LSR on either side. The sky panel is cropped to the
    footprint of the outer ring rather than to the clouds' bounding box, so the
    two panels cover the same piece of the galaxy.
    """
    if vfield is None or sigma is None:
        _sigma, _vfield = field_moments(cube, step=step)
        sigma = _sigma if sigma is None else sigma
        vfield = _vfield if vfield is None else vfield
    vsys = float(np.nanmedian(vfield))
    span = float(np.nanpercentile(np.abs(vfield - vsys), 99))
    fig, ax_sky, ax_dep = _pair_axes(cube)

    plot_position_angles(tab, cat, cube, vfield=vfield, sigma=sigma,
                         ax=ax_sky, deproject=False, zoom=None, guides=True,
                         rmax_kpc=rmax_kpc, dr_kpc=dr_kpc, incl=incl, pa=pa,
                         cbar=False, vsys=vsys, span=span, orient=orient, **kw)
    plot_position_angles(tab, cat, cube, vfield=vfield, sigma=sigma,
                         ax=ax_dep, deproject=True,
                         rmax_kpc=rmax_kpc, dr_kpc=dr_kpc, incl=incl, pa=pa,
                         cbar=True, vsys=vsys, span=span, orient=orient, **kw)

    ax_sky.set_title("Observed -- on the sky")
    ax_dep.set_title(rf"Deprojected -- face-on  "
                     rf"(PA {pa:.1f}$^\circ$, $i$ {incl:.0f}$^\circ$)")
    n = len(tab.merge(cat.assign(Name=[f"GMC-{int(v)}" for v in cat["N"]]),
                      on="Name"))
    fig.suptitle(rf"M33: $\theta_{{\rm HI}}$ over the HI velocity field  "
                 rf"(n={n})", y=0.97)
    fig.tight_layout()
    return fig


def _figure_of(obj):
    """The plot helpers return an Axes or a Figure; normalise to a Figure."""
    return obj if isinstance(obj, plt.Figure) else obj.figure


def make_figures(tab, cube=None, cat=None, outdir=FIGDIR, examples=3,
                 fmt="pdf", dpi=200, sigma=None,
                 rho_hi=0.9, rho_lo=0.5):
    """Render the whole figure set to `outdir`; returns the paths written.

    The population figures need only `tab`. The field map and the per-cloud
    figures need the cube, so they are skipped when `cube` is None -- that makes
    the function usable straight off a saved table without touching the 2 GB
    file. `examples` is either a count (that many of the most planar clouds) or
    an explicit list of names like ["GMC-1", "GMC-8"].
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    if cat is None:
        cat = load_catalog()
    written = []

    def save(obj, stem):
        fig = _figure_of(obj)
        path = out / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
        return path

################## NEXT STEPS: sort ^ based on amount/complexity of information needed to make them then call functions in that order vv


    save(plot_pearson_summary(tab), "fig06_pearson")
    save(plot_angular_momentum(tab, hdr=cube.hdr if cube else None),
         "fig07_angular_momentum")
    save(plot_j_vs_size(tab), "fig09_j_vs_size")
    save(plot_size_angular_momentum(tab), "fig10_size_angular_momentum")

    if cube is None:
        return written

    # one pass over the cube gives both maps
    if sigma is None:
        sigma, vfield = field_moments(cube, step=4)
    else:
        vfield = field_moments(cube, step=4)[1]
    save(plot_field_map_pair(cube, cat, sigma=sigma), "fig01_field_map")
    save(plot_position_angles_pair(tab, cat, cube, vfield=vfield, sigma=sigma),
         "fig11_position_angles")

    ranked = tab.sort_values("rho_HI", ascending=False)["Name"]
    names = list(ranked.head(examples)) if isinstance(examples, int) \
        else list(examples)
    picks = select_grid_clouds(tab, cat, cube, rho_hi=rho_hi, rho_lo=rho_lo)
    by_name = {f"GMC-{int(n)}": row for n, row in
               zip(cat["N"], (r for _, r in cat.iterrows()))}

    cache = {}

    def analysed(name):
        if name not in cache:
            gmc = by_name.get(name)
            cache[name] = analyze_gmc(gmc, cube) if gmc is not None else None
        return cache[name]

    for name in names:
        res = analysed(name)
        if res is None:
            continue
        tag = name.replace("GMC-", "").zfill(3)
        save(plot_cloud_maps(res), f"fig03_maps_{tag}")

    grid, kept = [], []
    for pick in picks:
        res = analysed(pick["name"])
        if res is not None:
            grid.append(res)
            kept.append(pick)
    if grid:
        cols = [f"{lab}  ($f$ {lo:.2f}-{min(hi, 1.0):.2f})"
                for lab, (lo, hi, _) in OVERLAP_BANDS.items()]
        rows = [rf"$\rho$ > {rho_hi}", rf"$\rho$ < {rho_lo}"]
        notes = [rf"$f$ = {p['overlap']:.2f}" for p in kept]
        save(plot_envelope_spectrum_grid(grid, col_titles=cols,
                                         row_titles=rows, notes=notes),
             "fig02_spectra")
        save(plot_velocity_profile_grid(grid, col_titles=cols,
                                        row_titles=rows, notes=notes),
             "fig05_profiles")

    return written


def _fmt(tab):
    disp = tab.copy()
    disp.columns = ["Name", "M_HI", "R_HI", "Omega_HI", "theta_HI", "rho_HI", "j_HI"]
    return disp.to_string(
        index=False,
        float_format=lambda v: f"{v:8.3g}",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=None,
                    help="only process the first N GMCs (testing)")
    ap.add_argument("--snr", type=float, default=5.0,
                    help="min peak/noise for a GMC to be kept")
    ap.add_argument("--out", default="hi_table2_m33.csv")
    ap.add_argument("--figures", action="store_true",
                    help="also render the figure set")
    ap.add_argument("--figdir", default=FIGDIR,
                    help=f"where --figures writes (default {FIGDIR})")
    args = ap.parse_args()

    table = run(nmax=args.nmax, snr=args.snr, out=args.out,
                figures=args.figures, figdir=args.figdir)
    print(f"\nTable 2 (HI-only) -- M33 GMCs  [n={len(table)}]")
    print("  M_HI [1e4 Msun]  R_HI [pc]  Omega_HI [1e-2 km/s/pc]  "
          "theta [deg]  rho  j_HI [1e24 cm^2/s]\n")
    print(_fmt(table.head(25)))
    print(f"\nsaved -> {args.out}")
