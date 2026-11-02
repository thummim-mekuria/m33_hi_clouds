"""The rosolowsky2003_figs.py figure set, re-run on the full-resolution IBB
cube instead of the C+D+tp 120 pc mid-res cube, with the HI envelope redefined
as a fixed box+velocity cutout instead of a mass-scaled circular aperture.

Same catalogue (Rosolowsky+2003 BIMA CO, 45 GMCs), same 11 figures -- three
things differ from rosolowsky2003_figs.py:

  * HI cube:
        ../data/m33_C+D+tp_himidres_120pc.fits          (~1620x1350x241, 2.06 km/s/chan)
    ->  ../data/M33_A+B+C+D_HI_0p4kms_feathered_K_reimaged.fits
                                                         (7874x6635x572, 0.41 km/s/chan)
    the full-res feathered cube used in 03_replicating_ibb.ipynb / Imara,
    Bigiel & Blitz (2011).

  * FGND_CHAN.  The pipeline's default (60 channels) is tuned to the mid-res
    cube, whose velocity axis runs +70 -> -424 km/s and so needs its first 60
    channels (+70 -> -51 km/s) dropped as Milky Way foreground. This cube's
    velocity axis runs only -74 -> -310 km/s -- entirely within the M33 disk,
    nowhere near 0 km/s -- so it carries no MW foreground to trim. Trimming 60
    channels here would instead cut real disk emission from the approaching
    edge. Set to 0 for this run via a module-level override on
    mekuria_imara_pipeline (open_cube() and analyze_gmc_box() both read the
    constant directly, there is no parameter for it).

  * field_moments' block size for the fig1 background map. This cube's planes
    are ~24x larger (7874x6635 vs 1620x1350) than the mid-res cube's, and the
    machine here has 24 GB RAM, so the default chunk=64 (read as float64) would
    pull multi-GB blocks through memory at once. Pass a smaller chunk to keep
    that bounded.

  * HI envelope definition. mekuria_imara_pipeline.analyze_gmc() sizes each
    envelope from the mass-dependent accumulation-cylinder radius
    R_HI_cylinder(M_H2) and finds its velocity window by Gaussian-fitting the
    aperture spectrum. analyze_gmc_box() below replaces both: every cloud gets
    the same fixed BOX_PC x BOX_PC sky-plane square (not disk-deprojected --
    a square has no clean disk-plane equivalent) and the same fixed VWIN_KMS
    velocity window, centred on the catalogue's own tabulated VLSR rather than
    a fit to this cube. Consequences worth knowing before reading the output:
        - R_HI_pc is now a constant (BOX_PC/2) for every cloud, so fig7, fig8
          and fig11 -- which plot quantities against R_HI -- show the HI side
          as a single vertical column rather than a size distribution, and
          fig8's HI power-law fit (j vs R_HI) is degenerate (no x range).
        - j_HI uses that same constant R_HI_pc in j = c R^2 Omega, so it is a
          rescaled Omega_HI, not an independent measurement.
        - there is no more per-cloud kinematic-association rejection (no
          Gaussian fit to fail); a cloud is only dropped if its cutout is too
          small at the map edge or its plane fit is under-determined.

Output goes to ../paper/figures_ibb2011_comparisons/. Run:

    /Users/thummimm/miniconda3/bin/python3 ibb2011_comparison_figs.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # headless; must precede pyplot / mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u
from matplotlib.patches import Rectangle

import mekuria_imara_pipeline as mp
import rosolowsky2003_catalog as rc

FIGDIR = Path("../paper/figures_ibb2011_comparisons")
FULLRES_CUBE = Path("home/tmekuria/data/M33_A+B+C+D_HI_0p4kms_feathered_K_reimaged.fits")
MIDRES_CUBE = Path("../data/m33_C+D+tp_himidres_120pc.fits")   # this pipeline's
                                                 # original cube; ~14.6 pc/pixel,
                                                 # so the 140 pc box is ~10 px
                                                 # across here vs ~45 on FULLRES_CUBE
TAB2 = Path("../data/rosolowsky2003_tab2.csv")   # rosolowsky2003_catalog's own
                                                 # default path is a stale name
MI_TAB2 = Path("../data/mekuria_imara_tab2.csv")  # Mekuria & Imara 2026, Table 2
IBB_TAB1 = Path("../data/imara2011_tab1.csv")     # Imara, Bigiel & Blitz 2011,
                                                 # ApJ 732, 79, Table 1
OUT_CSV = "hi_table2_m33_rosolowsky2003_ibbcube.csv"
OUT_CSV_MIDRES = "hi_table2_m33_rosolowsky2003_midrescube_boxcut.csv"

# Rosolowsky+2003 assumed D = 840 kpc; the pipeline uses 859.  Match the
# rescaling rosolowsky2003_catalog applies to its sizes and masses so the two
# sides of every ratio are on one distance scale (a 2.3% effect either way).
DFAC = rc.D_PIPE / rc.D_PUB

# 1 pc km/s expressed in 1e24 cm^2/s
PC_KMS_TO_1E24 = (1 * u.pc).to(u.cm).value * (1 * u.km).to(u.cm).value / 1e24

# HI envelope definition for analyze_gmc_box(): a fixed box_pc x box_pc sky-plane
# square and a fixed vwin_kms velocity window, both centred on the catalogue's
# own GMC position and VLSR -- see the module docstring.
BOX_PC = 140.0
VWIN_KMS = 50.0


# ---------------------------------------------------------------------------
# data assembly
# ---------------------------------------------------------------------------
def load_gmc_side(cat):
    """Rosolowsky+2003 Table 2 -> the CO-side quantities, indexed by cloud N.

    Omega_GMC   [km/s/pc]     from gradv (tabulated in 1e-2 km/s/pc)
    Phi_GMC     [deg]         position angle of the CO velocity gradient, taken
                              as-tabulated and assumed degrees E of N like
                              theta_PA_deg; a constant convention offset would
                              just shift the fig6 Phi median
    j_GMC_1e24  [1e24 cm^2/s] from j1 (pc km/s); NaN for the 16 unresolved clouds
    M_GMC_1e4   [1e4 Msun]    from M_CO
    R_GMC_pc    [pc]          resolved cloud radius; NaN if unresolved
    R_kpc       [kpc]         galactocentric radius, from the loader
    """
    t = pd.read_csv(TAB2, comment="#")
    # .to_numpy() drops the Series index: passing indexed Series into the
    # constructor alongside index= makes pandas reindex them onto Cloud, which
    # silently shifts every column by one row.
    d = pd.DataFrame(
        {
            "Omega_GMC": t["gradv_1e-2"].to_numpy() * 1e-2 / DFAC,   # km/s/pc ~ 1/D
            "Phi_GMC": t["phi_deg"].to_numpy(dtype=float),
            "j_GMC_1e24": t["j1"].to_numpy() * PC_KMS_TO_1E24 * DFAC,  # length-like ~ D
            "M_GMC_1e4": t["MCO_1e4"].to_numpy() * DFAC**2,          # CO mass ~ D^2
            "R_GMC_pc": t["r_pc"].to_numpy() * DFAC,                 # size ~ D
        },
        index=t["Cloud"].to_numpy().astype(int),
    )
    d["R_kpc"] = cat.set_index("N")["R"]        # aligns on index label
    return d


def load_mi_tab2():
    """Mekuria & Imara (2026) Table 2 -> per-cloud quantities, indexed by N.

    theta_GMC is recovered as theta_HI - (theta_HI - theta_GMC); M_HI is
    converted from the table's 1e5 Msun to 1e4 Msun to match everything else.
    """
    t = pd.read_csv(MI_TAB2, comment="#")
    return pd.DataFrame(
        {
            "Omega_GMC_1e-2": t["Omega_GMC_1e-2"].to_numpy(),
            "Omega_HI_1e-2": t["Omega_HI_1e-2"].to_numpy(),
            "theta_HI_deg": t["theta_HI_deg"].to_numpy(dtype=float),
            "theta_GMC_deg": (t["theta_HI_deg"] - t["dtheta_deg"]).to_numpy(float),
            "M_HI_1e4": t["M_HI_1e5"].to_numpy() * 10.0,
            "M_GMC_1e4": t["M_GMC_1e4"].to_numpy(),
        },
        index=t["N"].to_numpy().astype(int),
    )


def load_ibb_tab1():
    """Imara, Bigiel & Blitz (2011) Table 1 -> per-cloud quantities, indexed by N.

    extent_pc is the length over which IBB fitted each HI linear gradient (NaN
    for the 6 clouds without a linear HI gradient); it is IBB's own size measure
    for the HI side, the counterpart of this pipeline's R_HI.
    theta_HI_deg is IBB's own HI gradient position angle, degrees East of North
    -- the same convention as this pipeline's theta_PA_deg (not theta_HI_deg,
    which in this pipeline's own tables is the pixel-frame angle).
    """
    t = pd.read_csv(IBB_TAB1, comment="#")
    return pd.DataFrame(
        {
            "Omega_GMC_1e-2": t["Omega_GMC_1e-2"].to_numpy(dtype=float),
            "Omega_HI_1e-2": t["Omega_HI_1e-2"].to_numpy(dtype=float),
            "theta_HI_deg": t["theta_HI_deg"].to_numpy(dtype=float),
            "extent_pc": t["extent_pc"].to_numpy(dtype=float),
            "M_HI_1e4": t["M_HI_1e5"].to_numpy() * 10.0,
            "M_GMC_1e4": t["M_GMC_1e4"].to_numpy(dtype=float),
        },
        index=t["N"].to_numpy().astype(int),
    )


def beta_rot(Omega_kms_pc, R_pc, M_Msun):
    """Uniform-sphere solid-body rotational energy fraction Omega^2 R^3 / (3 G M)."""
    O = np.asarray(Omega_kms_pc, float) * (u.km / u.s / u.pc)
    R = np.asarray(R_pc, float) * u.pc
    M = np.asarray(M_Msun, float) * u.Msun
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (O**2 * R**3 / (3 * const.G * M)).to(u.dimensionless_unscaled)
    return b.value


def _load_vlsr():
    """Rosolowsky+2003 Table 2 VLSR [km/s], indexed by cloud N.

    The catalogue's own measured CO systemic velocity for each cloud -- used to
    centre analyze_gmc_box()'s fixed velocity window, rather than fitting one
    from this cube.
    """
    t = pd.read_csv(TAB2, comment="#")
    return pd.Series(t["VLSR"].to_numpy(dtype=float),
                      index=t["Cloud"].to_numpy().astype(int))


def analyze_gmc_box(gmc, cube, vlsr, box_pc=BOX_PC, vwin_kms=VWIN_KMS):
    """HI envelope as a fixed box_pc x box_pc x vwin_kms cutout centred on the
    GMC's catalogue position and VLSR.

    Replaces mekuria_imara_pipeline.analyze_gmc()'s steps [1]-[3] (the
    mass-dependent R_HI_cylinder aperture and the Gaussian-fit kinematic
    association) with a fixed geometry: a plain sky-plane square (dx_pc,
    dy_pc within +/-box_pc/2) and a fixed velocity window
    (VLSR +/- vwin_kms/2). Steps [4]-[8] -- moment0, the first-moment velocity
    field, the plane fit for Omega/theta, the Pearson planarity test, and
    j = c R^2 Omega -- are unchanged, reusing the pipeline's own functions, with
    R fixed at box_pc/2 (there is no per-cloud size left to measure once the
    aperture is fixed). Returns a mekuria_imara_pipeline.GMCResult, with
    vbar=VLSR and sig=vwin_kms/4 so the existing plotting helpers (which mark
    vbar +/- nsigma*sig at their default nsigma=2) still draw the true
    +/-vwin_kms/2 edges -- or None if the cutout runs off the map edge, the
    catalogue has no VLSR for this cloud, or the plane fit is under-determined.
    """
    name = f"GMC-{int(gmc['N'])}"
    half = box_pc / 2.0
    r_pix = half / cube.pc_per_pix

    xc, yc = cube.wcs.wcs_world2pix(gmc["RAdeg"], gmc["DEdeg"], 0)
    xc, yc = float(xc), float(yc)
    x0, x1 = max(0, int(xc - r_pix - 1)), min(cube.nx, int(xc + r_pix + 2))
    y0, y1 = max(0, int(yc - r_pix - 1)), min(cube.ny, int(yc + r_pix + 2))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None

    vbar = float(vlsr.get(int(gmc["N"]), np.nan))
    if not np.isfinite(vbar):
        return None
    vmask = np.abs(cube.vax - vbar) <= vwin_kms / 2.0
    if vmask.sum() < 2:
        return None

    sub = np.asarray(cube.hdu.data[mp.FGND_CHAN:, y0:y1, x0:x1], dtype=float)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx_pc = (xx - xc) * cube.pc_per_pix
    dy_pc = (yy - yc) * cube.pc_per_pix
    ap = (np.abs(dx_pc) <= half) & (np.abs(dy_pc) <= half)

    subw = sub[vmask]
    vchan = cube.vax[vmask]

    # [4] HI mass over the fixed window (optically thin)
    m0 = mp.moment0(subw, vchan)                                  # K km/s
    N_HI = mp.NHI_COEFF * m0                                      # cm^-2
    mp_Msun = const.m_p.to(u.Msun).value
    pix_area = (cube.pc_per_pix * u.pc).to(u.cm).value ** 2
    M_HI = mp.MU * mp_Msun * np.nansum(N_HI[ap] * pix_area)       # Msun
    sigma_HI = mp.MU * mp_Msun * N_HI * u.pc.to(u.cm) ** 2        # Msun / pc^2

    # [5] velocity field
    vfield = mp.first_moment(subw, vchan)

    # [6] plane fit -> Omega, theta (plain sky-plane offsets, not disk-deprojected)
    fit = mp.fit_plane(vfield, dx_pc, dy_pc, ap)
    if fit is None:
        return None
    Wx, Wy, theta, px, py, pv = fit
    Omega = np.hypot(Wx, Wy)

    theta_pix = np.degrees(np.arctan2(Wy, Wx)) % 360.0
    du, dw = mp.deproject_direction(Wx, Wy, cube.hdr)
    theta_disk = np.degrees(np.arctan2(dw, du)) % 360.0
    theta_PA = float(mp.sky_position_angle(theta_pix, cube.hdr))

    # [7] planarity
    rho = mp.planarity_pearson(Wx, Wy, px, py, pv)
    proj = (px * Wx + py * Wy) / Omega if Omega else np.zeros_like(pv)

    # [8] specific angular momentum, R fixed at the box half-width
    R_pc = half
    j = mp.specific_angular_momentum(R_pc, Omega)

    spec = np.nansum(sub * ap[None, :, :], axis=(1, 2))

    return mp.GMCResult(
        name=name, gmc=gmc, R_pc=R_pc, xc=xc, yc=yc,
        extent=(dx_pc.min(), dx_pc.max(), dy_pc.min(), dy_pc.max()),
        dx_pc=dx_pc, dy_pc=dy_pc, ap=ap, spec=spec, vax=cube.vax,
        vbar=vbar, sig=vwin_kms / 4.0, nsigma=2.0, vmask=vmask, vchan=vchan,
        m0=m0, N_HI=N_HI, sigma_HI=sigma_HI, M_HI=M_HI, vfield=vfield,
        Wx=Wx, Wy=Wy, Omega=Omega, theta=theta, theta_pix=theta_pix,
        theta_PA=theta_PA, theta_disk=theta_disk, frame="sky", rho=rho, j=j,
        proj=proj, pv=pv,
    )


def analyse(cube, cat, vlsr):
    """Run analyze_gmc_box over the catalog -> (Table-2 DataFrame, {name: GMCResult})."""
    rows, results = [], {}
    for _, gmc in cat.iterrows():
        res = analyze_gmc_box(gmc, cube, vlsr)
        if res is None:
            print("  rejected GMC-{}".format(int(gmc["N"])))
            continue
        results[res.name] = res
        rows.append(mp.result_row(res))
    return pd.DataFrame(rows), results


def _aligned(tab, gmc):
    """gmc rows in tab order, keyed off the GMC-N in tab['Name']."""
    n = tab["Name"].str.removeprefix("GMC-").astype(int).to_numpy()
    return n, gmc.reindex(n)


def _grid_clouds(results, k=3):
    """Six GMCResults for the 2x3 grids: the k most planar clouds (top row) and
    the k least planar (bottom row), each row ordered by descending rho.

    The Rosolowsky+2003 sample is nearly all inner-disk and barely overlapping,
    so the pipeline's (rho x crowding) selection has no crowding axis to work
    with here; a plain planar / non-planar split is the informative one.
    """
    ranked = sorted(results.values(), key=lambda r: r.rho, reverse=True)
    return ranked[:k] + sorted(ranked[-k:], key=lambda r: r.rho, reverse=True)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig1(cube, results, sigma, gmc):
    """[A] Sigma_HI field, envelope apertures, and the HI and CO velocity-gradient
    arrows at each GMC.

    Grayscale (binary) background. The HI arrow points up-gradient from
    theta_pix (the pixel-frame convention meant for drawing on the sky); the CO
    arrow points along Rosolowsky+2003's tabulated gradient PA, converted to the
    pixel frame by inverting sky_position_angle. Both lengths scale mildly with
    their own |Omega|. Cropped to the inner disk where the BIMA GMCs sit.
    """
    # crop window: RA 1h35m40s -> 1h32m20s (20s of RA wider each side than the
    # 35m20s..32m40s ask), Dec 30d10' -> 31d10'
    ra_lim = [(1 + 35 / 60 + 40 / 3600) * 15.0, (1 + 32 / 60 + 20 / 3600) * 15.0]
    dec_lim = [30 + 10 / 60, 31 + 10 / 60]

    fig = plt.figure(figsize=(6.8, 8.8))          # portrait: the crop is tall
    ax = fig.add_subplot(projection=cube.wcs)
    im = ax.imshow(sigma, origin="lower", cmap="binary", vmin=0,
                   vmax=np.nanpercentile(sigma, 99.5))

    s1 = np.sign(cube.hdr["CDELT1"])
    s2 = np.sign(cube.hdr["CDELT2"])
    om_hi_med = np.median([r.Omega for r in results.values()])
    og_med = np.nanmedian(gmc["Omega_GMC"].to_numpy())
    C_GMC = "#ffb000"                              # amber: reads on black or white

    for r in results.values():
        n = int(r.name.removeprefix("GMC-"))
        rp = r.R_pc / cube.pc_per_pix           # box half-width, in pixels
        ax.add_patch(Rectangle((r.xc - rp, r.yc - rp), 2 * rp, 2 * rp,
                               fill=False, ec="cyan", lw=0.6, alpha=0.85))
        # HI gradient
        s = np.clip(r.Omega / om_hi_med, 0.4, 2.5)
        th = np.deg2rad(r.theta_pix)
        ax.arrow(r.xc, r.yc, 1.8 * rp * s * np.cos(th),
                 1.8 * rp * s * np.sin(th), color=mp.C_HI, width=0.12,
                 head_width=1.1, head_length=1.8, length_includes_head=True,
                 alpha=0.95, zorder=5)
        # CO gradient: PA (E of N) -> pixel-frame angle is atan2(s2 cos, s1 sin)
        pa = np.deg2rad(gmc.at[n, "Phi_GMC"])
        thg = np.arctan2(s2 * np.cos(pa), s1 * np.sin(pa))
        sg = np.clip(gmc.at[n, "Omega_GMC"] / og_med, 0.4, 2.5)
        ax.arrow(r.xc, r.yc, 1.8 * rp * sg * np.cos(thg),
                 1.8 * rp * sg * np.sin(thg), color=C_GMC, width=0.12,
                 head_width=1.1, head_length=1.8, length_includes_head=True,
                 alpha=0.95, zorder=6)

    ax.plot([], [], color=mp.C_HI, lw=2, label="HI gradient")
    ax.plot([], [], color=C_GMC, lw=2, label="CO gradient (Rosolowsky+2003)")
    ax.legend(loc="lower left", frameon=True, fontsize=9)

    # set the crop in pixel space (works whatever the CDELT signs are)
    cx, cy = cube.wcs.wcs_world2pix(ra_lim * 2, dec_lim[:1] * 2 + dec_lim[1:] * 2, 0)
    ax.set_xlim(cx.min(), cx.max())
    ax.set_ylim(cy.min(), cy.max())

    mp._label_sky_axes(ax, cube.hdr)
    for c, side in ((ax.coords[0], "b"), (ax.coords[1], "l")):   # keep RA on the
        c.set_ticklabel_position(side)                           # bottom, Dec on
        c.set_axislabel_position(side)                           # the left
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(r"$\Sigma_{\rm HI}$  [$M_\odot$ pc$^{-2}$]")
    ax.set_title("M33 HI field (IBB full-res cube) -- Rosolowsky+2003 GMC "
                 "envelopes (n={}), HI and CO velocity gradients".format(
                     len(results)), pad=12)
    return fig


def fig2(results):
    """[B] 2x3 grid of aperture spectra: rho high (top) / rho low (bottom)."""
    six = _grid_clouds(results)
    return mp.plot_envelope_spectrum_grid(
        six, row_titles=[r"$\rho$ high", r"$\rho$ low"],
        notes=[r"$\rho$ = {:.2f}".format(r.rho) for r in six])


def fig3(results):
    """[B] 2x3 grid of v vs r_perp planar fits: rho high (top) / rho low (bottom).

    plot_velocity_profile already annotates rho in each panel, so no `notes`.
    """
    six = _grid_clouds(results)
    return mp.plot_velocity_profile_grid(
        six, row_titles=[r"$\rho$ high", r"$\rho$ low"])


def fig4(tab):
    """[C] Planarity summary -- rho distribution, then theta and j against rho."""
    return mp.summary_stats(tab)


def _pair_hist(ax, hi, gmc, xlabel, log=True, nbins=18, bins=None,
               circular=False, fmt="{:.2g}", hi_ref=None, gmc_ref=None):
    """HI (red) and CO GMC (blue) histograms of one quantity as filled bars,
    with optional Mekuria & Imara Table 2 counterparts as unfilled (step) bars
    in the same colours. `circular` switches the centre statistic from the
    median to the circular mean (position angles)."""
    def clean(a):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        return a[a > 0] if log else a

    hi, gmc = clean(hi), clean(gmc)
    series = [hi, gmc]
    if hi_ref is not None:
        hi_ref = clean(hi_ref)
        series.append(hi_ref)
    if gmc_ref is not None:
        gmc_ref = clean(gmc_ref)
        series.append(gmc_ref)
    if bins is None:
        lo = min(a.min() for a in series)
        hix = max(a.max() for a in series)
        bins = (np.logspace(np.log10(lo), np.log10(hix), nbins) if log
                else np.linspace(lo, hix, nbins))

    def centre(a):
        if circular:
            r = np.deg2rad(a)
            return np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))
        return np.median(a)

    tag = "mean" if circular else "med"

    def lab(name, a):
        return "{}  ({} {}, n={})".format(name, tag, fmt.format(centre(a)), a.size)

    ax.hist(hi, bins=bins, color=mp.C_HI, alpha=0.8, label=lab("HI", hi))
    ax.hist(gmc, bins=bins, color=mp.C_H2, alpha=0.6, label=lab("CO GMC", gmc))
    if hi_ref is not None:
        ax.hist(hi_ref, bins=bins, histtype="step", color=mp.C_HI, lw=1.8,
                ls="--", label=lab("HI M&I", hi_ref))
    if gmc_ref is not None:
        ax.hist(gmc_ref, bins=bins, histtype="step", color=mp.C_H2, lw=1.8,
                ls="--", label=lab("CO GMC M&I", gmc_ref))
    ax.axvline(centre(hi), color=mp.C_HI, ls="--", lw=1.0, alpha=0.5)
    ax.axvline(centre(gmc), color=mp.C_H2, ls="--", lw=1.0, alpha=0.5)
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("N")
    ax.legend(frameon=False, fontsize=7)
    mp._style(ax)


def fig5(tab, gmc, mi):
    """[C] The fig6 quantities as distributions -- Omega, j, M and the gradient
    position angle -- a separate HI and CO histogram per panel (this work,
    filled) with the Mekuria & Imara (2026) Table 2 values overplotted as
    unfilled step histograms.

    M&I Table 2 does not tabulate j, so the j panel's M&I curves use their
    Table 2 Omega with the same per-cloud radii this work uses.
    """
    _, g = _aligned(tab, gmc)
    _, m = _aligned(tab, mi)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))

    _pair_hist(axes[0, 0], tab["Omega_HI_1e-2"].to_numpy(),
               g["Omega_GMC"].to_numpy() * 100.0,
               r"$\Omega$  [$10^{-2}$ km s$^{-1}$ pc$^{-1}$]",
               hi_ref=m["Omega_HI_1e-2"].to_numpy(),
               gmc_ref=m["Omega_GMC_1e-2"].to_numpy())

    jfac = mp.C_MOI * PC_KMS_TO_1E24 / 10.0            # R[pc]^2 Omega[km/s/pc] -> 1e25 cm^2/s
    r_hi = tab["R_HI_pc"].to_numpy()
    r_g = g["R_GMC_pc"].to_numpy()
    _pair_hist(axes[0, 1], tab["j_HI_1e24"].to_numpy() / 10.0,
               g["j_GMC_1e24"].to_numpy() / 10.0,
               r"$j$  [$10^{25}$ cm$^2$ s$^{-1}$]   (M&I: their $\Omega$, same $R$)",
               hi_ref=jfac * r_hi**2 * m["Omega_HI_1e-2"].to_numpy() * 1e-2,
               gmc_ref=jfac * r_g**2 * m["Omega_GMC_1e-2"].to_numpy() * 1e-2)

    _pair_hist(axes[1, 0], tab["M_HI_1e4"].to_numpy(),
               g["M_GMC_1e4"].to_numpy(),
               r"$M$  [$10^4\,M_\odot$]",
               hi_ref=m["M_HI_1e4"].to_numpy(),
               gmc_ref=m["M_GMC_1e4"].to_numpy())

    _pair_hist(axes[1, 1], mp._wrap180(tab["theta_PA_deg"].to_numpy()),
               mp._wrap180(g["Phi_GMC"].to_numpy()),
               r"gradient position angle  [deg]",
               log=False, bins=np.linspace(-180, 180, 19),
               circular=True, fmt="{:.0f}$^\\circ$",
               hi_ref=mp._wrap180(m["theta_HI_deg"].to_numpy()),
               gmc_ref=mp._wrap180(m["theta_GMC_deg"].to_numpy()))

    fig.suptitle("HI vs CO GMC distributions (IBB full-res cube) -- this work "
                 "(filled) vs Mekuria & Imara 2026, Table 2 (step)", y=1.0)
    fig.tight_layout()
    return fig


def fig6(tab, gmc):
    """[B] Per-cloud HI-to-GMC ratios: Omega, j, M as log ratios; Phi as the
    folded gradient-axis offset."""
    _, g = _aligned(tab, gmc)
    om_hi = tab["Omega_HI_1e-2"].to_numpy() / 100.0
    j_hi = tab["j_HI_1e24"].to_numpy()
    m_hi = tab["M_HI_1e4"].to_numpy()
    pa_hi = tab["theta_PA_deg"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))

    def ratio_panel(ax, x, label):
        x = x[np.isfinite(x) & (x > 0)]
        lr = np.log10(x)
        ax.hist(lr, bins=18, color=mp.C_HI, alpha=0.85)
        ax.axvline(0.0, color=mp.C_INK, lw=1.0)
        ax.axvline(np.median(lr), color=mp.C_INK, ls="--", lw=1.4)
        ax.annotate("median = {:.2g}  (n={})".format(10 ** np.median(lr), x.size),
                    xy=(0.04, 0.9), xycoords="axes fraction", fontsize=9,
                    color=mp.C_INK)
        ax.set_xlabel(r"$\log_{10}\,$" + label)
        ax.set_ylabel("N")
        mp._style(ax)

    ratio_panel(axes[0, 0], om_hi / g["Omega_GMC"].to_numpy(),
                r"$\Omega_{\rm HI}/\Omega_{\rm GMC}$")
    ratio_panel(axes[0, 1], j_hi / g["j_GMC_1e24"].to_numpy(),
                r"$j_{\rm HI}/j_{\rm GMC}$")
    ratio_panel(axes[1, 0], m_hi / g["M_GMC_1e4"].to_numpy(),
                r"$M_{\rm HI}/M_{\rm GMC}$")

    d = mp._wrap180(pa_hi - g["Phi_GMC"].to_numpy())
    d = np.where(d > 90, d - 180, np.where(d < -90, d + 180, d))   # axis, not vector
    ax = axes[1, 1]
    d = d[np.isfinite(d)]
    ax.hist(d, bins=np.linspace(-90, 90, 19), color=mp.C_HI, alpha=0.85)
    ax.axvline(0.0, color=mp.C_INK, lw=1.0)
    ax.axvline(np.median(d), color=mp.C_INK, ls="--", lw=1.4)
    ax.annotate("median $\\Delta\\Phi$ = {:.0f}$^\\circ$  (n={})".format(
                    np.median(d), d.size),
                xy=(0.04, 0.9), xycoords="axes fraction", fontsize=9,
                color=mp.C_INK)
    ax.set_xlabel(r"$\Phi_{\rm HI}-\Phi_{\rm GMC}$ [deg]  "
                  r"(gradient axis, folded to $\pm90^\circ$)")
    ax.set_ylabel("N")
    mp._style(ax)

    fig.suptitle("HI envelope vs Rosolowsky+2003 CO GMC -- per-cloud comparison "
                 "(IBB full-res cube)", y=1.0)
    fig.tight_layout()
    return fig


def fig7(tab, gmc, ibb=None):
    """[D] Omega vs cloud radius -- HI points at R_HI, CO points at R_GMC --
    with the size histograms on top.  This work is drawn as filled markers;
    Imara, Bigiel & Blitz (2011) Table 1 is overplotted as open markers (HI at
    its gradient extent, GMC at the Rosolowsky+2003 radius on IBB's 840 kpc
    scale).  Omega is in km/s/pc, no 1e-2 rescaling."""
    if ibb is None:
        ibb = load_ibb_tab1()
    n, g = _aligned(tab, gmc)
    gi = ibb.reindex(n)

    om_hi = tab["Omega_HI_1e-2"].to_numpy() / 100.0   # -> km/s/pc
    r_hi = tab["R_HI_pc"].to_numpy()
    om_g = g["Omega_GMC"].to_numpy()                  # already km/s/pc
    r_g = g["R_GMC_pc"].to_numpy()

    om_hi_i = gi["Omega_HI_1e-2"].to_numpy() / 100.0
    ext_i = gi["extent_pc"].to_numpy()                # IBB's HI-side size
    om_g_i = gi["Omega_GMC_1e-2"].to_numpy() / 100.0
    r_g_i = r_g / DFAC                                # undo the 859/840 rescale

    fig = plt.figure(figsize=(7.6, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.06)
    axh = fig.add_subplot(gs[0])
    axm = fig.add_subplot(gs[1], sharex=axh)

    lo = np.nanmin([np.nanmin(r_hi), np.nanmin(r_g), np.nanmin(ext_i)])
    hi = np.nanmax([np.nanmax(r_hi), np.nanmax(r_g), np.nanmax(ext_i)])
    bins = np.logspace(np.log10(lo), np.log10(hi), 16)
    axh.hist(r_hi, bins=bins, color=mp.C_HI, alpha=0.8, label="HI envelope")
    axh.hist(r_g[np.isfinite(r_g)], bins=bins, color=mp.C_H2, alpha=0.6,
             label="CO GMC")
    axh.hist(ext_i[np.isfinite(ext_i)], bins=bins, histtype="step",
             color=mp.C_HI, lw=1.3, label="IBB HI extent")
    axh.set_xscale("log")
    axh.tick_params(labelbottom=False)
    axh.set_ylabel("N")
    axh.legend(frameon=False, fontsize=8)
    mp._style(axh)

    axm.scatter(r_hi, om_hi, s=32, color=mp.C_HI, alpha=0.85, lw=0,
                label=r"HI envelope ($R_{\rm HI}$)")
    axm.scatter(r_g, om_g, s=32, marker="s", color=mp.C_H2, alpha=0.85, lw=0,
                label=r"CO GMC ($R_{\rm GMC}$)")
    axm.scatter(ext_i, om_hi_i, s=34, marker="o", facecolors="none",
                edgecolors=mp.C_HI, lw=1.1, alpha=0.9,
                label="IBB 2011 HI (extent)")
    axm.scatter(r_g_i, om_g_i, s=34, marker="s", facecolors="none",
                edgecolors=mp.C_H2, lw=1.1, alpha=0.9, label="IBB 2011 GMC")
    axm.set_xscale("log")
    axm.set_yscale("log")
    axm.set_xlabel("cloud radius  [pc]")
    axm.set_ylabel(r"$\Omega$  [km s$^{-1}$ pc$^{-1}$]")
    axm.legend(frameon=False, fontsize=8, ncol=2)
    mp._style(axm)

    fig.suptitle("Velocity gradient vs cloud radius (IBB full-res cube)", y=0.94)
    return fig


def _jsize_panel(ax, R, j, color, xlabel, keep=None,
                 fitted_label="fitted", excluded_label=None):
    """One log-log j vs R panel with a power-law fit, a +/-1 sigma scatter band
    and a +/-1 sigma fit-uncertainty band -- the pipeline's _size_relation, but
    with the colour, x label and legend text passed in so it works for the CO
    clouds as well as the HI envelopes."""
    R = np.asarray(R, float)
    j = np.asarray(j, float)
    finite = np.isfinite(R) & np.isfinite(j)
    keep = finite if keep is None else (np.asarray(keep, bool) & finite)

    ax.scatter(R[keep], j[keep], s=24, color=color, lw=0, alpha=0.85, zorder=2,
               label=fitted_label)
    exc = finite & ~keep
    if excluded_label and exc.any():
        ax.scatter(R[exc], j[exc], s=24, facecolors="none", edgecolors=color,
                   lw=1.0, alpha=0.85, zorder=2, label=excluded_label)

    # analyze_gmc_box() gives every HI envelope the same fixed R (box_pc/2), so
    # R[keep] can have zero spread -- np.polyfit(..., cov=True) on a single x
    # value is singular and raises, rather than just failing to fit. Skip the
    # fit in that case instead of letting it crash the whole figure.
    degenerate = keep.sum() < 3 or np.ptp(np.log10(R[keep])) == 0
    fit = None if degenerate else mp._loglog_fit(R[keep], j[keep])
    if degenerate and keep.sum():
        ax.annotate("R fixed at {:.0f} pc -- no size relation to fit".format(
                        np.median(R[keep])),
                    xy=(0.04, 0.9), xycoords="axes fraction", fontsize=10,
                    color=mp.C_INK)
    if fit:
        slope, err, icept, pivot = fit
        grid = np.logspace(np.log10(R[keep].min()), np.log10(R[keep].max()), 50)
        lg = np.log10(grid)
        model = icept + slope * lg
        lx, ly = np.log10(R[keep]), np.log10(j[keep])
        rms = float((ly - (icept + slope * lx)).std(ddof=2))
        ax.fill_between(grid, 10 ** (model - rms), 10 ** (model + rms),
                        color=color, alpha=0.30, lw=0, zorder=1,
                        label=r"$\pm1\sigma$ scatter ({:.2f} dex)".format(rms))
        band = err * np.abs(lg - pivot)
        ax.fill_between(grid, 10 ** (model - band), 10 ** (model + band),
                        color="black", alpha=1.0, lw=0, zorder=3,
                        label=r"$\pm1\sigma$ fit uncertainty")
        ax.plot(grid, 10 ** model, color=mp.C_GRID, ls="--", lw=0.7, zorder=4)
        ax.annotate(r"$j \propto R^{{{:.2f}\pm{:.2f}}}$".format(slope, err),
                    xy=(0.04, 0.9), xycoords="axes fraction", fontsize=11,
                    color=mp.C_INK)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$j$  [$10^{25}$ cm$^2$ s$^{-1}$]")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    mp._style(ax)


def fig8(tab, gmc):
    """[C] j vs cloud size, log-log with power-law fits: CO GMCs (left, R_GMC)
    and HI envelopes (right, R_HI).  Every resolved CO GMC is fitted; HI systems
    with rho <= 0.75 are drawn open and dropped from the fit."""
    _, g = _aligned(tab, gmc)
    fig, (axg, axh) = plt.subplots(1, 2, figsize=(11.4, 4.7), sharey=True)

    _jsize_panel(axg, g["R_GMC_pc"].to_numpy(), g["j_GMC_1e24"].to_numpy() / 10.0,
                 mp.C_H2, r"$R_{\rm GMC}$ [pc]", fitted_label=r"CO GMC ($j_1$)")
    _jsize_panel(axh, tab["R_HI_pc"].to_numpy(), tab["j_HI_1e24"].to_numpy() / 10.0,
                 mp.C_HI, r"$R_{\rm HI}$ [pc]",
                 keep=tab["rho_HI"].to_numpy() > 0.75,
                 fitted_label=r"$\rho > 0.75$ (fitted)",
                 excluded_label=r"$\rho \leq 0.75$ (excluded)")

    axg.set_title("CO GMCs -- Rosolowsky+2003")
    axh.set_title("HI envelopes (IBB full-res cube)")
    axh.set_ylabel("")
    fig.tight_layout()
    return fig


def fig9(tab, gmc):
    """Histograms of specific angular momentum, HI envelopes vs CO GMCs."""
    _, g = _aligned(tab, gmc)
    j_hi = tab["j_HI_1e24"].to_numpy() / 10.0          # -> 1e25 cm^2/s
    j_g = g["j_GMC_1e24"].to_numpy() / 10.0
    j_g = j_g[np.isfinite(j_g) & (j_g > 0)]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    lo = min(j_hi.min(), j_g.min())
    hi = max(j_hi.max(), j_g.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 20)
    ax.hist(j_hi, bins=bins, color=mp.C_HI, alpha=0.8,
            label="HI envelope  (median {:.1f})".format(np.median(j_hi)))
    ax.hist(j_g, bins=bins, color=mp.C_H2, alpha=0.6,
            label=r"CO GMC, $j_1$  (median {:.2f}, n={})".format(
                np.median(j_g), j_g.size))
    ax.axvline(np.median(j_hi), color=mp.C_HI, ls="--", lw=1.4)
    ax.axvline(np.median(j_g), color=mp.C_H2, ls="--", lw=1.4)
    ax.set_xscale("log")
    ax.set_xlabel(r"$j$  [$10^{25}$ cm$^2$ s$^{-1}$]")
    ax.set_ylabel("N")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Specific angular momentum: HI envelopes (IBB full-res cube) "
                 "vs CO GMCs")
    mp._style(ax)
    return fig


def fig10(tab, gmc):
    """Histograms of beta_rot for the HI envelopes and the CO GMCs."""
    _, g = _aligned(tab, gmc)
    b_hi = beta_rot(tab["Omega_HI_1e-2"].to_numpy() / 100.0,
                    tab["R_HI_pc"].to_numpy(),
                    tab["M_HI_1e4"].to_numpy() * 1e4)
    b_g = beta_rot(g["Omega_GMC"].to_numpy(),
                   g["R_GMC_pc"].to_numpy(),
                   g["M_GMC_1e4"].to_numpy() * 1e4)
    b_hi = b_hi[np.isfinite(b_hi) & (b_hi > 0)]
    b_g = b_g[np.isfinite(b_g) & (b_g > 0)]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    lo = min(b_hi.min(), b_g.min())
    hi = max(b_hi.max(), b_g.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 33)
    ax.hist(b_hi, bins=bins, color=mp.C_HI, alpha=0.8,
            label=r"$\beta_{{\rm rot,HI}}$  (median {:.2g})".format(np.median(b_hi)))
    ax.hist(b_g, bins=bins, color=mp.C_H2, alpha=0.6,
            label=r"$\beta_{{\rm rot,GMC}}$  (median {:.2g}, n={})".format(
                np.median(b_g), b_g.size))
    ax.axvline(np.median(b_hi), color=mp.C_HI, ls="--", lw=1.4)
    ax.axvline(np.median(b_g), color=mp.C_H2, ls="--", lw=1.4)
    ax.axvline(1.0, color=mp.C_INK, ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta_{\rm rot} = E_{\rm rot}/|E_{\rm grav}| "
                  r"= \Omega^2 R^3 / (3 G M)$")
    ax.set_ylabel("N")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Rotational energy fraction (uniform sphere, solid body)")
    mp._style(ax)
    return fig


def fig11(tab, ibb=None):
    """Pipeline R_HI (y) vs the IBB 2011 Table 1 HI-gradient extent (x), one
    point per cloud, with the 1:1 line dashed.  Only the 39 clouds for which
    IBB fitted a linear HI gradient (finite extent) appear."""
    if ibb is None:
        ibb = load_ibb_tab1()
    n = tab["Name"].str.removeprefix("GMC-").astype(int).to_numpy()
    gi = ibb.reindex(n)

    ext = gi["extent_pc"].to_numpy()
    r_hi = tab["R_HI_pc"].to_numpy()
    m = np.isfinite(ext) & np.isfinite(r_hi) & (ext > 0) & (r_hi > 0)
    ext, r_hi = ext[m], r_hi[m]
    ratio = r_hi / ext

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    lo = 0.9 * min(ext.min(), r_hi.min())
    hi = 1.1 * max(ext.max(), r_hi.max())
    ax.plot([lo, hi], [lo, hi], ls="--", color=mp.C_INK, lw=1.2, label="1:1")
    ax.scatter(ext, r_hi, s=34, color=mp.C_HI, alpha=0.85, lw=0,
               label="M33 clouds (n={})".format(ext.size))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("IBB 2011 HI gradient extent  [pc]")
    ax.set_ylabel(r"pipeline $R_{\rm HI}$  [pc]  (IBB full-res cube)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title(r"$R_{{\rm HI}}$ vs IBB extent  "
                 r"(median $R_{{\rm HI}}/{{\rm extent}}$ = {:.2f})".format(
                     np.median(ratio)))
    mp._style(ax)
    return fig


def _connect_common(ax, kept):
    """Thin gray segment between each cloud's two points, for clouds present
    in both of exactly two series. Both points share the same IBB x by
    construction (same cloud, same ibb_series lookup), so the segment is
    vertical. `kept` is a list of dicts with 'n', 'x', 'y' arrays (as built by
    _loglog_ratio_panel and fig12's theta panel); a no-op unless there are
    exactly two series, since "corresponding point" is undefined otherwise.
    """
    if len(kept) != 2:
        return
    a, b = kept
    _, ia, ib_ = np.intersect1d(a["n"], b["n"], return_indices=True)
    for xi, ya, yb in zip(a["x"][ia], a["y"][ia], b["y"][ib_]):
        ax.plot([xi, xi], [ya, yb], color="0.6", lw=0.8, alpha=0.6, zorder=1)


def _scatter_by_rho(ax, k, rho_cut=0.75):
    """Scatter one series' points, split into filled (rho_HI >= rho_cut) and
    open (rho_HI < rho_cut) markers -- the same "planarity fit is untrustworthy
    below this threshold" convention fig8's _jsize_panel uses for the size-
    relation fits. `k` is one of _loglog_ratio_panel's / fig12's theta panel's
    `kept` dicts, now carrying a 'rho' array alongside x/y. The legend entry is
    attached to whichever of the two calls is non-empty first, so each series
    gets exactly one legend line regardless of the filled/open split.
    """
    filled = k["rho"] >= rho_cut
    label = "{} (n={})".format(k["lab"], k["x"].size)
    if filled.any():
        ax.scatter(k["x"][filled], k["y"][filled], s=34, color=k["color"],
                   marker=k["marker"], alpha=0.85, lw=0, zorder=2, label=label)
        label = None
    if (~filled).any():
        ax.scatter(k["x"][~filled], k["y"][~filled], s=34, facecolors="none",
                   edgecolors=k["color"], marker=k["marker"], lw=1.1,
                   alpha=0.85, zorder=2, label=label)


def _loglog_ratio_panel(ax, ibb_series, series, label, unit="", rho_cut=0.75):
    """One log-log 1:1 panel: IBB's per-cloud value (indexed by cloud N) on
    x, one or more of this work's own per-cloud series on y, for a quantity
    where a ratio is the natural comparison (Omega, M -- both strictly
    positive and multiplicative). Shared by fig12's Omega and M panels; theta
    gets its own panel since an angle has no ratio.

    `series` is a list of (n, y_values, rho_values, label, color, marker)
    tuples. Each is matched against ibb_series independently via n (rather
    than sharing one pre-aligned x) because different series -- e.g. the same
    box-cutout method run on two different cubes -- can keep different
    clouds. Points with rho_HI < rho_cut are drawn open (_scatter_by_rho),
    since the plane fit behind Omega/theta is unreliable there. When there
    are exactly two series, _connect_common draws a line between each cloud's
    two points (only for clouds both series kept).

    x and y get their own independent limits (no set_aspect("equal")): IBB's
    own values (x) and this work's fitted values (y) can have very different
    ranges -- a poorly-constrained fit can put a y-point an order of magnitude
    below anything IBB reports -- and forcing a shared square range to keep
    the 1:1 line at a literal 45 degrees was wasting most of the x-axis on a
    region no IBB point ever falls in. The 1:1 line is still drawn (spanning
    the full combined range, clipped to each axis' own limits by matplotlib),
    it just will not visually read as exactly diagonal any more.
    """
    kept, all_x, all_y, titles = [], [], [], []
    for n, y, rho, lab, color, marker in series:
        x = ibb_series.reindex(n).to_numpy(dtype=float)
        y = np.asarray(y, float)
        rho = np.asarray(rho, float)
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        n_k, x_k, y_k, rho_k = np.asarray(n)[ok], x[ok], y[ok], rho[ok]
        kept.append(dict(n=n_k, x=x_k, y=y_k, rho=rho_k, lab=lab, color=color,
                         marker=marker))
        all_x.append(x_k)
        all_y.append(y_k)
        titles.append("{}: {:.2f}".format(lab, np.median(y_k / x_k)))

    _connect_common(ax, kept)
    for k in kept:
        _scatter_by_rho(ax, k, rho_cut=rho_cut)

    lo_x = 0.8 * min(a.min() for a in all_x if a.size)
    hi_x = 1.2 * max(a.max() for a in all_x if a.size)
    lo_y = 0.8 * min(a.min() for a in all_y if a.size)
    hi_y = 1.2 * max(a.max() for a in all_y if a.size)
    diag_lo, diag_hi = min(lo_x, lo_y), max(hi_x, hi_y)
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], ls="--", color=mp.C_INK,
            lw=1.2, zorder=0, label="1:1")
    ax.scatter([], [], facecolors="none", edgecolors="0.3", marker="o", lw=1.1,
               label=r"open: $\rho_{{\rm HI}}<{:.2f}$".format(rho_cut))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)
    ax.set_xlabel("IBB 2011  " + label + unit)
    ax.set_ylabel("this work  " + label + unit)
    ax.set_title("median ratio (this work / IBB):\n" + "\n".join(titles),
                 fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    mp._style(ax)


def fig12(tab, tab_mid=None, ibb=None):
    """Direct per-cloud comparison of this work's HI-side quantities (the
    box-cutout method) against Imara, Bigiel & Blitz (2011) Table 1's own
    published values, for the same 45 clouds: Omega_HI, the HI gradient
    position angle theta_HI, and M_HI.

    tab is the box-cutout method run on the IBB full-res cube (this module's
    main analysis); tab_mid, if given, is the same box-cutout method
    (analyze_gmc_box, same catalogue, same VLSR centring) run instead on the
    original mid-res cube (m33_C+D+tp_himidres_120pc.fits) -- included to see
    whether the box-cutout method's agreement with IBB depends on which cube's
    resolution the envelope is measured from, rather than on the method
    itself. The two series are matched against IBB independently since they
    can keep different clouds (a 140 pc box is ~45 full-res pixels across but
    only ~10 mid-res pixels, so edge/fit rejections can differ).

    Each panel is a 1:1 scatter -- departure from the dashed line is the
    disagreement. Omega and M get a log-ratio median (both are strictly
    positive, multiplicative quantities); theta has no ratio, so its panel
    instead reports the median folded angular offset. theta_HI in this work's
    own tables is stored in the pixel frame (theta_HI_deg) and as a sky
    position angle (theta_PA_deg, deg East of North) -- IBB's theta_HI_deg is
    the latter convention, so this work's theta_PA_deg is what is compared,
    wrapped onto IBB's (-180, 180] range with mp._wrap180.

    Points with this work's own rho_HI < 0.75 are drawn open in every panel
    (_scatter_by_rho) -- the plane fit behind Omega and theta is unreliable
    there, per the same rho cut fig8 uses to exclude such clouds from its
    size-relation fits.
    """
    if ibb is None:
        ibb = load_ibb_tab1()

    def _n(t):
        return t["Name"].str.removeprefix("GMC-").astype(int).to_numpy()

    series_meta = [(_n(tab), tab, "this work (full-res cube)", mp.C_HI, "o")]
    if tab_mid is not None:
        series_meta.append((_n(tab_mid), tab_mid,
                            "this work (mid-res cube)", "#00acc1", "^"))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))

    _loglog_ratio_panel(
        axes[0], ibb["Omega_HI_1e-2"],
        [(n, t["Omega_HI_1e-2"].to_numpy(), t["rho_HI"].to_numpy(), lab, c, m)
         for n, t, lab, c, m in series_meta],
        r"$\Omega_{\rm HI}$", r"  [$10^{-2}$ km s$^{-1}$ pc$^{-1}$]")

    ax = axes[1]
    ax.plot([-190, 190], [-190, 190], ls="--", color=mp.C_INK, lw=1.2,
            zorder=0, label="1:1")
    kept, dth_titles = [], []
    for n, t, lab, color, marker in series_meta:
        th_ibb = ibb["theta_HI_deg"].reindex(n).to_numpy(dtype=float)
        th_mine = mp._wrap180(t["theta_PA_deg"].to_numpy())
        rho = t["rho_HI"].to_numpy()
        ok = np.isfinite(th_ibb) & np.isfinite(th_mine)
        n_k = np.asarray(n)[ok]
        th_ibb_k, th_mine_k, rho_k = th_ibb[ok], th_mine[ok], rho[ok]
        kept.append(dict(n=n_k, x=th_ibb_k, y=th_mine_k, rho=rho_k, lab=lab,
                         color=color, marker=marker))
        dth = mp._wrap180(th_mine_k - th_ibb_k)
        dth_titles.append(r"{}: {:.0f}$^\circ$".format(lab, np.median(dth)))

    _connect_common(ax, kept)
    for k in kept:
        _scatter_by_rho(ax, k)
    ax.scatter([], [], facecolors="none", edgecolors="0.3", marker="o", lw=1.1,
               label=r"open: $\rho_{\rm HI}<0.75$")
    ax.set_xlim(-190, 190)
    ax.set_ylim(-190, 190)
    ax.set_xticks(np.arange(-180, 181, 90))
    ax.set_yticks(np.arange(-180, 181, 90))
    ax.set_aspect("equal")
    ax.set_xlabel(r"IBB 2011  $\theta_{\rm HI}$  [deg]")
    ax.set_ylabel(r"this work  $\theta_{\rm HI}$  [deg]")
    ax.set_title(r"median $\Delta\theta$ (this work $-$ IBB):" + "\n"
                 + "\n".join(dth_titles), fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    mp._style(ax)

    _loglog_ratio_panel(
        axes[2], ibb["M_HI_1e4"],
        [(n, t["M_HI_1e4"].to_numpy(), t["rho_HI"].to_numpy(), lab, c, m)
         for n, t, lab, c, m in series_meta],
        r"$M_{\rm HI}$", r"  [$10^4\,M_\odot$]")

    fig.suptitle("This work (box-cutout method) vs Imara, Bigiel & Blitz "
                 "(2011) Table 1 -- IBB full-res cube"
                 + (" vs the original mid-res cube" if tab_mid is not None
                    else ""), y=1.08)
    fig.tight_layout()
    return fig


def _scatter_open_by_rho(ax, x, y, rho, color, marker, label, rho_cut=0.75):
    """Scatter (x, y), split into filled (rho >= rho_cut) and open (rho <
    rho_cut) markers -- same convention as _scatter_by_rho, but taking plain
    arrays rather than a 'kept' dict, since fig13's x-axis is the cloud
    number itself rather than an IBB reference value. IBB's own points have
    no rho and are always plotted filled by the caller, not through this
    function.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rho = np.asarray(rho, float)
    filled = rho >= rho_cut
    lbl = label
    if filled.any():
        ax.scatter(x[filled], y[filled], s=42, color=color, marker=marker,
                   alpha=0.9, lw=0, zorder=3, label=lbl)
        lbl = None
    if (~filled).any():
        ax.scatter(x[~filled], y[~filled], s=42, facecolors="none",
                   edgecolors=color, marker=marker, lw=1.2, alpha=0.9,
                   zorder=3, label=lbl)


def fig13(tab, tab_mid, ibb):
    """Per-cloud profile view: one x position per cloud (GMC number 1-45),
    stacked vertically for Omega_HI, theta_HI and M_HI. At each cloud, three
    markers -- IBB 2011's own published value (circle), this work's box-
    cutout method on the IBB full-res cube (triangle), and the same method on
    the original mid-res cube (square) -- joined by a thin vertical line, so
    the column at each x reads as one comparison rather than three
    independent points.

    This work's two series are drawn open where their own rho_HI < 0.75
    (_scatter_open_by_rho); IBB's value carries no rho of its own and is
    always filled. theta_HI is this work's theta_PA_deg (sky position angle,
    deg East of North -- the same convention as IBB's own theta_HI_deg),
    wrapped onto IBB's (-180, 180] range with mp._wrap180, matching fig12.
    """
    n_hi = tab["Name"].str.removeprefix("GMC-").astype(int).to_numpy()
    n_lo = tab_mid["Name"].str.removeprefix("GMC-").astype(int).to_numpy()

    om_hi = pd.Series(tab["Omega_HI_1e-2"].to_numpy(), index=n_hi)
    om_lo = pd.Series(tab_mid["Omega_HI_1e-2"].to_numpy(), index=n_lo)
    th_hi = pd.Series(mp._wrap180(tab["theta_PA_deg"].to_numpy()), index=n_hi)
    th_lo = pd.Series(mp._wrap180(tab_mid["theta_PA_deg"].to_numpy()), index=n_lo)
    m_hi = pd.Series(tab["M_HI_1e4"].to_numpy(), index=n_hi)
    m_lo = pd.Series(tab_mid["M_HI_1e4"].to_numpy(), index=n_lo)
    rho_hi = pd.Series(tab["rho_HI"].to_numpy(), index=n_hi)
    rho_lo = pd.Series(tab_mid["rho_HI"].to_numpy(), index=n_lo)

    COL_IBB, COL_HI, COL_LO = "0.25", mp.C_HI, "#00acc1"

    # x-axis order: not GMC catalog number, but descending
    # M_HI(mid-res)/M_HI(full-res) ratio, so the ratio falls monotonically
    # left to right and crosses 1 where mid-res stops exceeding full-res.
    # Clouds missing a mass on one of the two cubes (so the ratio can't be
    # formed) are appended last, by ascending GMC number.
    common = np.sort(np.union1d(np.union1d(ibb.index.to_numpy(), n_hi), n_lo))
    both = np.intersect1d(n_hi, n_lo)
    ratio = m_lo.reindex(both).to_numpy() / m_hi.reindex(both).to_numpy()
    order_idx = np.argsort(-ratio)
    both_sorted, ratio_sorted = both[order_idx], ratio[order_idx]
    n_gt = int(np.sum(ratio_sorted > 1))   # M_HI(mid-res) > M_HI(full-res)
    n_lt = int(np.sum(ratio_sorted < 1))   # M_HI(mid-res) < M_HI(full-res)
    grp_other = np.sort(np.setdiff1d(common, both))
    order = np.concatenate([both_sorted, grp_other])
    pos_of = {n: i for i, n in enumerate(order)}

    def panel(ax, y_ibb, y_hi, y_lo, ylabel, log=False):
        for n in order:
            vals = [v for v in (y_ibb.get(n), y_hi.get(n), y_lo.get(n))
                    if v is not None and np.isfinite(v)]
            if len(vals) >= 2:
                x = pos_of[n]
                ax.plot([x, x], [min(vals), max(vals)], color="0.75", lw=0.8,
                        zorder=1)

        x_ibb = np.array([pos_of[n] for n in y_ibb.index])
        ax.scatter(x_ibb, y_ibb.to_numpy(), s=42,
                   color=COL_IBB, marker="o", alpha=0.9, lw=0, zorder=3,
                   label="IBB 2011")
        x_hi = np.array([pos_of[n] for n in y_hi.index])
        _scatter_open_by_rho(ax, x_hi, y_hi.to_numpy(),
                            rho_hi.reindex(y_hi.index).to_numpy(),
                            COL_HI, "^", "this work (full-res cube)")
        x_lo = np.array([pos_of[n] for n in y_lo.index])
        _scatter_open_by_rho(ax, x_lo, y_lo.to_numpy(),
                            rho_lo.reindex(y_lo.index).to_numpy(),
                            COL_LO, "s", "this work (mid-res cube)")
        if log:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        mp._style(ax)

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 11.5), sharex=True)

    panel(axes[0], ibb["Omega_HI_1e-2"], om_hi, om_lo,
         r"$\Omega_{\rm HI}$  [$10^{-2}$ km s$^{-1}$ pc$^{-1}$]", log=True)
    panel(axes[1], ibb["theta_HI_deg"], th_hi, th_lo,
         r"$\theta_{\rm HI}$  [deg]")
    axes[1].set_ylim(-190, 190)
    axes[1].set_yticks(np.arange(-180, 181, 90))
    panel(axes[2], ibb["M_HI_1e4"], m_hi, m_lo,
         r"$M_{\rm HI}$  [$10^4\,M_\odot$]", log=True)

    for ax in axes[:2]:
        ax.tick_params(labelbottom=False)

    axes[2].scatter([], [], facecolors="none", edgecolors="0.3", marker="o",
                    lw=1.1, label=r"open: $\rho_{\rm HI}<0.75$")
    axes[2].legend(frameon=False, fontsize=9, ncol=2, loc="lower left")

    n_ax = len(order)
    axes[2].set_xlabel("GMC number, ordered by descending "
                       r"$M_{\rm HI}^{\rm mid\,res} / M_{\rm HI}^{\rm full\,res}$"
                       " (Rosolowsky et al. 2003 catalog numbers)")
    axes[2].set_xlim(-1, n_ax)
    tick_pos = np.arange(0, n_ax, 5)
    axes[2].set_xticks(tick_pos)
    axes[2].set_xticklabels([str(order[i]) for i in tick_pos])

    for ax in axes:
        if 0 < n_gt < n_ax:
            ax.axvline(n_gt - 0.5, color="0.6", lw=0.8, ls="--", zorder=0)
        if n_gt + n_lt < n_ax:
            ax.axvline(n_gt + n_lt - 0.5, color="0.6", lw=0.8, ls="--",
                       zorder=0)

    label_kw = dict(transform=axes[0].get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=9, color="0.35")
    if n_gt:
        axes[0].text(n_gt / 2, 1.02, "mid-res $M_{\\rm HI}$ >", **label_kw)
    if n_lt:
        axes[0].text(n_gt + n_lt / 2, 1.02, "mid-res $M_{\\rm HI}$ <",
                     **label_kw)
    if len(grp_other):
        axes[0].text(n_gt + n_lt + len(grp_other) / 2, 1.02, "missing one res.",
                     **label_kw)

    fig.suptitle("Per-cloud profiles: IBB 2011 vs this work (box-cutout "
                 "method, two cube resolutions)", y=0.995)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for old in sorted(FIGDIR.glob("fig*.pdf")):     # clear the previous numbering
        old.unlink()

    cat = rc.load(path=str(TAB2))
    gmc = load_gmc_side(cat)
    mi = load_mi_tab2()
    vlsr = _load_vlsr()

    # this cube's velocity axis (-74 -> -310 km/s) never reaches the MW
    # foreground band the mid-res cube's default FGND_CHAN=60 trims -- see the
    # module docstring. open_cube() and analyze_gmc_box() both read the
    # constant off the module directly, so it is overridden here rather than
    # passed.
    mp.FGND_CHAN = 0
    # this cube's planes are ~24x larger than the mid-res cube's; keep
    # field_moments' per-block memory bounded on this 24 GB machine.
    cube = mp.open_cube(path=str(FULLRES_CUBE))

    tab, results = analyse(cube, cat, vlsr)
    tab.to_csv(OUT_CSV, index=False)
    print("catalog n={}  kept n={}  -> {}".format(len(cat), len(tab), OUT_CSV))

    sigma, _ = mp.field_moments(cube, step=4, chunk=8)

    # same box-cutout method (analyze_gmc_box via analyse()), run again on the
    # original mid-res cube -- lets fig12 show whether the box-cutout method's
    # agreement with IBB depends on cube resolution rather than the method
    # itself. Done after the FULLRES_CUBE figures above so their FGND_CHAN=0 stays
    # in effect for field_moments(); this cube's own velocity axis does reach
    # the MW foreground band, so it needs the pipeline's default trim back.
    mp.FGND_CHAN = 60
    cube_mid = mp.open_cube(path=str(MIDRES_CUBE))
    tab_mid, _ = analyse(cube_mid, cat, vlsr)
    tab_mid.to_csv(OUT_CSV_MIDRES, index=False)
    print("catalog n={}  kept n={}  -> {}".format(
        len(cat), len(tab_mid), OUT_CSV_MIDRES))

    ibb = load_ibb_tab1()

    figs = {
        "fig1_field_gradients": fig1(cube, results, sigma, gmc),
        "fig2_spectra": fig2(results),
        "fig3_profiles": fig3(results),
        "fig4_planarity": fig4(tab),
        "fig5_quantity_distributions": fig5(tab, gmc, mi),
        "fig6_hi_gmc_ratios": fig6(tab, gmc),
        "fig7_omega_vs_size": fig7(tab, gmc, ibb),
        "fig8_j_vs_size": fig8(tab, gmc),
        "fig9_j_histograms": fig9(tab, gmc),
        "fig10_beta_rot_histograms": fig10(tab, gmc),
        "fig11_rhi_vs_ibb_extent": fig11(tab, ibb),
        "fig12_vs_ibb_direct": fig12(tab, tab_mid, ibb),
        "fig13_per_cloud_profiles": fig13(tab, tab_mid, ibb),
    }
    for stem, fig in figs.items():
        path = FIGDIR / "{}.pdf".format(stem)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("  wrote", path)


if __name__ == "__main__":
    main()
