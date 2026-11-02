"""New figure set (the 2026-08-31 TARGET list) for the Rosolowsky+2003 BIMA
catalog of M33 GMCs.

Everything here is the HI envelopes and the CO GMCs, nothing subtracted -- the
"HI without GMCs" split from the notebook plan is still deferred.

    fig1   (A)  Sigma_HI field map + envelope apertures + HI velocity-gradient
                vectors
    fig2   (B)  grid of aperture spectra (2x3, rho high / rho low)
    fig3   (B)  grid of v vs r_perp planar fits (2x3, rho high / rho low)
    fig4   (C)  planarity summary: rho distribution, theta and j vs rho
    fig5   (C)  distributions of the fig6 quantities (Omega, j, M, gradient PA),
                as separate HI and CO histograms rather than ratios
    fig6   (B)  HI-to-GMC ratios per cloud: Omega, j, M, and the gradient-axis
                offset Phi
    fig7   (D)  Omega vs cloud radius (R_HI for HI, R_GMC for CO), Omega
                histograms on top
    fig8   (C)  j vs cloud size, log-log with power-law fits -- CO GMCs (left),
                HI envelopes (right)
    fig9        histograms of j_HI and j_GMC
    fig10       histograms of beta_rot for the HI envelopes and the CO GMCs

GMC-side quantities (Omega_GMC, Phi_GMC, j_GMC, M_GMC, R_GMC) are taken straight
from Rosolowsky et al. (2003) Table 2.  beta_rot is the uniform-sphere,
solid-body rotational energy fraction

    beta_rot = E_rot / |E_grav| = (1/5 M R^2 Omega^2) / (3/5 G M^2 / R)
             = Omega^2 R^3 / (3 G M)

computed separately for the two sides -- (Omega_HI, R_HI, M_HI) and
(Omega_GMC, R_GMC, M_GMC).

Output goes to ../paper/figures_rosolowsky2003/.  Run:

    /Users/thummimm/miniconda3/bin/python3 rosolowsky2003_figs.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # headless; must precede pyplot / mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u
from matplotlib.patches import Circle

import mekuria_imara_pipeline as mp
import rosolowsky2003_catalog as rc

FIGDIR = Path("../paper/figures_rosolowsky2003")
TAB2 = Path("../data/rosolowsky2003_tab2.csv")   # rosolowsky2003_catalog's own
                                                 # default path is a stale name
MI_TAB2 = Path("../data/mekuria_imara_tab2.csv")  # Mekuria & Imara 2026, Table 2
IBB_TAB1 = Path("../data/imara2011_tab1.csv")     # Imara, Bigiel & Blitz 2011,
                                                 # ApJ 732, 79, Table 1
OUT_CSV = "hi_table2_m33_rosolowsky2003.csv"

# Rosolowsky+2003 assumed D = 840 kpc; the pipeline uses 859.  Match the
# rescaling rosolowsky2003_catalog applies to its sizes and masses so the two
# sides of every ratio are on one distance scale (a 2.3% effect either way).
DFAC = rc.D_PIPE / rc.D_PUB

# 1 pc km/s expressed in 1e24 cm^2/s
PC_KMS_TO_1E24 = (1 * u.pc).to(u.cm).value * (1 * u.km).to(u.cm).value / 1e24


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
    """
    t = pd.read_csv(IBB_TAB1, comment="#")
    return pd.DataFrame(
        {
            "Omega_GMC_1e-2": t["Omega_GMC_1e-2"].to_numpy(dtype=float),
            "Omega_HI_1e-2": t["Omega_HI_1e-2"].to_numpy(dtype=float),
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


def analyse(cube, cat):
    """Run analyze_gmc over the catalog -> (Table-2 DataFrame, {name: GMCResult})."""
    rows, results = [], {}
    for _, gmc in cat.iterrows():
        res = mp.analyze_gmc(gmc, cube)
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
        rp = r.R_pc / cube.pc_per_pix
        ax.add_patch(Circle((r.xc, r.yc), rp, fill=False, ec="cyan",
                            lw=0.6, alpha=0.85))
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
    ax.set_title("M33 HI field -- Rosolowsky+2003 GMC envelopes (n={}), "
                 "HI and CO velocity gradients".format(len(results)), pad=12)
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

    fig.suptitle("HI vs CO GMC distributions -- this work (filled) vs "
                 "Mekuria & Imara 2026, Table 2 (step)", y=1.0)
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

    fig.suptitle("HI envelope vs Rosolowsky+2003 CO GMC -- per-cloud comparison",
                 y=1.0)
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

    fig.suptitle("Velocity gradient vs cloud radius", y=0.94)
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

    fit = mp._loglog_fit(R[keep], j[keep])
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
    axh.set_title("HI envelopes")
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
    ax.set_title("Specific angular momentum: HI envelopes vs CO GMCs")
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
    ax.set_ylabel(r"pipeline $R_{\rm HI}$  [pc]")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title(r"$R_{{\rm HI}}$ vs IBB extent  "
                 r"(median $R_{{\rm HI}}/{{\rm extent}}$ = {:.2f})".format(
                     np.median(ratio)))
    mp._style(ax)
    return fig


# ---------------------------------------------------------------------------
def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for old in sorted(FIGDIR.glob("fig*.pdf")):     # clear the previous numbering
        old.unlink()

    cat = rc.load(path=str(TAB2))
    gmc = load_gmc_side(cat)
    mi = load_mi_tab2()
    cube = mp.open_cube()

    tab, results = analyse(cube, cat)
    tab.to_csv(OUT_CSV, index=False)
    print("catalog n={}  kept n={}  -> {}".format(len(cat), len(tab), OUT_CSV))

    sigma, _ = mp.field_moments(cube, step=4)

    figs = {
        "fig1_field_gradients": fig1(cube, results, sigma, gmc),
        "fig2_spectra": fig2(results),
        "fig3_profiles": fig3(results),
        "fig4_planarity": fig4(tab),
        "fig5_quantity_distributions": fig5(tab, gmc, mi),
        "fig6_hi_gmc_ratios": fig6(tab, gmc),
        "fig7_omega_vs_size": fig7(tab, gmc),
        "fig8_j_vs_size": fig8(tab, gmc),
        "fig9_j_histograms": fig9(tab, gmc),
        "fig10_beta_rot_histograms": fig10(tab, gmc),
        "fig11_rhi_vs_ibb_extent": fig11(tab),
    }
    for stem, fig in figs.items():
        path = FIGDIR / "{}.pdf".format(stem)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("  wrote", path)


if __name__ == "__main__":
    main()
