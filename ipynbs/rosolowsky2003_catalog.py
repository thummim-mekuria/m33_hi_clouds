"""Rosolowsky et al. (2003) Table 2 -> the column schema mekuria_imara_pipeline
expects (Corbelli et al. 2017 Table 5).

Companion to rosolowsky_catalog.py, which does the same job for the 2007
catalog. The 2003 sample is 45 BIMA GMCs at 20 pc resolution -- much smaller
than either of the other two catalogs, but it is the one IBB itself used, so it
is the like-for-like comparison with that paper's Figure 21.

Mapping
    N       <- Cloud
    RAdeg   <- centre + dRA_arcsec / cos(dec)                  [deg]
    DEdeg   <- centre + dDec_arcsec                            [deg]
    R       <- deprojected galactocentric radius, computed here since
               Table 2 does not tabulate one                   [kpc]
    re      <- r_pc, the resolved cloud radius. 9 of the 45 clouds are
               unresolved and get the 10 pc projected beam radius the paper
               quotes as the limit -- an upper bound, not a measurement.
               Only used for overlay circles in the maps, never in the
               analysis, so the substitution cannot bias any result.  [pc]
    MH2     <- MCO_1e4 * 1e4                                    [Msun]
    sigmaC  <- VFWHM / (2 sqrt(2 ln 2))                         [km/s]

Rosolowsky+2003 assumed D = 840 kpc; the pipeline uses 859. Sizes are rescaled
by 859/840 and masses by (859/840)^2. Set rescale=False for published values.
"""
from pathlib import Path

import numpy as np
import pandas as pd

TAB2 = Path(__file__).resolve().parent.parent / "data" / "rosolowsky_2003.csv"

# galactic centre the offsets are measured from (Table 2 note a)
RA0 = (1 + 33 / 60 + 50.98 / 3600) * 15.0          # 01h33m50.98s -> deg
DE0 = 30 + 39 / 60 + 36.7 / 3600                   # +30d39'36.7"
BEAM_R_PC = 10.0                                   # projected beam radius
D_PUB, D_PIPE = 840.0, 859.0                       # kpc
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def load(path=None, rescale=True, resolved_only=False):
    df = pd.read_csv(path or TAB2, comment="#")
    f = D_PIPE / D_PUB if rescale else 1.0

    ra = RA0 + df["dRA_arcsec"] / 3600.0 / np.cos(np.deg2rad(DE0))
    dec = DE0 + df["dDec_arcsec"] / 3600.0

    import mekuria_imara_pipeline as mp
    R = np.hypot(*mp.deproject_points(ra.to_numpy(), dec.to_numpy()))

    cat = pd.DataFrame({
        "N": df["Cloud"].astype(int),
        "RAdeg": ra,
        "DEdeg": dec,
        "R": R,
        "re": df["r_pc"].fillna(BEAM_R_PC) * f,
        "MH2": df["MCO_1e4"] * 1e4 * f**2,
        "sigmaC": df["VFWHM"] * FWHM_TO_SIGMA,
        "resolved": df["r_pc"].notna(),
    })
    return cat[cat["resolved"]].reset_index(drop=True) if resolved_only else cat


if __name__ == "__main__":
    cat = load()
    out = Path(__file__).resolve().with_name("rosolowsky2003_cat.txt")
    cat.drop(columns="resolved").to_string(out, index=False)
    print(cat.describe().to_string())
    print("\nresolved: {}/{}  ->  {}".format(
        int(cat["resolved"].sum()), len(cat), out))
