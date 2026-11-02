"""Rosolowsky et al. (2007) Table 3 -> the column schema mekuria_imara_pipeline
expects (Corbelli et al. 2017 Table 5).

The pipeline only reads seven catalog columns -- N, RAdeg, DEdeg, R, re, MH2
and sigmaC -- so this module rewrites the Rosolowsky CO catalog into exactly
those, and nothing else is touched.

Mapping
    N       <- M33GMC
    RAdeg   <- RAJ2000  (sexagesimal -> deg)
    DEdeg   <- DEJ2000  (sexagesimal -> deg)
    R       <- Rgal                                             [kpc]
    re      <- sqrt(Amaj * Amin) / 2, the geometric-mean semi-axis of the
               nondeconvolved cloud, which is the closest analogue of
               Corbelli's effective radius that Table 3 carries        [pc]
    MH2     <- Mass * 1e5                                       [Msun]
    sigmaC  <- sigma, the CO velocity dispersion               [km/s]

Rosolowsky+2007 assumed D = 850 kpc; the pipeline uses 859 kpc. Sizes are
rescaled by 859/850 and masses by (859/850)^2 so the two catalogs sit on one
distance scale. Set rescale=False to keep the published numbers.
"""
from pathlib import Path

import numpy as np
import pandas as pd

TAB3 = Path(__file__).resolve().parent.parent / "data" / "rosolwsky2007_tab3.csv"
D_PUB, D_PIPE = 850.0, 859.0          # kpc


def _sex_to_deg(col, hours):
    parts = col.str.strip().str.split(r"[:\s]+", regex=True, expand=True).astype(float)
    sign = np.where(col.str.strip().str.startswith("-"), -1.0, 1.0)
    mag = parts[0].abs() + parts[1] / 60.0 + parts[2] / 3600.0
    return sign * mag * (15.0 if hours else 1.0)


def load(path=None, rescale=True):
    df = pd.read_csv(path or TAB3, comment="#")
    f = D_PIPE / D_PUB if rescale else 1.0
    return pd.DataFrame({
        "N": df["M33GMC"].astype(int),
        "RAdeg": _sex_to_deg(df["RAJ2000"], hours=True),
        "DEdeg": _sex_to_deg(df["DEJ2000"], hours=False),
        "R": df["Rgal"] * f,
        "re": 0.5 * np.sqrt(df["Amaj"] * df["Amin"]) * f,
        "MH2": df["Mass"] * 1e5 * f**2,
        "sigmaC": df["sigma"],
    })


if __name__ == "__main__":
    cat = load()
    out = Path(__file__).resolve().with_name("rosolowsky_tab3_cat.txt")
    cat.to_string(out, index=False)
    print(cat.describe().to_string())
    print(f"\nn={len(cat)}  ->  {out}")
