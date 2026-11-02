"""Re-run the Mekuria & Imara figure set on the Rosolowsky+2007 CO catalog.

Same analysis and same plots as the Corbelli run -- only the GMC catalog is
swapped -- so the two figure sets can be compared panel by panel.

    python run_rosolowsky.py [--nmax N] [--figdir DIR] [--out CSV]
"""
import argparse
from pathlib import Path

import mekuria_imara_pipeline as mp
import rosolowsky_catalog

FIGDIR = "../paper/figures_rosolowsky"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=None)
    ap.add_argument("--snr", type=float, default=5.0)
    ap.add_argument("--out", default="hi_table2_m33_rosolowsky.csv")
    ap.add_argument("--figdir", default=FIGDIR)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    cat = rosolowsky_catalog.load()
    if args.nmax:
        cat = cat.iloc[:args.nmax]
    cube = mp.open_cube()

    rows = []
    for _, gmc in cat.iterrows():
        res = mp.analyze_gmc(gmc, cube)
        if res is not None:
            rows.append(mp.result_row(res))

    tab = mp.pd.DataFrame(rows)
    tab.to_csv(args.out, index=False)
    print("catalog n={}  kept n={}  -> {}".format(len(cat), len(tab), args.out))

    if not args.no_figures:
        written = mp.make_figures(tab, cube=cube, cat=cat, outdir=args.figdir)
        for p in written:
            print("  wrote", p)
    return tab


if __name__ == "__main__":
    main()
