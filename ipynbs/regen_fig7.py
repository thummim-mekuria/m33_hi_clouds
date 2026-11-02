"""Rebuild only fig7 (Omega vs cloud radius, now radius on x / Omega on y).

Reuses the cached HI table written by the last full run of
rosolowsky2003_figs.main() (OUT_CSV) so the cube fit is not repeated, and
leaves every other fig*.pdf in FIGDIR untouched.

    /Users/thummimm/miniconda3/bin/python3 regen_fig7.py
"""
import pandas as pd

import rosolowsky2003_figs as rf

tab = pd.read_csv(rf.OUT_CSV)
cat = rf.rc.load(path=str(rf.TAB2))
gmc = rf.load_gmc_side(cat)

fig = rf.fig7(tab, gmc)
out = rf.FIGDIR / "fig7_omega_vs_size.pdf"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
