"""Build only fig11 (pipeline R_HI vs IBB 2011 HI gradient extent, 1:1 dashed).

Reuses the cached HI table (OUT_CSV) from the last full run so the cube fit is
not repeated, and leaves every other fig*.pdf in FIGDIR untouched.

    /Users/thummimm/miniconda3/bin/python3 regen_fig11.py
"""
import pandas as pd

import rosolowsky2003_figs as rf

tab = pd.read_csv(rf.OUT_CSV)

fig = rf.fig11(tab)
out = rf.FIGDIR / "fig11_rhi_vs_ibb_extent.pdf"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
