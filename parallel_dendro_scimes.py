"""Feed a saved parallel_dendro decomposition into SCIMES's real clustering engine.

    python parallel_dendro_scimes.py <run> [<run> ...] [--isol]      run = a key of RUNS

--isol sets save_isol_leaves=True and writes <prefix>_scimes_isol_* instead of <prefix>_scimes_*.

Recovered from the 2026-09-25 session (transcript 574cf5c5), where it lived in a
scratchpad that was later cleared. FakeStructure / FakeDendrogram and the
volume/luminosity proxies are unchanged (including that session's sorted_leaves fix);
main() is new: it loads a saved {catalog.csv, index_map.fits} pair instead of computing
a dendrogram on a crop, and accumulates the subtree totals in channel chunks.

SCIMES's cloudstering() (scimes/scimes.py) is written against astrodendro's
Structure interface (.idx, .parent, .children, .is_leaf, .sorted_leaves(),
.height) and an astropy Table catalog. This does NOT reimplement SCIMES's
clustering logic (affinity matrices, spectral embedding, k-means, silhouette
search, cluster cleaning) -- it makes parallel_dendro's real tree look like an
astrodendro dendrogram, then calls scimes.scimes.cloudstering() itself, unmodified.

Two things astrodendro's ppv_catalog() normally provides are NOT reproduced
(would need WCS/beam/distance calibration): true physical volume (pc^3) and
luminosity (K km/s pc^2). Proxies are used instead:
  - "volume"     -> subtree pixel count (npix, including all descendants)
  - "luminosity" -> subtree-summed voxel intensity in K from the original cube
                    (including all descendants), no beam or pixel-area correction.
These are the two default SCIMES clustering criteria. Both proxies differ from the
physical quantities by constant factors only, so the clustering itself is unchanged
up to the automatically estimated scaling parameter.

Outputs (data/maps/<prefix>_scimes_*):
  _clouds.csv     one row per SCIMES cloud
  _lookup.csv     structure idx -> cloud idx (-1 = not in any cloud); together with
                  the index map this gives the assignment cube without storing a
                  6 GB copy (and avoids make_asgncube's int16 cast)
  _cloudmap.fits  2D map of the cloud idx projected along velocity (max), for plots
"""
import sys, time
sys.path.insert(0, "/Users/thummimm/Documents/research")
import numpy as np, pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from scimes.scimes import cloudstering

MAPS = "/Users/thummimm/Documents/research/data/maps/"
CUBE = MAPS + "m33_all_co21_8p0as.K.fits"
RUNS = {  # prefix, rms in the dendrogram's own units (sets SCIMES's S/N cut on branch heights)
    "r5":  ("m33_co21_8p0as_eroded_r5_parallel_dendro", 0.0497),    # K; median of the noise map
    "r10": ("m33_co21_8p0as_eroded_r10_parallel_dendro", 0.0497),
    "r15": ("m33_co21_8p0as_eroded_r15_parallel_dendro", 0.0497),
    "snr": ("m33_all_co21_8p0as_snr_parallel_dendro", 1.0),          # dendrogram is in sigma units
    "snr_r5":  ("m33_co21_8p0as_eroded_r5_snr_parallel_dendro", 1.0),
    "snr_r10": ("m33_co21_8p0as_eroded_r10_snr_parallel_dendro", 1.0),
    "snr_r15": ("m33_co21_8p0as_eroded_r15_snr_parallel_dendro", 1.0),
    "snr10d2": ("m33_all_co21_8p0as_snr10d2_parallel_dendro", 1.0),
    "snr5d2":  ("m33_all_co21_8p0as_snr5d2_parallel_dendro", 1.0),
    "snr4d2":  ("m33_all_co21_8p0as_snr4d2_parallel_dendro", 1.0),
    "snr2d2":  ("m33_all_co21_8p0as_snr2d2_parallel_dendro", 1.0),
}
MAX_LEAVES = 20000  # leaves in multi-leaf trees; each affinity matrix is 8*N^2 bytes (20k -> 3.2 GB)
CHUNK = 20


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class FakeStructure:
    """Duck-types just enough of astrodendro.structure.Structure for
    scimes.scimes.cloudstering() to walk the tree and compute affinities."""

    def __init__(self, entry, registry):
        self.idx = entry["idx"]
        self.is_leaf = entry["is_leaf"]
        self.is_branch = not entry["is_leaf"]
        self.vmin = entry["vmin"]
        self.vmax = entry["vmax"]
        self.npix_own = entry["npix"]
        self._children_ids = entry["children"]
        self._parent_id = entry["parent"]
        self._registry = registry  # idx -> FakeStructure, shared, filled by caller

    @property
    def parent(self):
        return self._registry[self._parent_id] if self._parent_id is not None else None

    @property
    def children(self):
        return [self._registry[c] for c in self._children_ids]

    @property
    def level(self):
        lvl, s = 0, self
        while s.parent is not None:
            lvl += 1
            s = s.parent
        return lvl

    @property
    def ancestor(self):
        s = self
        while s.parent is not None:
            s = s.parent
        return s

    @property
    def height(self):
        # Matches astrodendro's Structure.height exactly: the merge
        # threshold for a branch (min of its children's vmin), or its own
        # peak for a leaf.
        ch = self.children
        return min(c.vmin for c in ch) if ch else self.vmax

    def sorted_leaves(self):
        # SCIMES only reads .idx off the returned list; sort order doesn't
        # affect its logic, so this returns them in an arbitrary order via
        # an iterative subtree walk (no recursion-depth risk on deep trees).
        if self.is_leaf:
            return [self]
        out, stack = [], list(self.children)
        while stack:
            s = stack.pop()
            if s.is_leaf:
                out.append(s)
            else:
                stack.extend(s.children)
        return out

    def __repr__(self):
        return f"<FakeStructure idx={self.idx} {'leaf' if self.is_leaf else 'branch'}>"


class FakeDendrogram:
    """Duck-types the handful of astrodendro.Dendrogram accesses cloudstering()
    makes: dendrogram[idx] and dendrogram.leaves."""

    def __init__(self, stats):
        self._by_idx = {}
        for entry in stats:
            self._by_idx[entry["idx"]] = FakeStructure(entry, self._by_idx)
        self.leaves = [s for s in self._by_idx.values() if s.is_leaf]

    def __getitem__(self, idx):
        return self._by_idx[idx]

    def __len__(self):
        return len(self._by_idx)


def load_stats(catalog_csv):
    """parallel_dendro's saved catalog -> the stats-dict list compute() returns."""
    c = pd.read_csv(catalog_csv)
    return [dict(idx=int(r.idx), is_leaf=bool(r.is_leaf), vmin=float(r.vmin), vmax=float(r.vmax),
                 npix=int(r.npix), parent=None if r.parent == -1 else int(r.parent),
                 children=[] if pd.isna(r.children) else [int(x) for x in str(r.children).split()])
            for r in c.itertuples()]


def subtree_totals(stats, index_map, data):
    """Real subtree-inclusive npix and summed intensity per structure, from the
    index map (own-pixel labels, astrodendro's subtree=False convention) and the
    original cube, aggregated bottom-up through the real tree. Reads both in
    channel chunks so neither 1.5e9-voxel array is held in memory."""
    n = len(stats)
    own_npix = np.zeros(n, dtype=np.int64)
    own_flux = np.zeros(n)
    for i in range(0, index_map.shape[0], CHUNK):
        ix = np.asarray(index_map[i:i + CHUNK]).ravel()
        valid = ix >= 0
        own_npix += np.bincount(ix[valid], minlength=n)
        own_flux += np.bincount(ix[valid], weights=np.asarray(data[i:i + CHUNK], dtype=np.float64).ravel()[valid],
                                minlength=n)

    children = {e["idx"]: e["children"] for e in stats}
    npix_total = {}
    flux_total = {}

    def compute(i):
        if i in npix_total:
            return npix_total[i], flux_total[i]
        npx, flx = int(own_npix[i]), float(own_flux[i])
        for c in children[i]:
            cn, cf = compute(c)
            npx += cn
            flx += cf
        npix_total[i], flux_total[i] = npx, flx
        return npx, flx

    for e in stats:
        compute(e["idx"])
    return npix_total, flux_total


def run(name, isol=False):
    prefix, rms = RUNS[name]
    tag = "_scimes_isol" if isol else "_scimes"
    stats = load_stats(MAPS + prefix + "_catalog.csv")
    index_map = fits.open(MAPS + prefix + "_index_map.fits", memmap=True)[0].data
    data = fits.open(CUBE, memmap=True)[0].data
    log(f"{name}: {len(stats):,} structures")
    n_clust_leaves = sum(1 for e in stats if e["is_leaf"] and e["parent"] is not None)
    if n_clust_leaves > MAX_LEAVES:
        log(f"{name}: SKIPPED, {n_clust_leaves:,} leaves in multi-leaf trees > {MAX_LEAVES:,} "
            f"(~{8 * n_clust_leaves**2 / 1e9:.0f} GB per affinity matrix)")
        return

    npix_total, flux_total = subtree_totals(stats, index_map, data)

    # catalog rows MUST be in idx order 0..n-1: cloudstering() indexes
    # dendrogram[i] for i in range(len(catalog)), assuming row i == structure i.
    stats_by_idx = sorted(stats, key=lambda e: e["idx"])
    assert [e["idx"] for e in stats_by_idx] == list(range(len(stats_by_idx)))
    catalog = Table({
        "_idx": [e["idx"] for e in stats_by_idx],
        "volume": [npix_total[e["idx"]] for e in stats_by_idx],       # proxy: subtree npix
        "luminosity": [flux_total[e["idx"]] for e in stats_by_idx],   # proxy: subtree summed intensity [K]
        "vmin": [e["vmin"] for e in stats_by_idx],
        "vmax": [e["vmax"] for e in stats_by_idx],
        "is_leaf": [e["is_leaf"] for e in stats_by_idx],
    })

    dendrogram = FakeDendrogram(stats)
    log(f"{name}: adapter built, {len(dendrogram.leaves)} leaves; running cloudstering()")
    t0 = time.time()
    criteria = ["volume", "luminosity"]
    out = cloudstering(
        dendrogram, catalog, criteria,
        user_k=0, user_ams=None, user_scalpars=[None] * len(criteria), user_iter=1,
        save_isol_leaves=isol, save_clust_leaves=False, save_branches=False,
        blind=True, rms=rms, s2nlim=3, locscal=False,
    )
    if len(out) == 2:                     # <= 2 clusterable leaves: SCIMES returns (clusts, AMs) only
        clusts, silhouette, out_catalog = out[0], np.nan, None
    else:
        clusts, out_catalog, _, _, silhouette = out
    # cloudstering returns the silhouette of the best-k configuration *before* cluster cleaning;
    # the post-cleaning value is only printed by SCIMES ("Final clustering configuration silhoutte")
    log(f"{name}: {len(clusts)} clouds in {time.time() - t0:.1f}s, best-k silhouette (pre-cleaning)={silhouette:.3f}")

    # --- cloud catalog: one row per SCIMES cloud
    rows = []
    for c in clusts:
        s = dendrogram[c]
        rows.append(dict(cloud_idx=c,
                         structure_type=out_catalog["structure_type"][c] if out_catalog is not None else "",
                         n_leaves=len(s.sorted_leaves()), npix_subtree=npix_total[c],
                         flux_subtree_K=flux_total[c], vmin=s.vmin, vmax=s.vmax, height=s.height))
    pd.DataFrame(rows).to_csv(MAPS + prefix + tag + "_clouds.csv", index=False)

    # --- structure -> cloud lookup (a structure belongs to the cloud at or above it in its tree)
    cloud_of = np.full(len(stats), -1, dtype=np.int64)
    cset = set(int(c) for c in clusts)
    for e in stats_by_idx:
        s = dendrogram[e["idx"]]
        while s is not None and s.idx not in cset:
            s = s.parent
        cloud_of[e["idx"]] = -1 if s is None else s.idx
    pd.DataFrame(dict(idx=np.arange(len(stats)), cloud_idx=cloud_of)).to_csv(
        MAPS + prefix + tag + "_lookup.csv", index=False)

    # --- projected cloud map, for figures
    cmap2d = np.full(index_map.shape[1:], -1, dtype=np.int32)
    for i in range(0, index_map.shape[0], CHUNK):
        ix = np.asarray(index_map[i:i + CHUNK])
        cmap2d = np.maximum(cmap2d, np.where(ix >= 0, cloud_of[np.clip(ix, 0, None)], -1).max(axis=0))
    hdr = WCS(fits.getheader(CUBE)).celestial.to_header()
    hdr["BUNIT"] = "cloud_idx"
    fits.writeto(MAPS + prefix + tag + "_cloudmap.fits", cmap2d, hdr, overwrite=True)

    n_isol = sum(1 for e in stats if e["parent"] is None and e["is_leaf"])
    log(f"{name}: saved; {int((cloud_of >= 0).sum())} of {len(stats)} structures in clouds, "
        f"{n_isol} isolated single-leaf trunks {'kept' if isol else 'not kept'} (save_isol_leaves={isol})")


if __name__ == "__main__":
    isol = "--isol" in sys.argv
    for name in (a for a in sys.argv[1:] if a != "--isol"):
        run(name, isol)
