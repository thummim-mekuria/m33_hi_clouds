"""parallel_dendro on an eroded CO(2-1) cube: python run_pd_eroded.py <r> [<r> ...]

Generalises the 2026-10-09 run_pd_r15.py (same thresholds, same outputs) to any of the
saved ipynbs/m33_co21_8p0as_eroded_r{r}.npy cubes.
"""
import sys, time
sys.path.insert(0, '/Users/thummimm/Documents/research')
import numpy as np, pandas as pd
from astropy.io import fits
import parallel_dendro as pdn

HDR = '/Users/thummimm/Documents/research/data/maps/m33_all_co21_8p0as.K.fits'   # npy has no header
MIN_VALUE, MIN_DELTA, MIN_NPIX = 0.5, 0.3, 5

for r in sys.argv[1:]:
    SRC = f'/Users/thummimm/Documents/research/M33/ipynbs/m33_co21_8p0as_eroded_r{r}.npy'
    OUT = f'/Users/thummimm/Documents/research/data/maps/m33_co21_8p0as_eroded_r{r}_parallel_dendro'

    src = np.load(SRC, mmap_mode='r')
    data = np.empty(src.shape, dtype=np.float32)          # float32 copy, filled in chunks (6 GB, not 12)
    for i in range(0, src.shape[0], 20):
        data[i:i + 20] = src[i:i + 20]
    del src
    print(f'r{r}: loaded', data.shape, flush=True)

    t = time.time()
    index_map, stats = pdn.compute(data, min_value=MIN_VALUE, min_delta=MIN_DELTA, min_npix=MIN_NPIX)
    del data
    print(f'r{r}: {len(stats):,} structures in {time.time() - t:.0f} s', flush=True)

    rows = [dict(idx=s['idx'], is_leaf=s['is_leaf'], vmin=s['vmin'], vmax=s['vmax'], npix=s['npix'],
                 parent=-1 if s['parent'] is None else s['parent'], n_children=len(s['children']),
                 children=' '.join(map(str, s['children']))) for s in stats]
    pd.DataFrame(rows).to_csv(OUT + '_catalog.csv', index=False)

    hdr = fits.getheader(HDR); hdr['BUNIT'] = 'structure_id'
    fits.writeto(OUT + '_index_map.fits', index_map, hdr, overwrite=True)
    del index_map, stats, rows
    print(f'r{r}: saved', OUT, flush=True)
