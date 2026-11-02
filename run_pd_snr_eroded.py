"""parallel_dendro on noise-weighted eroded CO cubes: python run_pd_snr_eroded.py <r> [<r> ...]

S/N = (ipynbs/m33_co21_8p0as_eroded_r{r}.npy) / sigma, with sigma from noise_map_co.py and
the same sigma-unit thresholds as run_pd_snr.py, so the only difference from that run is
the erosion. sigma is not smoothed (see run_pd_snr.py).
"""
import sys, time
sys.path.insert(0, '/Users/thummimm/Documents/research')
import numpy as np, pandas as pd
from astropy.io import fits
import parallel_dendro as pdn

CUBE = '/Users/thummimm/Documents/research/data/maps/m33_all_co21_8p0as.K.fits'   # npy has no header
NOISE = '/Users/thummimm/Documents/research/data/maps/m33_all_co21_8p0as_noise_map.fits'
MIN_VALUE, MIN_DELTA, MIN_NPIX = 10.0, 6.0, 5          # sigma units

sigma = fits.getdata(NOISE).astype(np.float32)
for r in sys.argv[1:]:
    SRC = f'/Users/thummimm/Documents/research/M33/ipynbs/m33_co21_8p0as_eroded_r{r}.npy'
    OUT = f'/Users/thummimm/Documents/research/data/maps/m33_co21_8p0as_eroded_r{r}_snr_parallel_dendro'

    src = np.load(SRC, mmap_mode='r')
    data = np.empty(src.shape, dtype=np.float32)       # float32 S/N cube, filled in chunks (6 GB)
    for i in range(0, src.shape[0], 20):
        data[i:i + 20] = np.asarray(src[i:i + 20], dtype=np.float32) / sigma
    del src
    print(f'r{r}: loaded', data.shape, 'voxels > min_value:', int(np.count_nonzero(data > MIN_VALUE)), flush=True)

    t = time.time()
    index_map, stats = pdn.compute(data, min_value=MIN_VALUE, min_delta=MIN_DELTA, min_npix=MIN_NPIX)
    del data
    print(f'r{r}: {len(stats):,} structures in {time.time() - t:.0f} s', flush=True)

    rows = [dict(idx=s['idx'], is_leaf=s['is_leaf'], vmin=s['vmin'], vmax=s['vmax'], npix=s['npix'],
                 parent=-1 if s['parent'] is None else s['parent'], n_children=len(s['children']),
                 children=' '.join(map(str, s['children']))) for s in stats]
    pd.DataFrame(rows).to_csv(OUT + '_catalog.csv', index=False)

    hdr = fits.getheader(CUBE)
    hdr['BUNIT'] = 'structure_id'
    hdr['COMMENT'] = f'parallel_dendro on eroded_r{r}/sigma; min_value=10, min_delta=6 (sigma), min_npix=5'
    fits.writeto(OUT + '_index_map.fits', index_map, hdr, overwrite=True)
    del index_map, stats, rows
    print(f'r{r}: saved', OUT, flush=True)
