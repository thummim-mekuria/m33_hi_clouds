"""parallel_dendro on the CO(2-1) cube weighted by its per-pixel noise map (S/N = T / sigma).

Same structure as the 2026-10-09 run_pd_r15.py, but on the full (un-eroded) cube divided
by noise_map_co.py's sigma map, so the threshold follows the local noise instead of
blanking a fixed 15 px rim. Thresholds are in sigma units, matched to the K run at the
median sigma (0.05 K): 0.5 K -> 10 sigma, 0.3 K -> 6 sigma. sigma is not smoothed: a
median filter at a tile rim mixes in quieter interior pixels and would raise S/N there.
"""
import sys, time
sys.path.insert(0, '/Users/thummimm/Documents/research')
import numpy as np, pandas as pd
from astropy.io import fits
import parallel_dendro as pdn

CUBE = '/Users/thummimm/Documents/research/data/maps/m33_all_co21_8p0as.K.fits'
NOISE = '/Users/thummimm/Documents/research/data/maps/m33_all_co21_8p0as_noise_map.fits'
OUT = '/Users/thummimm/Documents/research/data/maps/m33_all_co21_8p0as_snr_parallel_dendro'
MIN_VALUE, MIN_DELTA, MIN_NPIX = 10.0, 6.0, 5          # sigma units
INVERT = '--invert' in sys.argv                        # run on -T/sigma: every structure found is noise
args = [a for a in sys.argv[1:] if a != '--invert']
if len(args) == 2:                                     # python run_pd_snr.py <min_value> <min_delta> [--invert]
    MIN_VALUE, MIN_DELTA = float(args[0]), float(args[1])
    OUT = OUT.replace('_snr_', f'_snr{args[0]}d{args[1]}_')
if INVERT:
    OUT = OUT.replace('_snr', '_negsnr')

sigma = fits.getdata(NOISE).astype(np.float32)
print(OUT, f'min_value={MIN_VALUE:g} min_delta={MIN_DELTA:g} min_npix={MIN_NPIX}', flush=True)
print('median sigma [K]: %.4f' % np.nanmedian(sigma), flush=True)

with fits.open(CUBE, memmap=True) as h:
    src = h[0].data
    hdr = h[0].header.copy()
    data = np.empty(src.shape, dtype=np.float32)       # float32 S/N cube, filled in chunks (6 GB)
    for i in range(0, src.shape[0], 20):
        data[i:i + 20] = np.asarray(src[i:i + 20], dtype=np.float32) / sigma   # NaN sigma -> NaN S/N
del sigma
if INVERT:
    np.negative(data, out=data)
print('loaded', data.shape, 'voxels > min_value:', int(np.count_nonzero(data > MIN_VALUE)), flush=True)

t = time.time()
index_map, stats = pdn.compute(data, min_value=MIN_VALUE, min_delta=MIN_DELTA, min_npix=MIN_NPIX)
print(f'{len(stats):,} structures in {time.time() - t:.0f} s', flush=True)

rows = [dict(idx=s['idx'], is_leaf=s['is_leaf'], vmin=s['vmin'], vmax=s['vmax'], npix=s['npix'],
             parent=-1 if s['parent'] is None else s['parent'], n_children=len(s['children']),
             children=' '.join(map(str, s['children']))) for s in stats]
pd.DataFrame(rows).to_csv(OUT + '_catalog.csv', index=False)

hdr['BUNIT'] = 'structure_id'
hdr['COMMENT'] = f'parallel_dendro on {"-" if INVERT else ""}T/sigma; min_value={MIN_VALUE:g}, min_delta={MIN_DELTA:g} (sigma), min_npix={MIN_NPIX}'
fits.writeto(OUT + '_index_map.fits', index_map, hdr, overwrite=True)
print('saved', OUT, flush=True)
