import numpy as np
from astropy.io import fits
from scipy.ndimage import distance_transform_edt
MAPS='/Users/thummimm/Documents/research/data/maps/'
M33='/Users/thummimm/Documents/research/M33/'
C=25
cube=fits.open(MAPS+'m33_all_co21_8p0as.K.fits',memmap=True)[0].data
nz=cube.shape[0]
foot=np.zeros(cube.shape[1:],bool)
for i in range(0,nz,C): foot|=np.isfinite(np.asarray(cube[i:i+C])).any(0)
keep={}
for r in (5,10,15):
    e=np.load(M33+f'ipynbs/m33_co21_8p0as_eroded_r{r}.npy',mmap_mode='r'); k=np.zeros(foot.shape,bool)
    for i in range(0,nz,C): k|=np.isfinite(e[i:i+C]).any(0)
    keep[r]=k; print(f"r{r}: removes {np.count_nonzero(foot&~k):,} of {foot.sum():,} footprint px", flush=True)
dist=distance_transform_edt(foot)
sig=fits.getdata(MAPS+'m33_all_co21_8p0as_noise_map.fits')
print("footprint px with NaN sigma:",np.count_nonzero(foot&~np.isfinite(sig)), " median dist of NaN-sigma px:", np.median(dist[foot&~np.isfinite(sig)]))
bins=[0,2,5,10,15,30,70,1e9]
area=np.histogram(dist[foot],bins)[0]/foot.sum()
for tag in ('snr10d2','snr5d2','snr2d2'):
    idx=fits.open(MAPS+f'm33_all_co21_8p0as_{tag}_parallel_dendro_index_map.fits',memmap=True)[0].data
    cnt=np.zeros(foot.shape,np.int64); lost={r:set() for r in keep}; allids=set()
    for i in range(0,nz,C):
        ix=np.asarray(idx[i:i+C]); m=ix>=0; cnt+=m.sum(0)
        allids.update(np.unique(ix[m]).tolist())
        for r,k in keep.items(): lost[r].update(np.unique(ix[m & ~k[None]]).tolist())
    tot=cnt.sum()
    print(f"\n{tag}: {len(allids):,} structures, {tot:,} assigned voxels")
    for r,k in keep.items():
        v=cnt[~k].sum(); print(f"  r{r}: {v:,} voxels ({100*v/tot:.2f}%) and {len(lost[r]):,} structures ({100*len(lost[r])/len(allids):.1f}%) touch eroded px")
    vh=np.histogram(dist,bins,weights=cnt)[0]/tot
    print("  dist-to-edge px bin | share of footprint area | share of structure voxels | ratio")
    for a,b,x,y in zip(bins[:-1],bins[1:],area,vh): print(f"   {a:>4.0f}-{b:<5.0f} {x:7.3f} {y:7.4f} {y/x if x else 0:6.2f}")
