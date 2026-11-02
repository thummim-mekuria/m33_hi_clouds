import numpy as np, pandas as pd
from astropy.io import fits
from scipy.ndimage import distance_transform_edt
MAPS='/Users/thummimm/Documents/research/data/maps/'; P=MAPS+'m33_all_co21_8p0as_snr5d2_parallel_dendro'
hdr=fits.getheader(MAPS+'m33_all_co21_8p0as.K.fits'); sb=hdr['BMAJ']/abs(hdr['CDELT2'])/np.sqrt(8*np.log(2))
props=pd.read_csv(P+'_scimes_isol_cloudprops.csv').set_index('cloud_idx')
cl=pd.read_csv(P+'_scimes_clouds.csv').set_index('cloud_idx')
cat=pd.read_csv(P+'_catalog.csv').set_index('idx')
lk=pd.read_csv(P+'_scimes_lookup.csv').cloud_idx.values
ids=np.array(sorted(cl.index)); slot=np.full(lk.size,-1); slot[lk>=0]=np.searchsorted(ids,lk[lk>=0])
idx=fits.open(P+'_index_map.fits',memmap=True)[0].data; cube=fits.open(MAPS+'m33_all_co21_8p0as.K.fits',memmap=True)[0].data
n=len(ids); ny,nx=idx.shape[1:]; yy,xx=np.mgrid[0:ny,0:nx]
S={k:np.zeros(n) for k in ('w','x','y','xx','yy','xy','npix')}; vlo=np.full(n,1e9); vhi=np.full(n,-1.0)
for i in range(0,idx.shape[0],20):
    ix=np.asarray(idx[i:i+20]); m=ix>=0; c=slot[ix[m]]; k=c>=0; c=c[k]
    z,y,x=(a[k] for a in np.nonzero(m)); w=np.asarray(cube[i:i+20],dtype=np.float64)[m][k]
    for key,q in (('w',1),('x',x),('y',y),('xx',x*x),('yy',y*y),('xy',x*y)): S[key]+=np.bincount(c,weights=w*q,minlength=n)
    S['npix']+=np.bincount(c,minlength=n)
    np.minimum.at(vlo,c,z+i); np.maximum.at(vhi,c,z+i)
W=S['w']; mx,my=S['x']/W,S['y']/W
cxx,cyy,cxy=S['xx']/W-mx**2,S['yy']/W-my**2,S['xy']/W-mx*my
tr,det=cxx+cyy,cxx*cyy-cxy**2; d=np.sqrt(np.maximum(tr**2/4-det,0))
smaj,smin=np.sqrt(tr/2+d),np.sqrt(np.maximum(tr/2-d,0))
foot=fits.open('/Users/thummimm/Documents/research/M33/data/m33_co21_8p0as_eroded_masks.fits')['FOOT'].data.astype(bool)
edge=distance_transform_edt(foot)
df=pd.DataFrame(dict(smaj_beam=smaj/sb,smin_beam=smin/sb,axis_ratio=smaj/np.maximum(smin,1e-9),npix=S['npix'],
    nchan=vhi-vlo+1,edge_px=edge[np.round(my).astype(int),np.round(mx).astype(int)]),index=ids)
df=df.join(props[['R_pc','R_obs_pc','sigma_v_kms','M_Msun']]).join(cl[['n_leaves','structure_type']])
df['peak_snr']=cat.loc[ids,'vmax'].values
# also: did the unresolved ones fail on the minor axis only?
df['res']=df.R_pc.notna()
print("beam sigma = %.2f px; unresolved = minor or major axis <= beam sigma"%sb)
print("unresolved: major axis > beam:",int(((~df.res)&(df.smaj_beam>1)).sum()),"of",int((~df.res).sum()))
cols=['smaj_beam','smin_beam','axis_ratio','npix','nchan','n_leaves','peak_snr','R_obs_pc','sigma_v_kms','M_Msun','edge_px']
q=df.groupby('res')[cols].median().T; q.columns=['unresolved','resolved']; print(q.round(2))
print(df.groupby('res').structure_type.value_counts().unstack(fill_value=0))
print("single-leaf clouds:",df.groupby('res').apply(lambda t:int((t.n_leaves==1).sum())).to_dict())
print("flux share of clustered L: unresolved %.1f%%"%(100*df.M_Msun[~df.res].sum()/df.M_Msun.sum()))
for thr in (1.0,1.2,1.5):
    print(f"minor axis <= {thr} beam sigma:",int((df.smin_beam<=thr).sum()))
