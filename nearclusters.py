from all_data import all_data
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import astropy 
from astropy.io import fits
from pixell import enmap 
import pandas as pd
from scipy.spatial import KDTree as KD




def extract_clusters(scale=1,redshift=0.3,r0=1e-3):
	redmapper = fits.open('/data2/javierurrutia/data/y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22_lgt20_vl02_catalog.fit')[1].data
	szmap = enmap.read_map('/data2/javierurrutia/data/ilc_SZ_yy_noKspaceCor.fits')
	szcat = fits.open('/data2/javierurrutia/data/DR5_cluster-catalog_v1.1.fits')[1].data
	if type(redshift)==list:
		szcat = szcat[(szcat['redshift']>redshift[0]) & (szcat['redshift']<redshift[1])]
		redmapper = redmapper[(redmapper['Z']>redshift[0]) & (redmapper['Z']<redshift[1])]
	elif type(redshift)==float:
		szcat = szcat[(szcat['redshift']>redshift)]
		redmapper = redmapper[(redmapper['Z']>redshift)]
	szRA,szDec = np.array(szcat['RADeg']),np.deg2rad(szcat['decDeg'])
	szRA[szRA > 180] = szRA[szRA > 180] - 360
	szRA = np.deg2rad(szRA)
	rmRA,rmDec = np.array(redmapper['RA']),np.deg2rad(redmapper['DEC'])
	rmRA[rmRA > 180] = rmRA[rmRA > 180] - 360
	rmRA = np.deg2rad(rmRA)
	p1 = np.array([[rmRA[i],rmDec[i]] for i in range(len(rmRA))])
	p2 = np.array([[szRA[i],szDec[i]] for i in range(len(szRA))])
	distances,index = KD(p1).query(p2)
	ind = np.unique(index[distances<r0])
	ra,dec = rmRA[ind],rmDec[ind]
	nszcat = []#szcat[ind]
	nrmcat = redmapper[ind]
	SZsubmaps = []
	width = np.deg2rad(scale)
	for i in tqdm(range(len(ra))):
                box = [[dec[i]-width/2.,ra[i]-width/2.],[dec[i]+width/2.,ra[i]+width/2.]]               
                smap = szmap.submap(box)
                SZsubmaps.append(smap)
	return SZsubmaps,nrmcat,nszcat,ind

def rings(r_rings,submap,center=[29,29],pixel_size=1e-5,title='rings.png',save=False,method_func=np.median,alpha0=0.5,color='red'):
	x,y = np.indices(submap.shape)
	xpx,ypx = x*pixel_size,y*pixel_size
	xpx,ypx = (xpx-xpx[center[0],center[1]],ypx-ypx[center[0],center[1]])
	rpx = np.sqrt(xpx**2 + ypx**2)
	circles = []
	results = []
	for i in range(len(r_rings)-1):
		dd = np.array(submap)[(rpx<r_rings[i+1]) & (rpx>r_rings[i])]
		results.append(method_func(dd))
		if save==True:
			circles.append(plt.Circle((0,0),r_rings[i],alpha=(alpha0-i*(alpha0/len(r_rings))),color=color,edgecolor='black'))
			fig,(ax1,ax2) = plt.subplots(1,2)
			ax2.imshow(submap,extent=(((0-29)*pixel_size,(60-29)*pixel_size,(0 -29)*pixel_size,(60-29)*pixel_size)))
	if save==True:
		[plt.gca().add_artist(c) for c in circles]
		ax1.plot(r_rings[0:len(r_rings)-1],results)
		ax1.set(title='$Y_{sz}$ in function of ring radius',xlabel='radius',ylabel='$Y_{sz}$')
		fig.savefig(title)
	return results
