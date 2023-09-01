import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from pixell import enmap, utils, enplot
import pandas as pd
import os
import subprocess as sp
import sys
from tqdm import tqdm
from matplotlib.colors import LogNorm

# nsamples/coords filename scale dpi mode

args = sys.argv
filename = args[2]
scale = args[3]
dpi = int(args[4])
mode = str(args[5])

def cc2tSZ(catalog,SZmap,nsamples=20,coord=[None,None]):
	if coord[0]==None and coord[1]==None:
		cc = fits.open(catalog)
		cc = pd.DataFrame(cc[1].data)
		RA = cc['RADeg'].to_numpy()
		RA[RA > 180] = RA[RA > 180] - 360
		cc['RADeg'] = RA
		SZ = enmap.read_map(SZmap)
		if nsamples < len(cc):
			sample = cc.sample(n=nsamples)
		else:
			sample = cc
		dec,ra = sample['decDeg'].to_numpy(),sample['RADeg'].to_numpy()
		dec,ra = np.deg2rad(dec),np.deg2rad(ra)
		width = np.deg2rad(float(scale))
		m,s = np.mean(SZ),np.std(SZ)
#	dir_name = '/home/javierurrutia/plots/'
#	print(os.path.dirname(dir_name))
#	plt.rcParams['savefig.directory'] = os.path.dirname(dir_name)
		for i in tqdm(range(len(dec))):
			box = [[dec[i]-width/2.,ra[i]-width/2.],[dec[i]+width/2.,ra[i]+width/2.]]
			smap = SZ.submap(box)
			print(box,end='\n')
			if np.mean(smap)==0:
				continue
			box = np.rad2deg(box)
			im = plt.imshow(smap,vmin=-1e-6,vmax=1e-5,
				extent=[box[0][1],box[1][1],box[0][0],box[1][0]],cmap='magma')
			plt.colorbar(im)
			plt.savefig(f'{sample.iloc[i]["name"]}.png',dpi=dpi)
			plt.close()
	else:
		SZ = enmap.read_map(SZmap)
		ra,dec = coord
		ra,dec = np.deg2rad(ra),np.deg2rad(dec)
		width = np.deg2rad(float(scale))
		box = [[dec-width/2.,ra-width/2.],[dec+width/2.,ra+width/2.]]
		smap = SZ.submap(box)
		if np.mean(smap)==0:
			return None
		box = np.rad2deg(box)
		plt.imshow(smap,vmax=2e-5,vmin=-1e-5,extent=[box[0][1],box[1][1],box[0][0],box[1][0]],cmap='magma')
		plt.savefig(f'sample_{ra}+{dec}.png',dpi=dpi)

cluster_catalog = 'DR5_cluster-catalog_v1.1.fits'
tSZ_map = 'ilc_SZ_yy_noKspaceCor.fits'

if mode=='catalog':
	nsamples = int(args[1])
	cc2tSZ(cluster_catalog,tSZ_map,nsamples,[None,None])
	sp.run(f'sh create_tar.sh {filename}',shell=True)
elif mode=='single':
	coord = np.array(list(args[1].replace('[','').replace(']','').split(','))).astype(float)
	cc2tSZ(cluster_catalog,tSZ_map,None,coord)

