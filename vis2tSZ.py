import numpy as np
import matplotlib.pyplot as plt
from pixell import utils,enmap
import pandas as pd
from astropy.io import fits
import os

szdata = 'ilc_SZ_yy_noKspaceCor.fits'
szmap = enmap.read_map(szdata)
path = '/data2/cristobal/actpol/cmblensing/des/data/redmapper/'
files = os.listdir(path)
visfit1 = fits.open(f'{path}/{files[0]}')
vitdata = visfit1[1].data
ra = np.array([vitdata[i][1] for i in range(len(vitdata))])
ra[ra>180] = ra[ra>180] - 180
dec = np.array([vitdata[i][2] for i in range(len(vitdata))])
indxs = np.random.randint(0,len(ra),5)
for i in indxs:
	ra0,dec0 = np.deg2rad([ra[i],dec[i]])
	width = np.deg2rad(1)
	box = [[dec0-width/2,ra0-width/2],[dec0+width/2,ra0+width/2]]
	submap = szmap.submap(box)
	fig = plt.figure()
	plt.imshow(submap,vmin=-1e-5,vmax=1e-6)
	fig.savefig(f'SZmap{np.round(ra0,2)}{np.round(dec0,2)}.png')
