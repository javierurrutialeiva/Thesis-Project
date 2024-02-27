import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import astropy
from astropy.cosmology import WMAP9 as cosmo
from astropy.io import fits
from pixell import enmap,utils 
import pandas as pd
import astropy.units as u
from scipy.spatial import KDTree as KD
import matplotlib
from scipy.signal import convolve2d,fftconvolve

def extract_clusters(scale = 1,redshift = 0.3,r0 = 1e-3,FWHM_correction = False,FWHM = 1.5,match = True,skip = 1):
	if match==True:
		redmapper = fits.open('/data2/javierurrutia/szeffect/data/y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22_lgt20_vl02_catalog.fit')[1].data
		szmap = enmap.read_map('/data2/javierurrutia/szeffect/data/ilc_SZ_yy_noKspaceCor.fits')
		szcat = fits.open('/data2/javierurrutia/szeffect/data/DR5_cluster-catalog_v1.1.fits')[1].data
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
		distances2,index2 = KD(p2).query(p1)
		ind = np.unique(index[distances<r0])
		ind2 = np.unique(index2[distances2<r0])
		ra,dec = rmRA[ind],rmDec[ind]
		nszcat = szcat[ind2]
		nrmcat = redmapper[ind]
		SZsubmaps = []
		radii = []
		width = np.deg2rad(scale)
		for i in tqdm(range(len(ra))):
			box = [[dec[i]-width/2.,ra[i]-width/2.],[dec[i]+width/2.,ra[i]+width/2.]]
			smap = szmap.submap(box)
			boxsize = enmap.box(smap.shape,smap.wcs)/utils.degree
			deci,decf,rai,raf = boxsize[0][0],boxsize[1][0],boxsize[0][1],boxsize[1][1]
			adec,ara = np.linspace(deci,decf,smap.shape[0]),np.linspace(rai,raf,smap.shape[1])
			adec,ara = np.meshgrid(adec,ara)
			adec,ara = np.deg2rad(adec),np.deg2rad(ara)
			cdec,cra = (decf + deci)/2,(raf + rai)/2
			cdec,cra = np.deg2rad(cdec),np.deg2rad(cra)
			theta = np.arccos(np.sin(cdec)*np.sin(adec) + np.cos(cdec)*np.cos(adec)*np.cos(cra - ara))
			distance = cosmo.angular_diameter_distance(nrmcat[i]['Z'])
			R = (distance*theta).to(u.kpc)
			radii.append(R)
			if FWHM_correction==True:
				x,y = np.indices(np.shape(smap))
				x0 = y0 = np.shape(smap)[0]//2
				r = np.sqrt((x-x0)**2 + (y-y0)**2)
				px2min = scale/np.shape(smap)[0]*60
				FWHMpx = FWHM/px2min
				sigmapx = FWHMpx/(2*np.sqrt(2*np.log(2)))
				gauss = np.exp(-r**2/(2*FWHMpx**2))
				smap = fftconvolve(smap,gauss,'same')
			SZsubmaps.append(smap)
		return SZsubmaps,nrmcat,nszcat,ind,radii
	elif match==False:
		redmapper = fits.open('/data2/javierurrutia/szeffect/data/y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22_lgt20_vl02_catalog.fit')[1].data
		szmap = enmap.read_map('/data2/javierurrutia/szeffect/data/ilc_SZ_yy_noKspaceCor.fits')
		if type(redshift)==list:
			redmapper = redmapper[(redmapper['Z']>redshift[0]) & (redmapper['Z']<redshift[1])]
		elif type(redshift)==float or type(redshift)==int:
			redmapper = redmapper[(redmapper['Z']>redshift)]
		rmRA, rmDEC = np.array(redmapper['RA']),np.deg2rad(redmapper['DEC'])
		rmRA[rmRA > 180] = rmRA[rmRA > 180] - 360
		ra,dec = np.deg2rad(rmRA),rmDEC
		SZsubmaps = []
		radii = []
		width = np.deg2rad(scale)
		for i in tqdm(range(0,len(ra),skip)):
			box = [[dec[i]-width/2.,ra[i]-width/2.],[dec[i]+width/2.,ra[i]+width/2.]]
			smap = szmap.submap(box)
			boxsize = enmap.box(smap.shape,smap.wcs)/utils.degree
			deci,decf,rai,raf = boxsize[0][0],boxsize[1][0],boxsize[0][1],boxsize[1][1]
			adec,ara = np.linspace(deci,decf,smap.shape[0]),np.linspace(rai,raf,smap.shape[1])
			adec,ara = np.meshgrid(adec,ara)
			adec,ara = np.deg2rad(adec),np.deg2rad(ara)
			cdec,cra = (decf + deci)/2,(raf + rai)/2
			cdec,cra = np.deg2rad(cdec),np.deg2rad(cra)
			theta = np.arccos(np.sin(cdec)*np.sin(adec) + np.cos(cdec)*np.cos(adec)*np.cos(cra - ara))
			distance = cosmo.angular_diameter_distance(redmapper[i]['Z'])
			R = (distance*theta).to(u.kpc)
			radii.append(R)
			if FWHM_correction==True:
				x,y = np.indices(np.shape(smap))
				x0 = y0 = np.shape(smap)[0]//2
				r = np.sqrt((x-x0)**2 + (y-y0)**2)
				px2min = scale/np.shape(smap)[0]*60
				FWHMpx = FWHM/px2min
				sigmapx = FWHMpx/(2*np.sqrt(2*np.log(2)))
				gauss = np.exp(-r**2/(2*FWHMpx**2))
				smap = fftconvolve(smap,gauss,'same')
			SZsubmaps.append(smap)
		return SZsubmaps,redmapper,[],[],radii

def rings(r_rings,radii,submap,z=1e-5,center=[29,29],pixel_size=1e-5,title='rings.png',save=False,method_func=np.median,alpha0=0.1,color='red',ID='0'):
	circles = []
	results = []
	sigma = []
	brighter_pixel = np.max(np.array(submap)[(radii<r_rings[2])])
	indx_brighter_pixel = np.where(np.array(submap)==brighter_pixel)
	for i in range(len(r_rings)-1):
		dd = np.array(submap)[(radii<r_rings[i+1]) & (radii>r_rings[i])]
		results.append(method_func(dd))
		sigma.append(np.std(dd)/np.sqrt(dd.size))
		r_degrees = radii/cosmo.angular_diameter_distance(z).to(u.kpc)
		r_degrees = np.rad2deg(r_degrees.value)*60
		r1 = r_degrees[0,0]/2
		r2 = -r1
		r3 = r_degrees[-1,0]/2
		r4 = -r3
		if save==True:
			r_circle = np.rad2deg((r_rings[i+1]/cosmo.angular_diameter_distance(z).to(u.kpc)).value)*60
			circles.append(plt.Circle((0,0),r_circle,alpha=(alpha0-i*(alpha0/len(r_rings))),facecolor=color,edgecolor='black'))
			circles.append(plt.Circle((0,0),r_circle,edgecolor='red',fill=False))
	if save==True:
		fig,(ax1,ax2) = plt.subplots(1,2,figsize=(6,3))
		ax2.imshow(submap,origin='lower',extent=(r1,r2,r3,r4))
		ax2.set(xlabel='(arcmin)',ylabel='(arcmin)')
		ax2.set_aspect('auto')
		cord_x = np.interp(indx_brighter_pixel[1], (0, submap.shape[1]), (r1, r2))
		cord_y = np.interp(indx_brighter_pixel[0], (0, submap.shape[0]), (r3, r4))
		ax2.scatter(cord_x,cord_y,marker='>',color='None',edgecolors='red',s=40)
		ax2.scatter(0,0,marker='o',color='None',edgecolors='snow',s=40)
		[plt.gca().add_artist(c) for c in circles]
		ax1.errorbar(r_rings[0:len(r_rings)-1].value,results,yerr=sigma,ecolor='black',color='red',capsize=1)
		ax1.grid(True)
		ax1.set(yscale='log',xlabel='radio (kpc)',ylabel='$y_{sz}$')
	#	fig.suptitle(f'Cúmulo redMaPPer ID={ID}',fontsize=20)
		fig.tight_layout()
		fig.savefig(title)
		plt.close()
	return results,sigma
