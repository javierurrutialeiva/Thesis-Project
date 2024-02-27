from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os

class radio_source():
	def __init__(self,RA,DEC,sources_catalog=None,hosts_catalog=None,DRAGN=False,DRAGN_data=None,method='basic',fits_file=False):
		ra_catalog = hosts_catalog['RA_Source']
		dec_catalog = hosts_catalog['DEC_Source']
		if method != 'skycoords':
			distances = np.sqrt((ra_catalog - RA)**2 + (dec_catalog - DEC)**2)
			indx = np.argmin(distances)
			radio_source = hosts_catalog.iloc[indx]
		elif method == 'skycoords':
			pass #for now
		self.source = radio_source
		self.name = radio_source.Source_name
		self.ra = radio_source.RA_Source
		self.dec = radio_source.DEC_Source
		self.total_flux = radio_source.Total_flux_source
		self.angular_size = radio_source.Angular_size
		self.component_names = np.array([radio_source['Component_name_1'],radio_source['Component_name_2'],radio_source['Component_name_3']])
		self.peak_ratios = np.array([radio_source['Peak_flux_ratio_12'],radio_source['Peak_flux_ratio_13'],radio_source['Peak_flux_ratio_23']])
		self.components = np.array([sources_catalog[sources_catalog['Component_name']==self.component_names[0]]
						,sources_catalog[sources_catalog['Component_name']==self.component_names[1]]
						,sources_catalog[sources_catalog['Component_name']==self.component_names[2]]])
		if fits_file == True:
			with open(os.devnull, 'w') as fnull:
				self.image = fits.open(radio_source['VLASS_cutout_url'])

		if DRAGN == True:
			ra_dragn = DRAGN_data['RA']
			dec_dragn = DRAGN_data['DEC']
			distances = np.sqrt((self.ra - ra_dragn)**2 + (self.dec - dec_dragn)**2)
			indx = np.argmin(distances)
			if distances[indx] > 0.05:
				self.dragn = None
			else:
				agn = DRAGN_data[indx]
				self.dragn = agn

	def download_fits(self,save=False):
		with open(os.devnull, 'w') as fnull:
			self.image = fits.open(self.source['VLASS_cutout_url'])[0].data
		if save==True:
			fig = plt.figure(figsize=(8,8))
			ax = plt.axes()
			ax.imshow(self.image,origin='lower',cmap='gray')
			ax.set(title=self.name)
			fig.savefig(f'{self.name}.png')
			plt.close()
