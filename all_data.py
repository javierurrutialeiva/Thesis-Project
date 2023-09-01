from astropy.io import fits
from pixell import enmap
def all_data():
	f1 = enmap.read_map('ilc_SZ_yy_noKspaceCor.fits')
	f2 = fits.open('DR5_cluster-catalog_v1.1.fits')
	f3 = fits.open('y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22_lgt20_vl02_catalog.fit')
	return f1,f2,f3
