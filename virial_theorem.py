import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from nearclusters import rings

def radials_profiles(szmap,redmapper,r_rings,pixel_size,save=False,method_func=np.median):
	l = []
	rr = []
	for i in tqdm(range(len(szmap))):
		lamda = redmapper[i]['LAMBDA_CHISQ']
		l.append(lamda)
		ID = redmapper[i]['MEM_MATCH_ID']
		r = rings(r_rings,szmap[i],pixel_size=pixel_size,title=f'redmapperID={ID}.jpg',save=save,method_func=method_func)
		rr.append(r)
	return l,rr

def meanl(lr,l,data):
	indx = np.where((l>=lr[0]) & (l<=l[1]))
	return np.mean(data[indx[0]])

def means_lambda_and_radius(dl1,dl2,l,rr,r_rings,plot=False,save=False):
	d0 = np.min(l)
	dm = np.max(l)+dl1
	dm2 = np.max(l)+dl2
	lr = np.arange(d0,dm,dl1)
	lr2 = np.arange(d0,dm2,dl2)
	r_means = []
	mradius = []
	for r in range(len(r_rings)-1):
		r_submeans = []
		sr = np.array(rr)[:,r]
		for i in range(len(lr)-1):
			mm = meanl([lr[i],lr[i+1]],l,sr)
			print(mm)
			r_submeans.append(mm)
		r_means.append(r_submeans)
	for i in range(len(lr2)-1):
		indx = np.where((l>=lr2[i]) & (l<=lr2[i+1]))
		mradius.append(np.mean(rr,axis=0))
	return r_means,mradius

def MLZ(l,z,log10M0,Flambda,Gz):
	return 10**(log10M0)*(l/40)**(Flambda)*((1+z)/(1+0.35))**(Gz)
def YVT(M):
	return M**(5/3)

