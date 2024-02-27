from AGN_extractor import radio_source
import time
import corner
from scipy.stats import norm
from scipy.stats import lognorm
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d as interp
import emcee
from scipy.special import erfinv
import os
from pixell import enmap,utils
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from astropy import units as u
from extract_clusters import extract_clusters,rings
from halo_models import radial_profiles,cluster_data,optimal_lamda_range
from astropy.io import fits
from scipy.optimize import curve_fit
from astropy.cosmology import WMAP9 as cosmo
import pandas as pd
from scipy.integrate import trapz,simps
from matplotlib.colors import LogNorm
from scipy.stats.kde import gaussian_kde
from scipy.stats import cauchy
from collections import OrderedDict
import matplotlib as mpl
from multiprocessing import Pool

linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('long dash with offset',              (5, (10, 3))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),
     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])



def first_steps_mcmc(parameters,nwalkers,random_dist = 'uniform',args=[0,1]):
	nparams = len(parameters)
	steps = np.zeros((nwalkers, nparams))
	for i in range(nwalkers):
		fraction =  1e5 * np.abs(np.random.normal(*args))
		steps[i] = [(parameters[j] + parameters[j] * fraction) for j in range(nparams)]
	return steps

def random_params_space(param,ndims,nwalkers,ptype=['log'],limits=[[-9,-6]]):
	param = np.tile(param,nwalkers).reshape(-1,ndims)
	for i in range(np.shape(param)[0]):
		for j in range(np.shape(param)[1]):
			if ptype[j]=='log':
				param[i][j] = 10**np.random.uniform(limits[j][0],limits[j][1])
			else:
				p = np.random.uniform(limits[j][0],limits[j][1])
				param[i][j] = p
	return np.array(param).astype(np.float64)

def scale_relationship(M):
	a = 1.47
	b = 0.75
	mass_pivot = 3e14
	return a + b*(np.log10(M) - np.log10(mass_pivot))

#MCMC general function:

def t_student_prior(theta, scale=1.0, shift=0.0):
	slope = theta
	log_prior_slope = cauchy.logpdf((slope - shift) / scale, loc=0, scale=scale)
	return log_prior_slope


def gaussian_likelihood(theta,x,y,yerr):
	mu = model(theta,x)
	log_likelihood = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((y - mu)**2) / (sigma**2)
	return np.sum(log_likelihood)


def flat_prior(theta,vmin,vmax):
	if (vmin <= theta <= vmax)==True:
		return 0.0
	return -np.inf

#========================

# =========invidivual MCMCM======


def projected_GNFW(R,M,params):
	p0,slope,gamma,b0,b1,alpha,c500,rc = params
	R_los = np.logspace(-7,9,200)[:,None]
	R = np.transpose([np.hypot(*np.meshgrid(R_los[:,i],R[:])) for i in range(R_los.shape[1])],axes=(1,2,0))
	profile = 2 * simps(GNFW(R,M,P0,gamma,beta,alpha,rc,c500),R_los[None],axis=1)[:,0]
	return profile

def GNFW(r,M,p0,slope,gamma,b0,b1,alpha,rc,c500):
	rc = 10**rc
	P0 = p0*10**slope
	beta = b0*10**b1
	return (P0)/(((c500*r/rc)**gamma)*(1 + (c500*r/rc)**alpha)**((beta-gamma)/alpha))

model = projected_GNFW


def ln_likelihood_individual(theta,x,y,sigma):
	mu = model(theta,x)
	log_likelihood = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((y - mu)**2) / (sigma**2)
	return np.sum(log_likelihood)


def ln_prior_individual(theta):
	P0,gam,bet,alp,c,rc = theta
	return t_student_prior(gam,1,0) + t_student_prior(bet,1,0) + t_student_prior(alp,1,0) + flat_prior(c,1.2,1.3) + flat_prior(rc,3,3.2) + flat_prior(P0,-25,-5)


def ln_posterior_individual(theta,x,y,sigma):
	prior = ln_prior_individual(theta)
	if np.isinf(prior):
		return prior
	likelihood = ln_likelihood_individual(theta,x,y,sigma)
	posterior = prior + likelihood
	if np.isnan(posterior)==True:
		return -1e12
	else:
		return prior + likelihood


# ==============================

#====first model=====

#first model with a mass dependence only in normalization P0:

def first_model(R,M,params):
	p0,slope,gamma,beta,alpha,c500,rc = params
	R_los = np.logspace(-7,9,200)[:,None]
	R = np.transpose([np.hypot(*np.meshgrid(R_los[:,i],R[:])) for i in range(R_los.shape[1])],axes=(1,2,0))
	profile = 2 * simps(GNFW_first_model(R,M,p0,slope,gamma,beta,alpha,rc,c500),R_los[None],axis=1)[:,0]
	return profile

def GNFW_first_model(r,M,p0,slope,gamma,beta,alpha,rc,c500):
	rc = 10**rc
	P0 = 10**p0
	mass_dependence = P0 * M ** slope
	return (mass_dependence)/(((c500*r/rc)**gamma)*(1 + (c500*r/rc)**alpha)**((beta-gamma)/alpha))

def ln_prior_first_model(theta):
	A,B,gam,bet,alp,c,rc = theta
	return t_student_prior(gam,1,-7) + t_student_prior(alp,5,0) + flat_prior(c,1.2,1.3) + flat_prior(rc,2.8,3.3) + flat_prior(A,-22,-15) + t_student_prior(B,5,1) + t_student_prior(bet,1,0)

def ln_likelihood_first_model(params,R,lamda_intervals,data,errors):
	mu = np.array([stacked_halo_model(params, R, lamda_interval) for lamda_interval in lamda_intervals])
	log_likelihood = -0.5 * np.log(2 * np.pi * errors**2) - 0.5 * ((data - mu)**2) / (errors**2)
	return np.sum(log_likelihood)

def ln_posterior_first_model(params,R,lamda_intervals,data,errors):
	prior = ln_prior_first_model(params)
	if np.isinf(prior):
	        return prior
	likelihood = ln_likelihood_first_model(params,R,lamda_intervals,data,errors)
	posterior = likelihood + prior
	if np.isnan(posterior) == True:
	        return -np.inf
	else:
	        return posterior

#====================

#====second model====
#model with two mass dependence: one in the nornalization and the other one in beta (external slope)

def second_model(R,M,params):
	p0,slope,gamma,B0,B1,alpha,c500,rc = params
	R_los = np.logspace(-7,9,200)[:,None]
	R = np.transpose([np.hypot(*np.meshgrid(R_los[:,i],R[:])) for i in range(R_los.shape[1])],axes=(1,2,0))
	profile = 2 * simps(GNFW_second_model(R,M,p0,slope,gamma,B0,B1,alpha,rc,c500),R_los[None],axis=1)[:,0]
	return profile

def GNFW_second_model(r,M,p0,slope,gamma,B0,B1,alpha,rc,c500):
	rc = 10**rc
	P0 = 10**p0
	bet = 10**B0
	normalization = P0 * M ** slope
	external_slope = bet * M ** B1
	return (normalization)/(((c500*r/rc)**gamma)*(1 + (c500*r/rc)**alpha)**((external_slope-gamma)/alpha))

def ln_prior_second_model(theta):
        A,B,gam,B0,B1,alp,c,rc = theta
        return t_student_prior(gam,1,-7) + t_student_prior(alp,1,4) + flat_prior(c,1.2,1.3) + flat_prior(rc,2.8,3.3) + flat_prior(A,-22,-15) + t_student_prior(B,1,0.5) + t_student_prior(B0,1,-1) + flat_prior(B1,-10,10)

def ln_likelihood_second_model(params,R,lamda_intervals,data,errors):
        mu = np.array([stacked_halo_model(params, R, lamda_interval) for lamda_interval in lamda_intervals])
        log_likelihood = -0.5 * np.log(2 * np.pi * errors**2) - 0.5 * ((data - mu)**2) / (errors**2)
        return np.sum(log_likelihood)

def ln_posterior_second_model(params,R,lamda_intervals,data,errors):
        prior = ln_prior_second_model(params)
        if np.isinf(prior):
                return prior
        likelihood = ln_likelihood_second_model(params,R,lamda_intervals,data,errors)
        posterior = likelihood + prior
        if np.isnan(posterior) == True:
                return -np.inf
        else:
                return posterior

#============================

#====third model====
#model with mass dependence and radio dependence in normalization

def third_model(R,M,params):
	p0,slope,gamma,beta,alpha,c500,rc,ap = params
	R_los = np.logspace(-7,9,200)[:,None]
	R = np.transpose([np.hypot(*np.meshgrid(R_los[:,i],R[:])) for i in range(R_los.shape[1])],axes=(1,2,0))
	profile = 2 * simps(GNFW_third_model(R,M,p0,slope,gamma,beta,alpha,rc,c500,ap),R_los[None],axis=1)[:,0]
	return profile

def GNFW_third_model(r,M,p0,slope,gamma,beta,alpha,rc,c500,ap):
	rc = 10**rc
	P0 = 10**p0
	x = r/rc
	ar = (0.1) - (ap + 0.1)*((x/0.5)**2/(1 + (x/0.5)**3))
	normalization = P0 * M ** (slope + ar)
	return (normalization)/(((c500*x)**gamma)*(1 + (c500*x)**alpha)**((beta-gamma)/alpha))


def ln_prior_third_model(theta):
	A,B,gam,bet,alp,c,rc,ap = theta
	return t_student_prior(gam,1,0) + t_student_prior(alp,1,0) + flat_prior(c,1.2,1.4) + flat_prior(rc,2.8,3.3) + flat_prior(A,-22,-19) + t_student_prior(B,1,0) + t_student_prior(bet,1,0) + t_student_prior(ap,1,0)

def ln_likelihood_third_model(params,R,lamda_intervals,data,errors):
	mu = np.array([stacked_halo_model(params, R, lamda_interval) for lamda_interval in lamda_intervals])
	log_likelihood = -0.5 * np.log(2 * np.pi * errors**2) - 0.5 * ((data - mu)**2) / (errors**2)
	return np.sum(log_likelihood)

def ln_posterior_third_model(params,R,lamda_intervals,data,errors):
	prior = ln_prior_third_model(params)
	if np.isinf(prior):
		return prior
	likelihood = ln_likelihood_third_model(params,R,lamda_intervals,data,errors)
	posterior = likelihood + prior
	if np.isnan(posterior) == True:
		return -np.inf
	else:
		return posterior

#============================


def param_step(p,p_center,log=False,Np=5,dp=2):
	if p_center!=p[0] and p_center!=p[-1]:
		minpindx = np.where(p==p_center)[0]
		start,end = [p[minpindx-1][0],p[minpindx+1][0]]
		if log==True:
			start,end = np.log10(start),np.log10(end)
			return np.logspace(start,end,Np,endpoint=False)
		else:
			return np.linspace(start,end,Np,endpoint=False)
	elif p_center==p[0]:
		if log==True:
			return np.logspace(np.log10(p[0]) - 2*dp, np.log10(p[0]),Np,endpoint=False)
		else:
			return np.linspace(p[0] - 2*dp, p[0],Np,endpoint=False)
	elif p_center==p[-1]:
		if log==True:
			return np.logspace(np.log10(p_center),np.log10(p_center)+2*dp,Np,endpoint=False)
		else:
			return np.linspace(p_center, p_center + 2*dp, Np,endpoint=False)


def marg_param(parr,interp,dchi2_1sigma,dchi2_2sigma,best_p):
	marg_chi2_p_interp = interp(parr)
	indx = np.argmin(np.abs(marg_chi2_p_interp - best_p))

	leftp = marg_chi2_p_interp[:indx]
	rightp = marg_chi2_p_interp[indx::]
	psigma1L = parr[:indx][np.argmin(np.abs(leftp - dchi2_1sigma))]
	psigma1R = parr[indx::][np.argmin(np.abs(rightp - dchi2_1sigma))]
	psigma2L = parr[:indx][np.argmin(np.abs(leftp - dchi2_2sigma))]
	psigma2R = parr[indx::][np.argmin(np.abs(rightp - dchi2_2sigma))]


	return marg_chi2_p_interp,psigma1L,psigma1R,psigma2L,psigma2R




def plot_params_with_sigma(lamda_intervals,params,log=False,label='P_0'):
	params = np.array(params)
	print(params)
	bp = params[:,0]
	print(bp)
	bp_erri = np.abs(params[:,1])
	bp_errs = np.abs(params[:,2])

	lamda = [(lamda_intervals[i-1]+lamda_intervals[i])/2 for i in range(1,len(lamda_intervals))]
	print(np.shape(lamda),np.shape(bp))
	fig = plt.figure()
	ax = plt.axes()
	ax.errorbar(lamda,bp,yerr=np.array([bp_erri,bp_errs]),ecolor='black',capsize=4,ls='--',color='black')
	ax.scatter(lamda,bp,s=2,color='black')
	ax.set(xlabel=f'{lamda}',ylabel=label,title=f'Parameter {label} in function of $\\lambda$')
	ax.grid(True)
	if log==True:
		ax.set(yscale='log')
	label = label.replace('$','').replace('\\','')
	fig.savefig(f'parameter_{label}.png')

def chi2min_p(chi2min,params,sort=True):
	for p in params:
		indx = np.where(params==p)
		if np.shape(indx)[1]>=2:
			m = np.min(chi2min[indx])
			print(m)
			if all(chi2min[indx]==m)==False:
				index2 = np.where((chi2min!=m) & (params==p))
				chi2min = np.delete(chi2min,index2)
				params = np.delete(params,index2)
			elif all(chi2min[indx]==m)==True:
				index2 = np.where(chi2min==m)[0]
				chi2min = np.delete(chi2min,index2[1::])
				params = np.delete(params,index2[1::])
	if sort==True:
		df = pd.DataFrame({'params':params,'chi2min':chi2min})
		df.sort_values(by='params',inplace=True)
		chi2min = df['chi2min']
		params = df['params']
		return chi2min,params

def lamda_prob(z,lamda_range, file='/data2/cristobal/actpol/lensing/cmblensing/des/selection/completeness_des.txt', plot = False, full_sample = False, target_shape = (192,6), reshape = False, print_output = False,):
	df = pd.read_csv(file,delimiter='|',usecols=[1,2,3,4]).to_numpy()
	lamda_obs = df[:,2]
	lamda_selected = np.where((lamda_obs <=lamda_range[1]) & (lamda_obs >= lamda_range[0]))
	prob_func = df[lamda_selected[0]]
	redshift = prob_func[:,0]
	closter_redshift = np.unique(redshift)[np.argmin(np.abs(np.unique(redshift) - z))]
	prob_func = prob_func[redshift == closter_redshift]
	lamda_true = prob_func[:,1]
	prob = []

	for i in range(len(lamda_true)):
		sl = prob_func[prob_func[:,1]==lamda_true[i]]
		prob.append(sl[:,-1])
	if np.shape(prob) != target_shape and reshape == True:
		print('\nchanging data shape...\n')
		prob_size = np.prod(np.shape(prob))
		target_size = np.prod(target_shape)
		stride = prob_size // target_size
		flatten = np.array(prob).flatten()
		first_reshaped_data = flatten[::stride]
		diff = abs(np.shape(first_reshaped_data) - target_size)
		if diff == 0:
			prob = np.array(first_reshape_data.reshape(target_shape))
		else:
			steps = np.shape(first_reshaped_data) // diff + 1
			second_reshaped_data = np.delete(first_reshaped_data, np.arange(0, first_reshaped_data.size, steps))
			if np.size(second_reshaped_data)>target_size:
				second_reshaped_data = np.delete(second_reshaped_data,np.random.randint(0,np.size(second_reshaped_data),int(np.size(second_reshaped_data) - target_size)))
			last_reshaped_data = np.reshape(second_reshaped_data,target_shape)
			prob = last_reshaped_data


	if plot == True:
		fig = plt.figure()
		ax = plt.axes()
		im = ax.imshow(np.log(np.abs(np.array(prob))),interpolation='nearest',origin='lower',extent=(lamda_true.min(),lamda_true.max(),lamda_range[0],lamda_range[1]))
		plt.colorbar(im,label='prob.',ax=ax)
		z_round = ''.join(list(str(z)))[0:6]
		ax.set(title=f'Probability function $P(\lambda_o|\lambda_t)$ at z={z_round} and $\\lambda\\in$[{np.round(lamda_range[0])},{np.round(lamda_range[1])}]',ylabel='$\\lambda_{obs}$', xlabel='$\\lambda_{true}$')
		yticks = np.round(np.linspace(lamda_range.min(),lamda_range.max(),8))
		xticks = np.round(np.linspace(lamda_true.min(),lamda_true.max(),8))
		ax.xaxis.set_ticks(xticks)
		ax.yaxis.set_ticks(yticks)
		ax.set_aspect('auto')
		plt.show()
		fig.savefig(f'joint_probability_at_z={str(np.round(z,4)).replace(".",",")}.png')
	if print_output == True:
		print(f'\nProbability function was created with z=\033[91m {np.round(z,4)}\033[00m and lambda in \033[91m[{np.round(lamda_range[0])},{np.round(lamda_range[1])}]\033[00m\noutput shape: {np.shape(prob)}')
	P_true = np.array(prob).transpose()
	lamda_obs = np.linspace(np.min(lamda_obs),np.max(lamda_obs),np.shape(P_true)[0])
	sigmaML = 0.25
	lamda_true = np.linspace(np.min(lamda_true),np.max(lamda_true),np.shape(prob)[0])
	mass_arr = np.logspace(12,16,30)
	lamda_true,mass = np.meshgrid(lamda_true,mass_arr)
	lamda_model = 10**np.array(scale_relationship(mass))
	P_mass = 1/(np.sqrt(2*np.pi*sigmaML)) * np.exp(-(np.log(lamda_true) - np.log(lamda_model))**2/(2*sigmaML**2))
	product = P_true[:,:,None] * P_mass.transpose()
	lamda_true = np.linspace(np.min(lamda_true),np.max(lamda_true),np.shape(prob)[0])
	P_obs = trapz(product,x=lamda_true,axis=1)
	P_MR = trapz(P_obs,x=lamda_obs,axis=0)
	if full_sample == True:
		return P_MR,P_obs,P_true
	return P_MR


def projected_radius(params):
	lamda = params['lamda']
	h = params['h']
	return 1.0*(lamda/100)**0.2/h*u.kpc*1e3

def projected_GNFW1(R,M,a,b,gamma,beta,alpha,rc,c500):
	R_los = np.logspace(-7,9,200)[:,None]
	R = np.transpose([np.hypot(*np.meshgrid(R_los[:,i],R[:])) for i in range(R_los.shape[1])],axes=(1,2,0))
	profile = 2 * simps(GNFW(R,M,a,b,gamma,beta,alpha,rc,c500),R_los[None],axis=1)[:,0]
	return profile

def projected_GNFW0(R,M,P0,gamma,beta,alpha,rc,c500):
        R_los = np.logspace(-7,9,200)[:,None]
        R = np.transpose([np.hypot(*np.meshgrid(R_los[:,i],R[:])) for i in range(R_los.shape[1])],axes=(1,2,0))
        profile = 2 * simps(GNFW0(R,M,P0,gamma,beta,alpha,rc,c500),R_los[None],axis=1)[:,0]
        return profile

def GNFW0(r,M,P0,gamma,beta,alpha,rc,c500):
        return P0/(((c500*r/rc)**gamma)*(1 + (c500*r/rc)**alpha)**((beta-gamma)/alpha))

def GNFW(r,M,a,b,gamma,beta,alpha,rc,c500):
	P0 = a*(M)**b
	return P0/(((c500*r/rc)**gamma)*(1 + (c500*r/rc)**alpha)**((beta-gamma)/alpha))


def projected_profile(R,M,params):
	p0,slope,gamma,b0,b1,alpha,c500,rc = params
	R_los = np.logspace(-7,9,200)[:,None]
	R = np.transpose([np.hypot(*np.meshgrid(R_los[:,i],R[:])) for i in range(R_los.shape[1])],axes=(1,2,0))
	profile = 2 * simps(radial_profile(R,M,p0,slope,gamma,b0,b1,alpha,rc,c500),R_los[None],axis=1)[:,0]
	return profile

def radial_profile(r,M,p0,slope,gamma,b0,b1,alpha,rc,c500):
	rc = 10**rc
	P0 = 10**p0
	B0 = 10**b0
	mass_dependence  = P0 * M ** slope
	mass_dependence_beta = B0 * M ** b1

	return (mass_dependence)/(((c500*r/rc)**gamma)*(1 + (c500*r/rc)**alpha)**((mass_dependence_beta-gamma)/alpha))


#radio data
"""
path = '/data2/javierurrutia/szeffect/data/'
hosts_file = 'CIRADA_VLASS1QL_table2_hosts.csv'
sources_file = 'CIRADA_VLASS1QLv3.1_table1_components.csv'
agn_file = 'sources_ql_epoch1.fits'
hosts_table = pd.read_csv(f'{path}{hosts_file}')
hosts_table = hosts_table[hosts_table['N_components']==3.0]
sources_table = pd.read_csv(f'{path}{sources_file}')
dragn_table = fits.open(f'{path}{agn_file}')[1].data

i = 0
while i<50:
	random_indx = np.random.randint(0,len(hosts_table))
	host = hosts_table.iloc[random_indx]
	rad = radio_source(host['RA_Source'],host['DEC_Source'],sources_table,hosts_table,True,dragn_table)
	if type(rad.dragn)==type(None):
		rad.download_fits(save=True)
	#def __init__(self,RA,DEC,sources_catalog=None,hosts_catalog=None,DRAGN=False,DRAGN_data=None,method='basic',fits_file=False):
#---
"""
#sz and optical data

scale = 1 #degrees
szmap,rmcat,nszcat,ind,radii = extract_clusters(scale,[0.1,1],FWHM_correction=True,match=True,skip=1)


r_rings = np.linspace(300,3300,8)*u.kpc

lamda,lamda_err,radial_pr,radial_pr_err,Z = radial_profiles(szmap,rmcat,r_rings,radii,1e-3,method_func=np.mean,save=False) #,norm_func = projected_radius)





lamda_range1 = optimal_lamda_range(lamda,radial_pr,radial_pr_err,19,240,1,1,4,6.5,full_sample=True,N_min=len(lamda)/10)
print(lamda_range1)
lamda_arr1 = lamda_range1
lamda_mean = [(lamda_arr1[i-1]+lamda_arr1[i])/2 for i in range(1,len(lamda_arr1))]

fig = plt.figure(figsize=(4,3))
ax = plt.axes()
ax.hist(lamda,bins=lamda_mean,histtype='step',color='black')
ax.set(xlabel='Intervalo de riqueza',ylabel='N° de cúmulos')
ax.grid(True)
fig.tight_layout()
fig.savefig('lambda_hist.png')
fig = plt.figure(figsize=(6,6))

gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
ax = fig.add_subplot(gs[1,0])
ax_histx = fig.add_subplot(gs[0,0],sharex=ax)
ax_histy = fig.add_subplot(gs[1,1],sharey=ax)
k = gaussian_kde(np.vstack([Z, lamda]))
x = Z
y = lamda
xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
ax.scatter(Z,lamda,color='red',alpha=0.8,s=5)
ax_histx.hist(Z,color='red',histtype='step')
ax.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5,cmap='Reds',vmin=np.min(zi)*5,levels=25)
ax_histx.tick_params(axis='x',labelbottom=False)
ax_histy.hist(lamda,color='red',histtype='step',orientation='horizontal')
ax_histy.tick_params(axis='y',labelleft=False)
ax.set(ylabel='Riqueza $\\lambda$',xlabel='Redshift $z$')
ax.grid(True)
ax_histy.grid(True)
ax_histx.grid(True)
fig.tight_layout()
fig.savefig('lamda_redshift_scatter.png')


clusters_groups = []


fig_means = plt.figure(figsize=(8,5))
ax_means = plt.axes()

bp0 = ba = bb = bc = np.array([])

greys = []

for i in range(len(lamda_arr1)):
	greys.append(mpl.colormaps['Greys'](int(50 + 206 / (len(lamda_arr1) - 1) * i)))

best_fittings = []
lower_errors = []
upper_errors = []
all_samples = []
fit='all'
y_profile = []
sigma_arr = []
lamda_min = []
lamda_max = []
lamda_intervals = []
joint_probability = []
redshifts = []
lamda_probs = {}
z_dict = {}
for i in range(len(lamda_arr1)-1):
	lamda_condition = np.where((lamda>=lamda_arr1[i]) & (lamda<lamda_arr1[i+1]))
	clusters_groups.append(cluster_data(radial_pr[lamda_condition],r_rings,lamda[lamda_condition],radial_pr_err[lamda_condition],redshift=Z[lamda_condition]))
	clusters_groups[-1].weight_mean(weighted=True)
	if i<len(list(linestyles.keys())):
		ls = linestyles[list(linestyles.keys())[i]]
		clusters_groups[-1].plot_weight_mean(ax_means,fig_means,label=False,color=greys[i],log=True)
	clusters_groups[-1].write_mean_in(path='./',file='mean_data.txt',mode='a')
	clusters_groups[-1].fit_datax = r_rings[1::].value
	clusters_groups[-1].fit_datay = clusters_groups[-1].mean['means']
	clusters_groups[-1].sigma = clusters_groups[-1].mean['errors']
	clusters_groups[-1].miscellanea([12,16],ndims=(30,10,10))
	kwargs = {'init_params':[1e-7,1,1,1,1,1e3],'ln_prior':ln_prior_individual,'nsteps':1000,'nwalkers':100}
	c = clusters_groups[-1]
	lamda_interval = np.array([np.round(np.min(c.lamda))    ,np.round(np.max(c.lamda))      ])
	y_profile.append(c.fit_datay)
	sigma_arr.append(c.sigma)
	lamda_min = np.append(lamda_min,np.min(lamda_interval))
	lamda_max = np.append(lamda_max,np.max(lamda_interval))
	lamda_intervals.append(lamda_interval)
	redshifts = np.append(redshifts,c.z_main)
	lamda_probs[str(lamda_intervals[-1])] = lamda_prob(redshifts[-1],np.array(lamda_interval),reshape = False ,plot = False)
	z_dict[str(lamda_intervals[-1])] = redshifts[-1]
	if i==len(lamda_arr1) - 2:
		lamda_intervals = np.array(lamda_intervals)
		dndM = np.loadtxt('dndlog10M.txt')
		mass_arr = np.logspace(12,16,len(dndM))
		def stacked_halo_model(params,R,lamda):
			z = z_dict[str(lamda)]
			P_MR = lamda_probs[str(lamda)]
			y_model = np.array([third_model(R,M,params) for M in mass_arr]).astype(np.float64)
			I = trapz(dndM[:,None] * (P_MR[:,None] * y_model),x=mass_arr,axis=0)
			norm = trapz(dndM * P_MR,x=mass_arr,axis=0)
			mean_profile = I/norm
			result = np.array(mean_profile)
			return np.array(result)
	if fit=='individual':
		clusters_groups[-1].fit(method='MCMC',kwargs=kwargs)
		best_fitting = clusters_groups[-1].best_fitting
		ndims = len(best_fitting)
		samples = clusters_groups[-1].MCMCsamples
		sampler = clusters_groups[-1].MCMCsampler
		last_steps = samples[-len(samples)//3::]
		median_parameters = np.median(last_steps,axis=0)
		lower,upper = np.percentile(last_steps,[2.5,97.5],axis=0)
		lower_errors.append(median_parameters - lower)
		upper_errors.append(median_parameters - upper)
		r = clusters_groups[-1].R[1:]
		fig = plt.figure(figsize=(8,5))
		ax = plt.axes()
		#for params in samples[np.random.randint(len(samples), size=100)]:
		#	fit = clusters_groups[-1].stacked_halo_model(params,r)
		#	ax.plot(r,fit,color='k',alpha=0.1)
		ax.plot(r,clusters_groups[-1].stacked_halo_model(best_fitting,r),color='red',alpha=0.6,label='Mejor ajuste')
		ax.errorbar(r, clusters_groups[-1].mean['means'], yerr=clusters_groups[-1].mean['errors'], fmt=".k",label='Datos',capsize=3)
		ax.set(yscale='log',xlabel='r (kpc)',ylabel='$\\langle y \\rangle$',ylim=(1e-7,1e-2))
		ax.legend()
		ax.grid(True)
		fig.savefig(f'best_fitting_[{lamda_arr1[i]},{lamda_arr1[i+1]}].png')
		plot_corner=False
		if plot_corner==True:
			fig = corner.corner(samples, labels=['$P_0$','$\\gamma$','\\beta','$\\alpha$','$c$','$r_s$'],
						truths=best_fitting, show_titles=True,
						)
			fig.savefig(f'corner_[{lamda_arr1[i]},{lamda_arr1[i+1]}].png')
		best_fittings.append(best_fitting)
		plt.close(fig)
		all_samples.append(samples)
	elif fit=='all':
		n_model = 'third'
		if i==len(lamda_arr1) - 2:
			y_profile = np.array(y_profile)
			sigma_arr = np.array(sigma_arr)
			initial_params = np.array([-13,0.1,1,1,1,1,3,1])
			nwalkers = 200
			nsteps = 5000
			ndim = len(initial_params)
			R = r_rings[1:].value
			use_last_parameters=False
			param_limits = [[-25,-19],[-5,5],[-5,5],[-5,5],[-5,5],[2,3],[2.5,4],[-5,5]]
			initial_params = random_params_space(initial_params,ndim,nwalkers,ptype=np.full(len(param_limits),'lin'),limits=param_limits)
			#initial_params = first_steps_mcmc(initial_params,nwalkers)
			print(initial_params)
			if use_last_parameters==True:
				last_samples = np.load(f'mcmc_samples_{n_model}_model.npy')
				ln_posterior = np.load(f'mcmc_ln_posterior_{n_model}_model.npy')
				best_parameters = last_samples[np.argmax(ln_posterior)]
				median_parameters = np.median(last_steps,axis=0)
				lower,upper = np.percentile(last_steps,[15.865,84.135],axis=0)
				param_limits = [[lower[j],upper[j]] for j in range(len(lower))]
				initial_params = random_params_space(median_parameters,ndim,nwalkers,ptype=['lin','lin','lin','lin','lin','lin','lin','lin','lin','lin'],limits=param_limits)
			with Pool(40) as pool:
				sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior_third_model,args=(R, lamda_intervals, y_profile, sigma_arr) ,pool=pool)
				start = time.time()
				sampler.run_mcmc(initial_params,nsteps,progress=True)
				end = time.time()
				multi_time = end - start
				print("Multiprocessing took {0:.1f} seconds".format(multi_time))
			samples = sampler.get_chain(flat=True)
			np.save(f'mcmc_samples_{n_model}_model.npy', samples)
			ln_posterior_values = sampler.get_log_prob(flat=True)
			np.save(f'mcmc_ln_posterior_{n_model}_model.npy', ln_posterior_values)
			best_index = np.argmax(ln_posterior_values)
			best_parameters = samples[best_index]
			last_steps = samples[-len(samples)//2::]
			median_parameters = np.median(last_steps,axis=0)
			lower,upper = np.percentile(last_steps,[15.865,84.135],axis=0)
			lower_errors.append(median_parameters - lower)
			upper_errors.append(median_parameters - upper)
			fig = plt.figure(figsize=(6,4))
			ax = plt.axes()
			for i in range(0,len(greys) - 1,2):
				ax.errorbar(R,y_profile[i],color=greys[i],yerr=sigma_arr[i],capsize=3,ls='None',marker='o')
				fit = stacked_halo_model(best_parameters,R,lamda_intervals[i])
				print(fit)
				ax.plot(R,stacked_halo_model(best_parameters,R,lamda_intervals[i]),color=greys[i])
			ax.set(yscale='log',xlabel='r (kpc)',ylabel='$\\langle y \\rangle$',ylim=(1e-7,1e-2),title='radial profiles of $\\langle y \\rangle$ with best fitting')
			ax.grid(True)
			greys = plt.get_cmap('Greys')
			cNorm  = colors.Normalize(vmin=np.min(lamda_arr1), vmax=np.max(lamda_arr1))
			scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=greys)
			cax = fig.add_axes([0.80, 0.58, 0.02, 0.25])
			colbar = plt.colorbar(scalarMap, cax = cax, ticks=np.linspace(np.min(lamda_arr1),np.max(lamda_arr1),10),
                        	                    orientation='vertical',label='riqueza $\lambda$')
			colbar.ax.yaxis.set_label_position('left')
			colbar.set_label(label='Riqueze $\lambda$',size=15,weight='bold')
			fig.savefig(f'best_fitting_all_intervals.png')
			plot_corner = True
			if plot_corner==True:
				fig = corner.corner(last_steps, labels=['$p_0$','$\\rho$','$\\gamma$','$\\beta$','$\\alpha$','$c$','$r_s$','$\\alpha_{p}$'],
				        	        truths=best_parameters, show_titles=True,
							)
			fig.savefig(f'corner_all_intervals.png')
	elif fit=='no':
		continue;



#====================chi squared distribution ========================#

def chi2(y,y1,sigma):
	return np.sum((y - y1)**2/np.array(sigma)**2)

def first_stacked_halo_model(params,R,lamda):
	z = z_dict[str(lamda)]
	P_MR = lamda_probs[str(lamda)]
	y_model = np.array([first_model(R,M,params) for M in mass_arr]).astype(np.float64)
	I = trapz(dndM[:,None] * (P_MR[:,None] * y_model),x=mass_arr,axis=0)
	norm = trapz(dndM * P_MR,x=mass_arr,axis=0)
	mean_profile = I/norm
	result = np.array(mean_profile)
	return np.array(result)

def second_stacked_halo_model(params,R,lamda):
	z = z_dict[str(lamda)]
	P_MR = lamda_probs[str(lamda)]
	y_model = np.array([second_model(R,M,params) for M in mass_arr]).astype(np.float64)
	I = trapz(dndM[:,None] * (P_MR[:,None] * y_model),x=mass_arr,axis=0)
	norm = trapz(dndM * P_MR,x=mass_arr,axis=0)
	mean_profile = I/norm
	result = np.array(mean_profile)
	return np.array(result)

def third_stacked_halo_model(params,R,lamda):
	z = z_dict[str(lamda)]
	P_MR = lamda_probs[str(lamda)]
	y_model = np.array([third_model(R,M,params) for M in mass_arr]).astype(np.float64)
	I = trapz(dndM[:,None] * (P_MR[:,None] * y_model),x=mass_arr,axis=0)
	norm = trapz(dndM * P_MR,x=mass_arr,axis=0)
	mean_profile = I/norm
	result = np.array(mean_profile)
	return np.array(result)



#=========

lamda_mean = [(lamda_arr1[i-1]+lamda_arr1[i])/2 for i in range(1,len(lamda_arr1))]
samples_individual_fit = [np.load(f'samples_[{np.min(c.lamda)}-{np.max(c.lamda)}].npy') for c in clusters_groups]
ln_posteriors_individual_fit = [np.load(f'ln_posterior_[{np.min(c.lamda)}-{np.max(c.lamda)}].npy].npy') for c in clusters_groups]
best_individual_fits = [samples_individual_fit[i][np.argmax(ln_posteriors_individual_fit[i])] for i in range(len(samples_individual_fit))]
r = r_rings[1::].value
fittings = [clusters_groups[i].stacked_halo_model(best_individual_fits[i],r) for i in range(len(samples_individual_fit))]
chi2_individual_fit = [chi2(clusters_groups[i].mean['means'],fittings[i],clusters_groups[i].mean['errors']) for i in range(len(fittings))]
chi2_radio_func_individual = [[( ( clusters_groups[i].mean['means'][j] - fittings[i][j] ) / clusters_groups[i].mean['errors'][j])**2 for j in range(len(clusters_groups[i].mean['means']))] for i in range(len(clusters_groups))]


samples_first_model = np.load('mcmc_samples_first_model.npy')
ln_posterior_first_model = np.load('mcmc_ln_posterior_first_model.npy')
best_parameters_first_model = samples_first_model[np.argmax(ln_posterior_first_model)]
fittings = [first_stacked_halo_model(best_parameters_first_model,r,lamda_intervals[i]) for i in range(len(lamda_intervals))]
chi2_first_model = [chi2(clusters_groups[i].mean['means'],fittings[i],clusters_groups[i].mean['errors']) for i in range(len(fittings))]
chi2_radio_func_first_moddel = [ [ ( (clusters_groups[i].mean['means'][j] - fittings[i][j]) / clusters_groups[i].mean['errors'][j])**2 for j in range(len(clusters_groups[i].mean['means'])) ] for i in range(len(clusters_groups))]

samples_second_model = np.load('mcmc_samples_second_model.npy')
ln_posterior_second_model = np.load('mcmc_ln_posterior_second_model.npy')
best_parameters_second_model = samples_second_model[np.argmax(ln_posterior_second_model)]
fittings = [second_stacked_halo_model(best_parameters_second_model,r,lamda_intervals[i]) for i in range(len(lamda_intervals))]
chi2_second_model = [chi2(clusters_groups[i].mean['means'],fittings[i],clusters_groups[i].mean['errors']) for i in range(len(fittings))]
chi2_radio_func_second_moddel = [[( (clusters_groups[i].mean['means'][j] - fittings[i][j]) / clusters_groups[i].mean['errors'][j])**2 for j in range(len(clusters_groups[i].mean['means'])) ] for i in range(len(clusters_groups))]


samples_third_model = np.load('mcmc_samples_third_model.npy')
ln_posterior_third_model = np.load('mcmc_ln_posterior_third_model.npy')
best_parameters_third_model = samples_third_model[np.argmax(ln_posterior_third_model)]
fittings = [third_stacked_halo_model(best_parameters_third_model,r,lamda_intervals[i]) for i in range(len(lamda_intervals))]
chi2_third_model = [chi2(clusters_groups[i].mean['means'],fittings[i],clusters_groups[i].mean['errors']) for i in range(len(fittings))]
print(best_parameters_third_model)
#chi2_radio_func_second_moddel = [[( (clusters_groups[i].mean['means'][j] - fittings[i][j]) / clusters_groups[i].mean['errors'][j])**2 for j in range(len(clusters_groups[i].mean['means'])) ] for i in range(len(clusters_groups))]



fig = plt.figure()
ax = plt.axes()
ax.plot(lamda_mean,chi2_individual_fit,color='red',label='individual_fittings')
ax.plot(lamda_mean,chi2_first_model,color='green',label='first model')
ax.plot(lamda_mean,chi2_second_model,color='blue',label='second_model')
ax.plot(lamda_mean,chi2_third_model,color='orange',label='third_model')

ax.set(yscale='log',ylabel='$\\chi^2$',xlabel='$\\lambda$',title='$\\chi^2$ distribution for each model in function of $\\lambda$')
ax.legend()
fig.savefig('chi2.png')

#==surface===

fig = plt.figure()
ax = plt.axes()
[ax.plot(r,chi2_radio_func_individual[i],label=f'{lamda_mean[i]}') for i in range(len(chi2_radio_func_individual[i]))]
ax.legend()
fig.savefig('chi2_individual_fittings.png')
#=========================================================================

samples = np.array(all_samples)
lamda = [(lamda_arr1[i-1]+lamda_arr1[i])/2 for i in range(1,len(lamda_arr1))]
best_fittings = np.array(best_fittings)
ndims = np.shape(best_fittings)[1]
fig,ax = plt.subplots(ndims,sharex=True,figsize=(5,3))
labels=['$P_0$','$\\gamma$','$\\beta$','$\\alpha$','$c$','$r_s$']
for i in range(ndims):
	yerr = np.abs(np.array([np.array(lower_errors)[:,i],np.array(upper_errors)[:,i]]))
	print(yerr.shape)
	ax[i].errorbar(lamda,best_fittings[:,i],yerr=yerr,capsize=3)
	ax[i].set(ylabel=labels[i],xlabel='$\\lambda$')

fig.savefig('params_distribution.png')


greys = plt.get_cmap('Greys')
cNorm  = colors.Normalize(vmin=np.min(lamda_arr1), vmax=np.max(lamda_arr1))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=greys)
cax = fig_means.add_axes([0.85, 0.58, 0.03, 0.35])
colbar = plt.colorbar(scalarMap, cax = cax, ticks=np.linspace(np.min(lamda_arr1),np.max(lamda_arr1),10),
                                            orientation='vertical',label='riqueza $\lambda$')

colbar.set_label(label='Riqueze $\lambda$',size=15,weight='bold')

ax_means.set_xlabel('$R$ (kpc)',fontsize=18)
ax_means.set_ylabel('$\\langle y \\rangle (R)$',fontsize=18)
fig_means.tight_layout()
fig_means.savefig('weighted_mean.png')


labels=['P_0','gamma','$beta','$alpha','c','r_s']
dict_params = {l:[] for l in labels}
dict_params['lambda'] = []

for i in range(len(lamda)):
	dict_params['lambda'].append(lamda[i])
	for j in range(len(labels)):
		dict_params[labels[j]].append([best_fittings[i,j],np.array(upper_errors)[i,j],np.array(lower_errors)[i,j]])
df = pd.DataFrame(dict_params)
df.to_csv('params.csv',sep='\t',header=True)










"""

	Np = 9
	P0_center = np.log10(1e-7)
	dp0 = 2
	a_center = 1
	b_center = 4
	c_center = 0
	da = 2
	db = 2
	dc = 10

	N=0

	P0arr,chi2minp0arr = np.array([]),np.array([])
	alp_arr = chi2alp_arr = np.array([])
	bet_arr = chi2bet_arr = np.array([])
	gam_arr = chi2gam_arr = np.array([])
	DF = len(clusters_groups[-1].mean['means']) - 4
	chi2_values = chi2_table.iloc[int(DF)-1]
	chi2_abs_min = 1000
	chi2min = 3
	fp0 = fa = fb = fc = 1

	dchi2_1sigma = 1.07
	dchi2_2sigma = 3.84

	P0 = np.logspace(P0_center - dp0,P0_center + dp0,Np)
	gamma = np.linspace(c_center - dc,c_center + dc,Np)
	beta = np.linspace(b_center - db,b_center + db,Np)
	alpha = np.linspace(a_center - da,a_center + da,Np)
	comb = np.array(np.meshgrid(P0,gamma,beta,alpha)).T.reshape(-1,4)

	while chi2_abs_min>chi2min:
		N+=1
		fixed_params = np.tile([0.7e3,1.177],np.shape(comb)[0]).reshape(-1,2)
		comb = np.append(comb,fixed_params,axis=1)
		clusters_groups[-1].fit(projected_GNFW0,method='stacked_halo_model',kwargs={'params':comb,'free_params':[0,1],
			'mass_range':[clusters_groups[-1].mass_range[0],clusters_groups[-1].mass_range[1]]})
		P0_center,c_center,b_center,a_center = clusters_groups[-1].best_fitting[0:4]

		P0 = param_step(P0,P0_center,log=True,Np=Np,dp=dp0)
		print(P0,'\n',gamma)
		gamma = param_step(gamma,c_center,log=False,Np=Np,dp=dc)
		beta = param_step(beta,b_center,log=False,Np=Np,dp=db)
		alpha = param_step(alpha,a_center,log=False,Np=Np,dp=da)

		comb = np.array(np.meshgrid(P0,gamma,beta,alpha)).T.reshape(-1,4)

		chi2_abs_min = np.min(clusters_groups[-1].chi2)
		print('chi2:',chi2_abs_min)
		for j in range(len(P0)):
			indx = np.where(comb[:,0]==P0[j])
			chi2min_P0 = np.min(clusters_groups[-1].chi2[indx])
			P0arr = np.append(P0arr,P0[j])
			chi2minp0arr = np.append(chi2minp0arr,chi2min_P0)
			indx = np.where(comb[:,3]==alpha[j])
			chi2min_alp = np.min(clusters_groups[-1].chi2[indx])
			alp_arr = np.append(alp_arr,alpha[j])
			chi2alp_arr = np.append(chi2alp_arr,chi2min_alp)
			indx = np.where(comb[:,2]==beta[j])
			chi2min_bet = np.min(clusters_groups[-1].chi2[indx])
			bet_arr = np.append(bet_arr,beta[j])
			chi2bet_arr = np.append(chi2bet_arr,chi2min_bet)
			indx = np.where(comb[:,1]==gamma[j])
			chi2min_gam = np.min(clusters_groups[-1].chi2[indx])
			gam_arr = np.append(gam_arr,gamma[j])
			chi2gam_arr = np.append(chi2gam_arr,chi2min_gam)
		if N>=20:
			break


	chi2minp0arr,P0arr = chi2min_p(chi2minp0arr,P0arr)
	chi2alp_arr,alp_arr = chi2min_p(chi2alp_arr,alp_arr)
	chi2bet_arr,bet_arr = chi2min_p(chi2bet_arr,bet_arr)
	chi2gam_arr,gam_arr = chi2min_p(chi2gam_arr,gam_arr)


	c = clusters_groups[-1]
	r = c.R[1::]


	bestP0,best_gam,best_bet,best_alp = c.best_fitting[0:4]


	intp0 =  interp(P0arr,chi2minp0arr)
	int_alp =  interp(alp_arr,chi2alp_arr)
	int_bet =  interp(bet_arr,chi2bet_arr)
	int_gam =  interp(gam_arr,chi2gam_arr)

	fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(8,8))

	try:
		ax1.scatter(P0arr,chi2minp0arr,s=1,alpha=0.5,color='black')
		P0arr = np.logspace(np.log10(np.min(P0arr)),np.log10(np.max(P0arr)),100)

		marg_chi2_P0_interp, P0sigma1L,P0sigma1R,P0sigma2L,P0sigma2R = marg_param(P0arr,intp0,dchi2_1sigma,dchi2_2sigma,bestP0)

		ax1.plot(P0arr,marg_chi2_P0_interp)

		ax1.axvline(x=P0sigma1L,ls='--',color='red',label='$1\\sigma$')
		ax1.axvline(x=P0sigma1R,ls='--',color='red')
		ax1.axvline(x=P0sigma2L,ls='--',color='green',label='$2\\sigma$')
		ax1.axvline(x=P0sigma2R,ls='--',color='green')
		ax1.axvline(x=bestP0,ls='--',color='black',label='best $P_0$')
		ax1.legend()

		ax1.set(title='Marginalization of $P_0$',xlabel='$P_0$',ylabel='$\\chi^2$',xscale='log')
		ax1.grid(True)

		ax2.scatter(alp_arr,chi2alp_arr,s=1,alpha=0.5,color='black')
		alp_arr = np.linspace(np.min(alp_arr),np.max(alp_arr),100)

		marg_chi2_alp_interp, alpsigma1L,alpsigma1R,alpsigma2L,alpsigma2R = marg_param(alp_arr,int_alp,dchi2_1sigma,dchi2_2sigma,best_alp)

		ax2.plot(alp_arr,marg_chi2_alp_interp)

		ax2.axvline(x=alpsigma1L,ls='--',color='red',label='$1\\sigma$')
		ax2.axvline(x=alpsigma1R,ls='--',color='red')
		ax2.axvline(x=alpsigma2L,ls='--',color='green',label='$2\\sigma$')
		ax2.axvline(x=alpsigma2R,ls='--',color='green')
		ax2.axvline(x=best_alp,ls='--',color='black',label='best $\\alpha$')
		ax2.legend()

		ax2.set(title='Marginalization of $\\alpha$',xlabel='$\\alpha$',ylabel='$\\chi^2$')
		ax2.grid(True)


		ax3.scatter(bet_arr,chi2bet_arr,s=1,alpha=0.5,color='black')
		bet_arr = np.linspace(np.min(bet_arr),np.max(bet_arr),100)

		marg_chi2_bet_interp,betsigma1L,betsigma1R,betsigma2L,betsigma2R = marg_param(bet_arr,int_bet,dchi2_1sigma,dchi2_2sigma,best_bet)

		ax3.plot(bet_arr,marg_chi2_bet_interp)

		ax3.axvline(x=betsigma1L,ls='--',color='red',label='$1\\sigma$')
		ax3.axvline(x=betsigma1R,ls='--',color='red')
		ax3.axvline(x=betsigma2L,ls='--',color='green',label='$2\\sigma$')
		ax3.axvline(x=betsigma2R,ls='--',color='green')
		ax3.axvline(x=best_bet,ls='--',color='black',label='best $\\beta$')
		ax3.legend()


		ax3.set(title='Marginalization of $\\beta$',xlabel='$\\beta$',ylabel='$\\chi^2$')
		ax3.grid(True)

		ax4.scatter(gam_arr,chi2gam_arr,s=1,alpha=0.5,color='black')
		gam_arr = np.linspace(np.min(gam_arr),np.max(gam_arr),100)

		marg_chi2_gam_interp,gamsigma1L,gamsigma1R,gamsigma2L,gamsigma2R = marg_param(gam_arr,int_gam,dchi2_1sigma,dchi2_2sigma,best_gam)


		ax4.plot(gam_arr,marg_chi2_gam_interp)

		ax4.axvline(x=gamsigma1L,ls='--',color='red',label='$1\\sigma$')
		ax4.axvline(x=gamsigma1R,ls='--',color='red')
		ax4.axvline(x=gamsigma2L,ls='--',color='green',label='$2\\sigma$')
		ax4.axvline(x=gamsigma2R,ls='--',color='green')
		ax4.axvline(x=best_gam,ls='--',color='black',label='best $\\gamma$')
		ax4.legend()
		ax4.set(title='Marginalization of $\\gamma$',xlabel='$\\gamma$',ylabel='$\\chi^2$')
		ax4.grid(True)

	except:
		print(':(')

	lamda_arr = [np.round(np.min(c.lamda)),np.round(np.max(c.lamda))]

	fig.suptitle(f'Marginalization of parameters in $\\lambda\\in${lamda_arr}',fontsize=25)
	fig.tight_layout()
	fig.savefig(f'marginalization_in_{lamda_arr}.png')

	best_fitting = c.best_fitting
	lamda_arr = [np.round(np.min(c.lamda)),np.round(np.max(c.lamda))]
	fig = plt.figure(figsize=(12,6))
	fit = projected_GNFW0(r,1,*best_fitting)
	ax = plt.axes()
	ax.plot(r,fit,label='fitting',color='black')
	ax.scatter(r,c.fit_datay,color='black')
	ax.errorbar(r,c.fit_datay,yerr=c.sigma,ls='--',label='data',color='black')
	ax.legend()
	ax.set(yscale='log',ylabel='$\\langle y \\rangle (R)$',xlabel='$R$(Kpc)')
	fig.suptitle(f'best fitting with GNFW using stacked halo model at $\\lambda \\in$ {lamda_arr}')
	ax.grid(True)
	fig.savefig(f'best_fitting_in_{lamda_arr}.png')

	bp0 = np.concatenate((bp0,[bestP0,P0sigma1L,P0sigma1R]))
	ba = np.concatenate((ba,[best_alp,alpsigma1L,alpsigma1R]))
	bc = np.concatenate((bc,[best_gam,gamsigma1L,gamsigma1R]))
	bb = np.concatenate((bb,[best_bet,betsigma1L,betsigma1R]))


	plt.close()
#	dic = {'lambda':str(lamda_arr),'a':best_fitting[0],'b':best_fitting[1],'gamma':best_fitting[2],'beta':best_fitting[3],'alpha':best_fitting[4],'r_s':best_fitting[5],'c_500':best_fitting[6]}
#	df = pd.DataFrame(dic,index=[i])
#	if i==0:
#		df.to_csv('fixed_params.csv',sep='\t',mode='a',header=True)
#	if i>0:
#		df.to_csv('fixed_params.csv',sep='\t',mode='a',header=False)


plot_params_with_sigma(lamda_arr1,bp0.reshape(-1,3),log=True,label='$P_0$')
plot_params_with_sigma(lamda_arr1,ba.reshape(-1,3),label='$\\alpha$')
plot_params_with_sigma(lamda_arr1,bb.reshape(-1,3),label='$\\beta$')
plot_params_with_sigma(lamda_arr1,bc.reshape(-1,3),label='$\\gamma$')


fig.suptitle('weighted mean profiles on different range of $\\lambda$',fontsize=28)
greys = plt.get_cmap('Greys')
cNorm  = colors.Normalize(vmin=np.min(lamda_arr1), vmax=np.max(lamda_arr1))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=greys)
cax = fig.add_axes([0.85, 0.55, 0.03, 0.35])
plt.colorbar(scalarMap, cax = cax, ticks=np.linspace(np.min(lamda_arr1),np.max(lamda_arr1),10),
                                            orientation='vertical')

ax.set_xlabel('$R$ (kpc)',fontsize=18)
ax.set_ylabel('$\\langle y \\rangle (R)$',fontsize=18)
fig.tight_layout()
fig.savefig('weighted_mean.png')



for c in clusters_groups:
	a,b = np.logspace(-21,-20,45),np.linspace(1,1.20,35)
	params_arr = np.array([[[ai,bi,1.177,1e3,0.30,5.49,1.051] for ai in a] for bi in b]).astype('object')
	I = c.stacked_halo_model([12,16],[c.lamda.min(),c.lamda.max()],[c.lamda_true.min(),c.lamda_true.max()],0,sigmaML=0.25,params_arr=params_arr)

	mean_prof = c.mean['means']
	err = c.mean['errors']
	sqxhi = np.array([[np.sum((mean_prof - I[i,j])**2) for i in range(I.shape[0])] for j in range(I.shape[1])])
	print(sqxhi.shape)
	best_fitting_indx = np.where(sqxhi == sqxhi.min())

	best_a = a[best_fitting_indx[0]]

	best_b = b[best_fitting_indx[1]]


	fig,(ax1,ax2) = plt.subplots(2,1)
	im = ax1.imshow(sqxhi , extent=[b[0],b[-1],a[0],a[-1]],origin='lower',norm=LogNorm())
	plt.colorbar(im,ax=ax1,label='fit error')
	ax1.set(yscale='log',xlabel='slope $b$',ylabel='normalization $A$',title='Distribution of $\\chi^2$')
	ax1.set_aspect('auto')
	ax1.scatter(best_b,best_a,marker='*',edgecolor='black',color='yellow',s=80)
	ax1.grid(True)
	R = c.R[1::]
	M = np.mean(10**14.45*(c.lamda/40)**1.29)
	best_a_arr.append(best_a)
	best_b_arr.append(best_b)
	mean_M.append(M)
	params = params_arr[0][0]
	ax2.plot(R,(best_a*M**best_b) /((params[2]*R/params[3])**params[4] * (1 + (params[2]*R/params[3])**params[6] )**( ( params[5] - params[4])/params[6])),label='best fitting',color='black',ls='--')
	ax2.errorbar(R,mean_prof,yerr=err,label='data',color='black',ecolor='black')
	ax2.scatter(R,mean_prof,color='black')
	ax2.set(ylabel='y',xlabel='R (kpc)',yscale='log',title=f'profile with $\\lambda \\in$ [{round(c.lamda.min())},{round(c.lamda.max())}]')
	ax2.grid(True)
	ax2.legend()
	fig.tight_layout()
	fig.savefig(f'stacked_halo_model_lamda_[{round(c.lamda.min())},round(c.lamda.max())].png')

fig,(ax1,ax2) = plt.subplots(2,1,figsize=(8,4))
ax1.scatter(mean_M,best_a_arr)
ax2.scatter(mean_M,best_b_arr)
fig.tight_layout()
fig.savefig('params_distribution.png')


print('\n\n\033[41m----- numerical derivates -----\033[00m\n\n')

def deriv3_datos(fxo, fx2, h):
    return (fx2 - fxo)/(2*h)

def deriv3_borde_datos(fxo, fx1, fx2, h):
    return (-3*fxo + 4*fx1 - fx2)/(2*h)

def deriv_datos(fx, h):
    n = len(fx)
    deriv = np.zeros(n)
    for i in range(n):
        if i == 0:
            deriv[i] = deriv3_borde_datos(fx[0], fx[1], fx[2], h)
        elif i == n - 1:
            deriv[i] = deriv3_borde_datos(fx[n-1], fx[n-2], fx[n-3], -h)
        else:
            deriv[i] = deriv3_datos(fx[i-1], fx[i+1], h)
    return deriv

r_rings = np.linspace(300,1000,6)*u.kpc
lamda,lamda_err,radial_pr,radial_pr_err,Z = radial_profiles(szmap,rmcat,r_rings,radii,1e-3,method_func=np.mean,)

small_clusters = radial_pr[lamda<50]
small_clusters = np.array([small_clusters[i]/small_clusters[i][0] for i in range(len(small_clusters))])
big_clusters = radial_pr[lamda>100]
big_clusters = np.array([big_clusters[i]/big_clusters[i][0] for i in range(len(big_clusters))])
h = r_rings[1] - r_rings[0]
mean_sc = np.mean(small_clusters,axis=0)
err_sc = np.std(small_clusters,axis=0)/np.sqrt(len(small_clusters))
mean_bc = np.mean(big_clusters,axis=0)
err_bc = np.std(big_clusters,axis=0)/np.sqrt(len(big_clusters))
fig,(ax,ax2) = plt.subplots(1,2,figsize=(8,4))
ax.scatter(r_rings[0:-1],mean_sc,color='blue',label='small clusters')
ax.errorbar(r_rings[0:-1],mean_sc,yerr=err_sc,color='blue')

ax.scatter(r_rings[0:-1],mean_bc,color='red',label='big clusters')
ax.errorbar(r_rings[0:-1],mean_bc,yerr=err_bc,color='red')

ax.legend()
ax.grid(True)

ax.set(xlabel='radius (kpc)',ylabel='$y_{sz}$',title='Profiles',yscale='log')

deriv_sc = r_rings[0:-1]/mean_sc * deriv_datos(mean_sc,h.value)
deriv_bc = r_rings[0:-1]/mean_bc * deriv_datos(mean_bc,h.value)

ax2.plot(r_rings[0:-1],deriv_sc,color='blue',label='small clusters')
ax2.plot(r_rings[0:-1],deriv_bc,color='red',label='red clusters')
ax2.set(xlabel='radius (kpc)',ylabel='$\\frac{\\log{y_{sz}}}{d\\log{r}}$',title='Derivates')
ax2.grid(True)
ax2.legend()
fig.tight_layout()
fig.savefig('sb.png')


"""
