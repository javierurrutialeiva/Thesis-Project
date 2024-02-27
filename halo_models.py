
from multiprocessing import Pool
import emcee
import os
import corner
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from extract_clusters import rings
from scipy.optimize import curve_fit
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import simps, trapz
import pyccl as ccl
from scipy.signal import convolve2d as conv2d


def random_params_space(param,ndims,nwalkers,ptype=['log'],limits=[[-9,-6]]):
	"""
	Create from a NxM array a random space of parameters.

	parameters:

	- param: array with shape M with the initial parameters.
	- ndims: dimension of the params (must match with the lenght of parameter).
	- nwalkers: number N of walkers.
	- ptype: a list with M elemenets. The element must be str and equal to 'log' (logarithm) or 'lin' (linear)
	- limits: an array with shape Mx2. The first element establishes the lower limit, the second establishes the upper limit.

	"""
	param = np.tile(param,nwalkers).reshape(-1,ndims)
	for i in range(np.shape(param)[0]):
		for j in range(np.shape(param)[1]):
			if ptype[j]=='log':
				param[i][j] = 10**np.random.uniform(limits[j][0],limits[j][1])
			else:
				p = np.random.uniform(limits[j][0],limits[j][1])
				param[i][j] = p
	return np.array(param).astype(np.float64)

def projected_GNFW_individual(R,M,params):
	"""
	projected GNFW from Nagai et al 2007
	"""
	P0,gamma,beta,alpha,c500,rc = params
	R_los = np.logspace(-7,9,200)[:,None]
	R = np.transpose([np.hypot(*np.meshgrid(R_los[:,i],R[:])) for i in range(R_los.shape[1])],axes=(1,2,0))
	profile = 2 * simps(GNFW(R,M,P0,gamma,beta,alpha,rc,c500),R_los[None],axis=1)[:,0]
	return profile

def GNFW(r,M,P0,gamma,beta,alpha,rc,c500):
	rc = 10**rc
	P0 = 10**P0
	return (P0)/(((c500*r/rc)**gamma)*(1 + (c500*r/rc)**alpha)**((beta-gamma)/alpha))


def optimal_lamda_range(lamda,profiles,errors,lamda_min,lamda_max,h_lamda,ratio=5,SNd=None,SN=None,full_sample=False,N_min=100,round_to_five=True):
	"""
	Finds the best possible bins of richness.
	"""

	if full_sample==True:
		lamdaSN = np.array([])
		lamdaSN_arr = np.array([])
	lamda_cut = np.where((lamda<lamda_max) & (lamda>=lamda_min))
	lamda_unsorted = lamda[lamda_cut]
	profiles_unsorted = profiles[lamda_cut]
	errors_unsorted = errors[lamda_cut]
	lamda = np.sort(lamda)
	sorted_indices = np.argsort(lamda_unsorted)
	profiles = profiles_unsorted[sorted_indices]
	errors = errors_unsorted[sorted_indices]
	intervals = [np.round(lamda_min)]
	selected_profiles = []
	selected_errors = []
	SNr_profiles = []
	rounded_lamda = np.round(lamda)
	unique_lamda = np.unique(rounded_lamda)
	total_SNr = 0
	for i in range(0,len(unique_lamda)-1):
		SNr_profiles = []
		current_lamda = unique_lamda[i]
		lamda_cut = np.where((rounded_lamda > intervals[-1]) & (rounded_lamda <= current_lamda))
		selected_profiles = profiles[lamda_cut]
		selected_errors = errors[lamda_cut]
		for j in range(len(selected_profiles)):
			current_SNr = np.sqrt(np.sum([ selected_profiles[j][k]**2/selected_errors[j][k]**2 for k in range(len(selected_errors[j]))]))
			if np.isnan(current_SNr)==True:
				SNr_profiles.append(0)
			else:
				SNr_profiles.append(current_SNr)
		total_SNr = np.mean(SNr_profiles)
		if total_SNr >= SN and len(SNr_profiles)>=N_min:
			intervals.append(current_lamda)
	intervals.append(np.round(np.max(lamda)))
	return intervals
	if round_to_five==True:
		intervals_to5 = []
		for i in range(len(intervals)):
			x = intervals[i]
			if x%5==0:
				intervals_to5.append(x)
			else:
				higher = x + (5 - x%5)
				lower = x - (5 - x%5)
				d1 = np.abs(x - higher)
				d2 = np.abs(x - lower)
				if d1<d2:
					intervals_to5.append(higher)
				elif d1>d2:
					intervals_to5.append(lower)
				elif d1==d2:
					intervals_to5.append(higher)

		intervals=intervals_to5
	if full_sample==True:
		return np.unique(intervals),lamdaSN_arr,lamdaSN,(SN,SNd)
	else:
		return np.unique(intervals)

def radial_profiles(szmap,redmapper,r_rings,radii,pixel_size,save=False,method_func=np.median,norm_func=None):
	lamda = []
	rings_data = []
	err_rings_data = []
	lamda_err = []
	redshift = []
	if norm_func==None:
		print(f'running without normalized radii')
		norm_func = lambda x: 1
	else:
		print(f'running with normalized radii')
	for i in tqdm(range(len(szmap))):
		lamda0 = redmapper[i]['LAMBDA_CHISQ']
		lamda_err0 = redmapper[i]['LAMBDA_CHISQ_E']
		lamda_err.append(lamda_err0)
		lamda.append(lamda0)
		redshift.append(redmapper[i]['Z'])
		ID = redmapper[i]['MEM_MATCH_ID']
		r_norm = norm_func({'lamda':lamda[-1],'h':0.67})
		r_rings_norm = r_rings*r_norm
		rings_data0,err_rings_data0 = rings(r_rings_norm,radii[i],szmap[i],pixel_size=pixel_size,title=f'redmapperID={ID}.jpg',ID=ID,save=save,method_func=method_func,z=redmapper[i]['Z'])
		rings_data.append(rings_data0)
		err_rings_data.append(err_rings_data0)
	return np.array(lamda),np.array(lamda_err),np.array(rings_data),np.array(err_rings_data),np.array(redshift)

def meanl(lr,l,data):
	indx = np.where((l>=lr[0]) & (l<=l[1]))
	mean = np.mean(data[indx[0]])
	err = np.std(data[indx[0]])/np.sqrt(data[indx[0]].size)
	return mean , err

def projected_radius(params,h=0.67):
	lamda = params['lamda']
	h = params['h']
	return 1.0*(lamda/100)**0.2/h*u.kpc*1000


class cluster_data():
	def __init__(self,profiles,radii,lamda,sigma,redshift,R_u='kpc',fit_methods = ['least-square','MCMC','MarquardtLevenberg','grid_chi2_reduction'],
			lamda_relationship='Costanzi2019',scale_type='richness-mass',int_method='trapz',stacked_model='GNFW',model='stacked_halo_model',mass_range=None):
		self.prof = np.array(profiles)
		self.R = np.array(radii)
		self.R_u = R_u
		self.z = np.array(redshift)
		self.lamda = np.array(lamda)
		self.s = np.array(sigma)
		self.fit_methods = fit_methods
		self.fit_datax = []
		self.fit_datay = []
		self.sigma_scale = 0.25
		self.int_method = 'trapz'
		if stacked_model=='GNFW':
			self.stacked_model = projected_GNFW_individual
		else:
			self.model = model
		if model=='stacked_halo_model':
			self.model = self.stacked_halo_model
		self.mass_range = np.log10(np.array([10**(14.45)*(lamda.min()/40)**1.29,
						10**(14.45)*(lamda.max()/40)**1.29]))
		print('\n',100*'=')
		print('\033[96mA new clusters/halos data was created with:\033[00m')
		print(f'N clusters: \033[91m{len(self.prof)}\033[00m')
		print(f'Profile shapes: \033[91m{np.shape(self.prof[0])}\033[00m')
		print(f'Lamda range: \033[91m[{np.round(np.min(self.lamda))}-{np.round(np.max(self.lamda))}]\033[00m')
		print(f'Radius interval: \033[91m[{np.round(np.min(self.R))}-{np.round(np.max(self.R))}]\033[00m {self.R_u}')
		print(f'{scale_type} relationship: \033[91m{lamda_relationship}\033[00m')
		z_main = np.mean(self.z)
		self.z_main = z_main
		print(f'z = {z_main}')
		print(f'Available fit methods:')
		for m in self.fit_methods:
			print(f'\t*{m}.')
		self.scale_func = self.mass_lambda_relationship(lamda_relationship,scale_var='mass')
		self.var_scale_func = lambda x: np.sqrt( (10**scale_func(x) - 1)/(10**(scale_func(x))**2) + self.sigma_scale**2)
		self.prob_lambda(lamda_range=[np.min(self.lamda),np.max(self.lamda)],z=z_main,plot=False)
		print(100*'=','\n')
		#self.plot_lambda_relationship()
	def weight_mean(self,weighted=False):
		if weighted==True:
			print('Calculating weighted mean.')
			weights = 1/(self.s)
			mean = []
			err = []
			for i in range(len(self.prof[0])):
				w = weights[:,i]
				x = self.prof[:,i]
				wx = np.sum([w[j]*x[j] for j in range(len(x))])
				wp = np.sum(w)
				mean.append(np.average(x,weights=w))
				err.append(np.std(self.prof[:,i])/np.sqrt(np.size(self.prof[:,i])))
				#err.append(np.sqrt(1/p))
			self.mean = {'means':mean,'errors':err}
		else:
			print('Calculating unweighted mean.')
			mean = []
			err = []
			for i in range(len(self.prof[0])):
				mean.append(np.mean(self.prof[:,i]))
				err.append(np.std(self.prof[:,i])/np.sqrt(self.prof[:,i].size))
			self.mean = {'means':mean,'errors':err}

	def plot_weight_mean(self,ax=None,fig=None,log=True,savefig=False,title='',
                         label=False,filename='figure.png',color='black',ls='-'):
		if ax==None or fig==None:
			fig = plt.figure(figsize=(12,8))
			ax = plt.axes()
		ax.errorbar(self.R[1::],self.mean['means'],yerr=self.mean['errors'],label=f'$\\lambda \\in $ [{np.round(np.min(self.lamda))},{np.round(np.max(self.lamda))}]',ecolor=color,color=color,ls=ls,capsize=4)
		ax.set(xlabel=f'R {self.R_u}',ylabel='$y$ pressure profile',title=title)
		ax.grid(True)
		if log==True:
			ax.set(yscale='log')
		if label==True:
			ax.legend()
		if savefig==True:
			fig.savefig(filename)
	def fit(self,method='least-square',kwargs={}):
		model = self.model
		if method in self.fit_methods:
			print(f'fitting data with: \033[96m{method}\033[00m')
			if len(self.fit_datax)!=0 and len(self.fit_datay)!=0:
				if method=='least-square':
					x = self.fit_datax
					y = self.fit_datay
					try:
						s = self.sigma
						print(f'err data is added to fitting. shape:',np.shape(s))
					except:
						s = None
						print(f'\033[96mrunning fit without err data.\033[00m')
					try:
						fit_p = curve_fit(model,x,y,sigma=s,**kwargs)[0]
					except:
						fit_p = kwargs['p0']
					self.fit_datay2 = model(x,*fit_p)
					self.params = fit_p
				elif method=='MarquardtLevenberg':
					x = self.fit_datax
					y = self.fit_datay
					fit_p = MarquardtLevenberg(x,y,model=model,**kwargs)
					self.fit_datay2 = model(x,fit_p)
					self.params = fit_p
				if method=='MCMC':
					x_data = self.fit_datax
					y_data = self.fit_datay
					y_err = np.array(self.sigma)
					init_params = np.array(kwargs['init_params'])
					if ('nwalkers' in list(kwargs.keys()))==False:
						nwalkers = 40
					else:
						nwalkers = kwargs['nwalkers']
					ndims = len(init_params)
					if ('nsteps' in list(kwargs.keys()))==False:
						nsteps = 500
					else:
						nsteps = kwargs['nsteps']
					if ('ln_likelihood' in list(kwargs.keys()))==False:
						def ln_likelihood(theta,x,y,sigma):
							y2 = model(theta,x)
							res = y - y2
							chi_squared = np.sum((res / y_err)**2)
							return -0.5 * chi_squared
					elif kwargs['ln_likelihood']=='gaussian':
						def ln_likelihood(theta,x,y,sigma):
						        mu = model(theta,x)
        						log_likelihood = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((y - mu)**2) / (sigma**2)
        						return np.sum(log_likelihood)


					else:
						ln_likelihood = kwargs['ln_likelihood']
					if ('ln_prior' in list(kwargs.keys()))==False:
						def ln_prior(theta):
							if (-1<theta[0]<1):
								return 0.0
							else:
								return -np.inf
					else:
						ln_prior = kwargs['ln_prior']
					if ('ln_posterior' in list(kwargs.keys()))==False:
						def ln_posterior(theta,x,y,sigma,**kwargs):
							prior = ln_prior(theta)
							if np.isinf(prior):
								return prior
							likelihood = ln_likelihood(theta,x,y,sigma)
							posterior = prior + likelihood
							if np.isnan(posterior)==True:
								return -1e12
							else:
								return prior + likelihood
					else:
						ln_posterior = kwargs['ln_posterior']

					param_limits = [[-6,-4],[-7,7],[-7,7],[-7,7],[0.4,2],[1,4]]
					init_params = random_params_space(init_params,ndims,nwalkers,ptype=['lin','lin','lin','lin','lin','lin'],limits=param_limits)
					if ('with_pool' in list(kwargs.keys()))==False:
						with_pool = False
					else:
						with_pool = True

					if with_pool==True:
						with Pool(40) as pool:
							sampler = emcee.EnsembleSampler(nwalkers, ndims, ln_posterior, args=(x_data, y_data, y_err), pool=pool)
							sampler.run_mcmc(init_params, nsteps, progress=True)
					else:
						sampler = emcee.EnsembleSampler(nwalkers, ndims, ln_posterior, args=(x_data, y_data, y_err))
						sampler.run_mcmc(init_params, nsteps, progress=True)
					samples = sampler.get_chain(flat=True)
					ln_posterior_values = np.array([ln_posterior(params, x_data, y_data, y_err) for params in samples])
					best_index = np.argmax(ln_posterior_values)
					best_parameters = samples[best_index]
					self.best_fitting = best_parameters
					self.MCMCsamples = samples
					self.MCMCsampler = sampler
					np.save(f'samples_[{np.round(np.min(self.lamda))}-{np.round(np.max(self.lamda))}].npy',samples)
					np.save(f'ln_posterior_[{np.round(np.min(self.lamda))}-{np.round(np.max(self.lamda))}].npy',ln_posterior_values)
				if method=='stacked_halo_model':
					x = self.fit_datax
					y = self.fit_datay
					s = np.array(self.sigma)
					params = kwargs['params']
					free_params = kwargs['free_params']
					R = self.R[1:]
					chi2 = np.zeros(np.shape(params)[0])
					if 'mass_range' in list(kwargs.keys()):
						M_range = kwargs['mass_range']
					else:
						M_range = [12,16]
					for i in tqdm(range(len(params))):
						p0 = params[i]
						chi2v = np.sum((y - self.stacked_halo_model(R,params=p0,model=model))**2/s**2)
						if np.isnan(chi2v)==False:
							chi2[i] = chi2v
						else:
							chi2[i] = 1e3
					indx = np.argmin(chi2)
					best_fitting = params[indx]
					self.best_fitting = params[indx]
					self.fit_params = params
					self.chi2 = chi2
	def write_mean_in(self,path='/',file='output.txt',mode='a'):
		print(f"Saving data in {path}{file} with {mode} mode...")
		entire_path = "{path}{file}"
		lamda_interval = np.array([np.min(self.lamda),np.max(self.lamda)])
		if hasattr(self,'mean'):
			profile_data = self.mean['means']
			radius = self.R
			sigma = self.mean['errors']
			if (file in os.listdir(path))==False:
				mode = 'w'
			with open(file,mode) as f:
				f.write(f"lamda={lamda_interval},z={self.z_main}:\n")
				f.write(f"\t\tradius: {radius}\n")
				f.write(f"\t\ty_profile: {profile_data}\n")
				f.write(f"\t\tsigma: {sigma}\n")
				f.close()
		else:
			print("The mean attr its empty!")
	def prob_lambda(self,file='/data2/cristobal/actpol/lensing/cmblensing/des/selection/completeness_des.txt',lamda_range=[20,300],plot=False,z=0.1):
		print(f'\nCreating selection function with z=\033[91m {np.round(z,4)} \033[00m and lambda in \033[91m[{np.round(lamda_range[0])},{np.round(lamda_range[1])}]\033[00m')
		df = pd.read_csv(file,delimiter='|',usecols=[1,2,3,4]).to_numpy()
		lamda_obs = df[:,2]
		lamda_selected = np.where((lamda_obs <=lamda_range[1]) & (lamda_obs >= lamda_range[0]))
		prob_func = df[lamda_selected[0]]
		redshift = prob_func[:,0]
		closter_redshift = np.unique(redshift)[np.argmin(np.abs(np.unique(redshift) - z))]
		prob_func = prob_func[redshift == closter_redshift]
		lamda_true = prob_func[:,1]
		self.lamda_true = lamda_true
		prob = []
		for i in range(len(lamda_true)):
			sl = prob_func[prob_func[:,1]==lamda_true[i]]
			prob.append(sl[:,-1])
		self.prob_lambda = prob
		if plot==True:
			fig = plt.figure()
			ax = plt.axes()
			im = ax.imshow(np.log(np.abs(np.array(prob))),interpolation='nearest',origin='lower',extent=(lamda_true.min(),lamda_true.max(),lamda_range[0],lamda_range[1]))
			plt.colorbar(im,label='prob.',ax=ax)
			z_round = ''.join(list(str(z)))[0:6]
			ax.set(title=f'Probability function $P(\lambda_o|\lambda_t)$at z={z_round} and $\\lambda\\in$[{np.round(lamda_range[0])},{np.round(lamda_range[1])}]',ylabel='$\\lambda_{obs}$', xlabel='$\\lambda_{true}$')
			yticks = np.round(np.linspace(self.lamda.min(),self.lamda.max(),8))
			xticks = np.round(np.linspace(lamda_true.min(),lamda_true.max(),8))
			ax.xaxis.set_ticks(xticks)
			ax.yaxis.set_ticks(yticks)
			ax.set_aspect('auto')
			fig.savefig(f'joint_probability_at_z={str(np.round(z,4)).replace(".",",")}.png')
			plt.close()
	def mass_func_and_bias(self,type='critical',masses=None,nmass=20,delta=500,fiducial=True,cosmo_params=None,save=True):
		if fiducial==True:
			cosmo_params = dict(Omega_c=0.25,Omega_b=0.05,h=0.7,n_s=0.95,sigma8=0.8)
		elif fiducial==False and cosmo_params==None:
			raise Exception('You must define cosmo_params before.')
		cosmo = ccl.Cosmology(**cosmo_params)
		if len(masses)==0:
			masses = 10**np.logspace(12,16,nmass)
		mdef = ccl.halos.massdef.MassDef(delta,type)
		a = 1/(1 + self.z_main)
		R = np.array([mdef.get_radius(cosmo,M,a) for M in masses])
		bias_tinker = ccl.halos.hbias.tinker10.HaloBiasTinker10(mdef)
		hbias = np.array([bias_tinker(cosmo,M,a) for M in masses])
		mfunc = ccl.halos.mass_function_from_name('Tinker10')
		mfunc = mfunc(cosmo,mdef)
		dndlog10M = np.array([mfunc(cosmo,M,a) for M in masses])
		if save==True:
			np.savetxt('dndlog10M.txt',dndlog10M,delimiter=' ,')
		return dndlog10M,hbias
	def mass_lambda_relationship(self,author='McClintock2019',lamda_pivot=40,params=None,scale_var='richness',mass_pivot=3e14):
		if params==None:
			file = '/data2/javierurrutia/des/Y1_redmapper/richness_mass_relationships.csv'
			relationships = pd.read_csv(file).to_numpy()
			try:
				if scale_var=='richness':
					self.scale_func_paper = author
					a,aerr,b,berr = relationships[np.where(relationships[:,0]==author)[0]][0][1:5].astype(float)
					self.scale_type = 'mass-richness'
					pivot=lamda_pivot
				elif scale_var=='mass':
					self.scale_func_paper = author
					self.scale_type = 'richness-mass'
					a,aerr,b,berr = relationships[np.where(relationships[:,0]==author)[0]][0][5:].astype(float)
					pivot=mass_pivot
				return lambda x: (a + b*(np.log10(x) - np.log10(pivot)),np.full(np.shape(x),np.abs(aerr - berr)))
			except:
				if scale_var=='richness':
					print(f'\033[91mFatal error!\033[00m trying to create mass-richness relationship. Now trying with default:\033[96m McClintock2019 \033[00m')
					self.scale_func_paper = 'McClintock2019'
					self.scale_type = 'mass-richness'
					a,aerr,b,berr = relationships[np.where(relationships[:,0]=="McClintock2019")[0]][0][1:5].astype(float)
				elif scale_var=='mass':
					print(f'\033[91mFatal error!\033[00m trying to create mass-richness relationship. Now trying with default:\033[96m Costanzi2019 \033[00m')
					self.scale_func_paper = 'Costanzi2019'
					self.scale_type = 'richness-mass'
					a,aerr,b,berr = relationships[np.where(relationships[:,0]=="Costanzi2019")[0]][0][5:].astype(float)
				return lambda x: (a + b*(np.log10(x) - np.log10(lamda_pivot)),np.full(np.shape(x),np.abs(aerr - berr)))
		else:
			return lambda x: params[0] + params[1]*(np.log10(x) - np.log10(params[2]))
	def plot_lambda_relationship(self,labels={'richness': '\\lambda', 'mass': 'M_{\\odot}'}):
		if True==True:
			if self.scale_type.split('-')[-1]=='mass':
				var_arr = 10**(np.linspace(self.mass_range[0],self.mass_range[1]))
			elif self.scale_type.split('-')[-1]=='richness':
				var_arr = np.linspace(self.lamda.min(),self.lamda.max())
			res,err = self.scale_func(var_arr)
			res = 10**res
			err = 10**err
			ylabel,xlabel = [labels[i] for i in self.scale_type.split('-')]
			ylabel = '$\\log10{'+ylabel+'}$'
			xlabel = f'${xlabel}$'
			fig = plt.figure(figsize=(8,5))
			ax = plt.axes()
			ax.errorbar(var_arr,res,yerr=err)
			ax.set(yscale='log',xscale='log',title=f'{self.scale_type} scale relation from {self.scale_func_paper}',xlabel=xlabel,ylabel=ylabel)
			ax.grid(True)
			fig.savefig(f'{self.scale_type}-{self.scale_func_paper} [{np.round(self.lamda.min())},{np.round(self.lamda.max())}].png')
			plt.close()
	def miscellanea(self,M_range,lamda_obs_range=None,method='trapz',ndims=(20),sigmaML = 0.25):
		P_true = np.array(self.prob_lambda).transpose() #P(lambda_obs|lambda_true)

		if type(lamda_obs_range)==type(None):
			lambda_obs = np.linspace(np.min(self.lamda),np.max(self.lamda),np.shape(P_true)[0]).astype(int)
		else:
			lambda_obs = np.linspace(lamda_obs_range[0],lamda_obs_range[1],ndims[1])
		mass_arr = np.logspace(M_range[0],M_range[1],ndims[0])
		lambda_true = self.lamda_true
		lambda_true,mass = np.meshgrid(lambda_true,mass_arr)
		lambda_model = 10**np.array(self.scale_func(mass))[0]
		P_mass = 1/(np.sqrt(2*np.pi*sigmaML)) * np.exp(-(np.log(lambda_true) - np.log(lambda_model))**2/(2*sigmaML**2))
		product = P_true[:,:,None] * P_mass.transpose()
		P_obs = trapz(product,x=self.lamda_true,axis=1)
		P_MR = trapz(P_obs,x=lambda_obs,axis=0)
		dndM,Mbias = self.mass_func_and_bias(masses=mass_arr)
		self.P_MR = P_MR
		self.dndM = dndM

	def stacked_halo_model(self,params,R):
		M_range = self.mass_range
		ndims = 20
		sigmaML = self.sigma_scale
		model = self.stacked_model
		method = self.int_method
		P_MR = self.P_MR
		dndM = self.dndM
		mass_arr = np.logspace(M_range[0],M_range[1],len(dndM)).astype(np.float64)
		y_model = np.array([model(R,M,params) for M in mass_arr]).astype(np.float64)
		if method=='trapz':
			I = trapz(dndM[:,None] * (P_MR[:,None] * y_model),x=mass_arr,axis=0)
		elif method=='simpson':
			I = simpson(dndM[:,None] * (P_MR[:,None] * y_model),x=mass_arr,axis=0)
		norm = trapz(dndM * P_MR,x=mass_arr,axis=0)
		mean_profile = I/norm
		return np.array(mean_profile).astype(np.float64)

def deriv_parcial(modelo,x,params,delta):
	deriv=np.zeros((len(x),len(delta)))
	for param_num in range(0,len(delta)):
		params_nuevo2=np.array(params);
		params_nuevo2[param_num]=np.array(params[param_num])+np.array(delta[param_num])
		temp1=np.array(modelo(x,params_nuevo2))
		temp2=np.array(modelo(x,params))
		deriv[:,param_num]=(temp1-temp2)/delta[param_num]  
	return np.array(deriv)


def MarquardtLevenberg(x,y,model,params0,h=0.0001,n_iter_max=100,err=1e-5,Marquardt_init=0.1,n_iterMarLevMax=200,MarLevRad=2):
	n_iteraciones=0;
	params=np.array(params0);
	residuos1 = np.sum((y - model(x,params))**2);
	residuos2 = residuos1/2
	while (residuos2+1e-6)/(residuos1+1e-6)<(1-err) and n_iteraciones<n_iter_max:
		Marquardt = Marquardt_init
		residuos1 = np.sum((y - model(x,params))**2);
		delta = params*h;
		dev = deriv_parcial(model, x, params, delta);
		B_vector = []
		for j in range(len(params)):
			B_vector.append(np.sum((y-model(x,params))*dev[:,j]))
			A_array = np.zeros((len(params),len(params)))
		for i in range(len(params)):
			for k in range(len(params)):
				A_array[i][k]= np.sum(dev[:,i]*dev[:,k]);
		n_iteraciones+=1;
		A=A_array + np.identity(len(params))*Marquardt
		delta_lambda = np.linalg.inv(A)@B_vector
		params_nuevo= params+delta_lambda
		residuos2= np.sum((y - model(x,params_nuevo))**2)
		n_iterMarLev=0;
		while residuos2>residuos1 and n_iterMarLev < n_iterMarLevMax:
                #exigimos que el valor de los residuales siempre disminuya
			Marquardt=Marquardt*MarLevRad
			A=A_array+np.identity(len(params))*Marquardt   
			delta_param=np.linalg.inv(A)@B_vector
			params_nuevo=params+delta_param
			residuos2=np.sum((y-model(x,params_nuevo))**2)
			n_iterMarLev+=1
		params=params_nuevo
	return params

