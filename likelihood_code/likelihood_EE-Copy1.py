#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import camb
import pandas as pd
import healpy as hp
from camb import model, initialpower

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
Nside_red=16
lmax=3*Nside_red-1
#pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
#pars.InitPower.set_params(As=2e-9, ns=0.965, r

const=1.88 * 10**(-9)
tau=0.06
# In[4]:
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.32, ombh2=0.02237, omch2=0.1201, mnu=0.06, omk=0, tau=tau)
pars.InitPower.set_params(As=const*np.exp(2*tau), ns=0.9651, r=0)
pars.set_for_lmax(lmax, lens_potential_accuracy=0)
results = camb.get_results(pars)
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK',raw_cl=False)#spectra are multiplied by l*(l+1)/2pi
totCL=powers['total']
ls = np.arange(totCL.shape[0])
cl_obs=np.asarray(totCL[0:lmax,1])
ell = np.arange(0,lmax+1)
print(totCL[0:lmax,0])


# In[2]:


f_ = np.load('outfile_0.npz')
#https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html if i have multiple npz files

print(f_.files) #give the keywords for the stored arrays
data=f_["data"]
tau=f_["tau"]
print(data.shape)
print(data[:15,0,2])


# In[11]:


high_nside = 512
low_nside= 16
window=hp.pixwin(low_nside,lmax=lmax)
res=hp.nside2resol(low_nside, arcmin=False) #false give output in radians
beam=hp.gauss_beam(2*res, lmax=lmax)
res_low=hp.nside2resol(low_nside, arcmin=False) 
res_high=hp.nside2resol(high_nside, arcmin=False)
n_pix=hp.nside2npix(low_nside)
print(res_low,res_high,n_pix)
sensitivity=4 #muK-arcmin
mu, sigma = 0, sensitivity*low_nside/high_nside 
Nl=(sigma*hp.nside2resol(512))**2
print(res_low,hp.nside2resol(16))
N_2=(4*np.deg2rad(1.0/60.0))**2
#print(beam.shape)
print(Nl,N_2)
print(Nl**0.5,N_2**0.5)
print(np.deg2rad(440.0/60.0), 2*res)


# In[4]:


#i prepare the data
print(cl_obs)
print(data[0,0,:])
def normalize_cl(input_cl):
    output_cl=np.zeros(len(input_cl))
    for i in range(1,len(input_cl)):
        output_cl[i]=input_cl[i]/i/(i+1)*2*np.pi
    return output_cl
    


# In[5]:


all_cl=np.zeros((len(tau)+1, lmax))
all_cl[0]=normalize_cl(cl_obs)*beam[:lmax]**2+Nl#*ell[:lmax]*(ell[:lmax]+1)/2/np.pi
for i in range(1,len(tau)+1):
    d=data[i-1,1,:]
    all_cl[i]=normalize_cl(d)*beam[:lmax]**2+Nl#*ell[:lmax]*(ell[:lmax]+1)/2/np.pi
print(all_cl.shape)
#print(tau[:2],all_cl[:3])


# In[6]:


def compute_likelihood(c_obs,c_th,l): #cl_th wants all the cl for all the tau
    cl_obs=c_obs[l]
    cl_th=c_th[:,l]
    cl_obs*=1#*l*(l+1)/2/np.pi
    logL=np.zeros(len(cl_th))
    const=30
    #L_l becomes L_l*e^const
    for i,cl in enumerate(cl_th):
        #print(cl_obs,cl)
        #print(cl)
        cl_r=cl#*l*(l+1)/2/np.pi
        #print(cl_r)
        logL_i=-(2*l+1)/2*((cl_obs/cl_r)+np.log(np.abs(cl_r)))+(2*l+1)/2*(1+np.log(np.abs(cl_obs)))
        #print(np.log(np.abs(cl_r)),logL_i)
        #logL_i=-(2*l+1)/2*((cl/cl_obs)+np.log(np.abs(cl_obs)))
        logL[i]=logL_i
    return(logL)
logL=0
l=lmax
#c=10**90
for l in ell[2:l]:
    logL+=compute_likelihood(all_cl[0],all_cl[1:],l)
L_l=np.exp(logL)
print(logL)


# In[7]:


plt.plot(tau,L_l,color='r',linestyle='None',marker='.', markersize = 1.0)


# In[8]:


#L_l=(L_l-np.mean(L_l))/np.std(L_l)
#plt.plot(tau,L_l,color='r',linestyle='None',marker='.', markersize = 1.0)
from scipy.integrate import trapz, simps
def unison_sorted_copies(a, b):
    assert len(a) == len(b)
    p = a.argsort()
    return a[p], b[p]
tau, L_l=unison_sorted_copies(tau,L_l)
print(tau)
#print(tau)
A=trapz(L_l, tau)
print(A)


# In[9]:


L_l=L_l/A
print(trapz(L_l, tau))


# In[10]:


mean=trapz(L_l*tau, tau)
print(mean)


# In[11]:


var=trapz(L_l*(tau-mean)**2,tau)
print(var)


# In[12]:


sigma=var**0.5
print(sigma)


# In[ ]:




