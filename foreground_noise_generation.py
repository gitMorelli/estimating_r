#!/usr/bin/env python
# coding: utf-8

# In[25]:


#!/usr/bin/env python
# coding: utf-8

import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import camb
import pandas as pd
import healpy as hp
from camb import model, initialpower
import useful_functions as uf
import pysm3
from fgbuster import (CMB, Dust, Synchrotron, basic_comp_sep,get_observation, get_instrument)
from fgbuster.visualization import corner_norm
seed_test=70
np.random.seed(seed_test)# i set a seed for the generation of the maps and the a_lm. I use a seed for reproducibility.


# In[ ]:


import time
start_time = time.time()


# In[26]:


instrument = get_instrument('LiteBIRD')
sensitivities=instrument["depth_p"]


# In[27]:


n_maps=10
n_freq=len(sensitivities)
nside=16
n_pix=hp.nside2npix(nside)


# In[28]:


noise_maps=np.ones(shape=(n_freq,n_maps,n_pix))
noise_maps_T=np.ones(shape=(n_freq,n_maps,n_pix))
for i,s in enumerate(sensitivities):
    noise=uf.generate_noise(n_maps,s,nside)
    noise_T=uf.generate_noise(n_maps,s/np.sqrt(2),nside)
    noise_maps[i]=noise
    noise_maps_T[i]=noise_T


# In[29]:


sync_freq_maps = get_observation(instrument, 'd0s0', noise=False, nside=nside, unit='uK_CMB')


# In[30]:


r=np.ones(1)*0.01
data=uf.generate_cl(n_spectra=1,Nside=512,Nside_red=nside,tau_interval=[0.06,0.06],r_interval=[0.01,0.01], raw=False)


# In[31]:


beam_w=2*hp.nside2resol(nside, arcmin=False)
QU_maps=uf.generate_maps(data=data, r=r,n_train=n_maps,nside=16, n_train_fix=0, beam_w=beam_w, kind_of_map="QU", raw=1 , 
                         distribution=0, n_channels=1, sensitivity=0,beam_yes=1 , verbose=0)[0]
T_maps=uf.generate_maps(data=data, r=r,n_train=n_maps,nside=16, n_train_fix=0, 
                        beam_w=beam_w, kind_of_map="TT", raw=1 , distribution=0, n_channels=1, 
                        sensitivity=0,beam_yes=1 , verbose=0)[0]
#recall T_maps are [n_maps,n_pix,n_channels]


# In[32]:


freq_maps=np.ones(shape=(n_maps,n_freq,3,n_pix))
for j in range(n_maps):
    for i in range(n_freq):
        freq_maps[j,i,0]=noise_maps_T[i,j]+sync_freq_maps[i,0]+T_maps[j,:,0]
        freq_maps[j,i,1]=noise_maps[i,j]+sync_freq_maps[i,1]+QU_maps[j,:,0]
        freq_maps[j,i,2]=noise_maps[i,j]+sync_freq_maps[i,2]+QU_maps[j,:,1]


# In[33]:


components = [CMB(), Dust(150.), Synchrotron(20.)]


# In[34]:


result = np.ones(shape=(n_maps,3,n_pix))
for i in range(n_maps):
    result[i]=basic_comp_sep(components, instrument, freq_maps[i]).s[0]


# residuals = np.ones(shape=(n_maps,3,n_pix))
# for i in range(n_maps):
#     residuals[i,0]=result[i,0]-T_maps[i,:,0]
#     residuals[i,1]=result[i,1]-QU_maps[i,:,0]
#     residuals[i,2]=result[i,2]-QU_maps[i,:,1]

# res=hp.nside2resol(nside, arcmin=True)
# for i in range(3):
#     print("std:",np.std(residuals[0,i])*res)
#     print("(other map)std:",np.std(residuals[2,i])*res)
# for i in range(3):
#     print("mean:",np.mean(residuals[0,i])*res)
#     print("(other map)mean:",np.mean(residuals[2,i])*res)

# bin_edges= np.histogram_bin_edges(residuals[0,2], bins='fd')
# print("n_of_bins:",len(bin_edges))
# counts_tau, bins_tau = np.histogram(residuals[0,2], bins=bin_edges)
# plt.stairs(counts_tau,bins_tau)

# In[55]:


print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:


np.savez("test_10_maps",noise_Q=residuals[:,1,:],noise_U=residuals[:,2,:])

