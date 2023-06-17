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

ncores=48
n_tot=10000
n_maps=n_tot-n_tot%ncores #so that is divisible
n_channels=2

seed_test=70
np.random.seed(seed_test)# i set a seed for the generation of the maps and the a_lm. I use a seed for reproducibility.


# In[ ]:
instrument = get_instrument('LiteBIRD')
sensitivities=instrument["depth_p"]
n_freq=len(sensitivities)
nside=16
n_pix=hp.nside2npix(nside)

sync_freq_maps = get_observation(instrument, 'd0s0', noise=False, nside=nside, unit='uK_CMB')

r=np.ones(1)*0.01
data=uf.generate_cl(n_spectra=1,Nside=512,Nside_red=nside,tau_interval=[0.06,0.06],r_interval=[0.01,0.01], raw=False)

beam_w=2*hp.nside2resol(nside, arcmin=False)

components = [CMB(), Dust(150.), Synchrotron(20.)]

noise_maps=np.ones(shape=(n_channels,n_freq,n_maps,n_pix))
#noise_maps_T=np.ones(shape=(n_freq,n_maps,n_pix))
for i,s in enumerate(sensitivities):
    for k in range(n_channels):
        noise=uf.generate_noise(n_maps,s*np.sqrt(2),nside) #i make the product with sqrt(2) because i am considering half of the time of observation (recall how S/N scales with time of observation)
        noise_maps[k,i]=noise
    #noise_T=uf.generate_noise(n_maps,s/np.sqrt(2),nside)
    #noise_maps_T[i]=noise_T

QU_maps=uf.generate_maps(data=data, r=r,n_train=n_maps,nside=16, n_train_fix=0, beam_w=beam_w, kind_of_map="QU", raw=1 , 
                         distribution=0, n_channels=n_channels, sensitivity=0,beam_yes=1 , verbose=0)[0]
#T_maps=uf.generate_maps(data=data, r=r,n_train=n_maps,nside=16, n_train_fix=0, 
                        #beam_w=beam_w, kind_of_map="TT", raw=1 , distribution=0, n_channels=1, 
                        #sensitivity=0,beam_yes=1 , verbose=0)[0]

freq_maps=np.ones(shape=(n_channels,n_maps,n_freq,2,n_pix))
for j in range(n_maps):
    for i in range(n_freq):
        #freq_maps[j,i,0]=noise_maps_T[i,j]+sync_freq_maps[i,0]+T_maps[j,:,0]
        #freq_maps[j,i,1]=noise_maps[i,j]+sync_freq_maps[i,1]+QU_maps[j,:,0]
        #freq_maps[j,i,2]=noise_maps[i,j]+sync_freq_maps[i,2]+QU_maps[j,:,1]
        for k in range(n_channels):
            freq_maps[k,j,i,0]=noise_maps[k,i,j]+sync_freq_maps[i,1]+QU_maps[j,:,k*n_channels]
            freq_maps[k,j,i,1]=noise_maps[k,i,j]+sync_freq_maps[i,2]+QU_maps[j,:,k*n_channels+1]
#salva le mappe di frequenza già in ncore file distinti
home_dir="/home/amorelli/temporary_storage/"
indexes=np.linspace(0,n_maps-1, ncores, dtype=np.int32)
for k in n_channels:
    for i in indexes:
        np.savez(home_dir+"ncore="+str(k)+"_seed="+str(seed_test),freq_Q1=freq_maps[0,:,0,:],freq_U1=result[0,:,1,:],
                 freq_Q2=freq_maps[1,:,0,:],freq_U2=result[1,:,1,:])

