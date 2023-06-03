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

Nside=512
Nside_red=16
tau=[(0.0,0.0),(0.05,0.05),(0.06,0.06),(0.07,0.07)]
r=(0.0,0.0)
n_test=2
n_channels=2
pol=2
res=hp.nside2resol(Nside_red) 
sensitivity=4 #muK-arcmin

for i in range(len(tau)):
    seed_test=70+i
    np.random.seed(seed_test)# i set a seed for the generation of the maps and the a_lm. I use a seed for reproducibility.
    outfile_name="test_data_r00"+str(int(r[0]*100))+"_t00"+str(int(tau[i][0]*100))+"_"+str(seed_test)
    
    data=uf.generate_cl(n_spectra=1,Nside=Nside,Nside_red=Nside_red,tau_interval=tau[i],r_interval=r,raw=1,verbose=0)
    
    noise_maps=uf.generate_noise_maps(n_train=n_test,n_channels=n_channels,nside=Nside_red,pol=pol,sensitivity=sensitivity,input_files=None)
    
    maps_per_cl_gen=uf.maps_per_cl(distribution=0)
    maps_per_cl=maps_per_cl_gen.compute_maps_per_cl([tau[i][0]],n_train=n_test,n_train_fix=n_test)
    
    mappe_QU,y_tau=uf.generate_maps(data, r=[tau[i][0]],n_train=n_test,nside=Nside_red, beam_w=2*res, noise_maps=noise_maps,
                                    map_per_cl=maps_per_cl, kind_of_map="QU", raw=1 , n_channels=n_channels,beam_yes=1 , verbose=0)
    
    np.savez(outfile_name,x_test=mappe_QU,y_test=y_tau) 
