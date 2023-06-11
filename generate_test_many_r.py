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
tau=[(0.06,0.06)]
n_test=10000
n_per_r=100
r_array=np.linspace(0,0.01,int(n_test/n_per_r))
r=[(x,x) for x in r_array]
n_channels=2
pol=1 #2
res=hp.nside2resol(Nside_red) 
sensitivity=4 #muK-arcmin

seed_test=40
np.random.seed(seed_test)# i set a seed for the generation of the maps and the a_lm. I use a seed for reproducibility.
outfile_name="test_data_r_0_001"+"_t00"+str(int(tau[0][0]*100))+"_"+str(seed_test)

input_folder="/home/amorelli/foreground_noise_maps/noise_generation"
input_files=os.listdir(input_folder)
for j in range(len(input_files)):
    input_files[j]=input_folder+"/"+input_files[j]
maps_per_cl_gen=uf.maps_per_cl(distribution=0)

y_test=np.ones((n_test,1))
n_pix=hp.nside2npix(Nside_red)
x_test=np.ones((n_test,n_pix,pol*n_channels))

for i in range(len(r)):
    data=uf.generate_cl(n_spectra=1,Nside=Nside,Nside_red=Nside_red,tau_interval=tau[0],r_interval=r[i],raw=1,verbose=0)
    noise_maps=uf.generate_noise_maps(n_train=n_per_r,n_channels=n_channels,nside=Nside_red,pol=2,
                                              sensitivity=sensitivity,input_files=input_files)
    noise_E,noise_B=uf.convert_to_EB(noise_maps)
    maps_per_cl=maps_per_cl_gen.compute_maps_per_cl([r[i][0]],n_train=n_per_r,n_train_fix=n_per_r)

    mappe,y_tau=uf.generate_maps(data, r=[r[i][0]],n_train=n_per_r,nside=Nside_red, beam_w=2*res, noise_maps=noise_B,
                                map_per_cl=maps_per_cl, kind_of_map="BB", raw=1 , n_channels=n_channels,beam_yes=1 , verbose=0)
    y_test[i*n_per_r:(i+1)*n_per_r]=y_tau
    x_test[i*n_per_r:(i+1)*n_per_r]=mappe
np.savez(outfile_name,x_test=x_test,y_test=y_test) 
