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
r=[(0.0,0.0)]
n_test=10000
n_per_r=100
r_array=np.linspace(0.01,0.13,int(n_test/n_per_r))
tau=[(x,x) for x in r_array]
n_channels=2
pol=1 #2
res=hp.nside2resol(Nside_red) 
sensitivity=4 #muK-arcmin

seed_test=411
np.random.seed(seed_test)# i set a seed for the generation of the maps and the a_lm. I use a seed for reproducibility.
outfile_name="test_data_r_0"+"_t00"+str(int(tau[0][0]*100))+"_"+str(seed_test)
outfile_name="test_data_r00"+str(int(r[0][0]*100))+"_t001_013_"+str(seed_test)

input_folder="/home/amorelli/foreground_noise_maps/noise_maps_d0s0_test/"
input_files=os.listdir(input_folder)
for j in range(len(input_files)):
    input_files[j]=input_folder+"/"+input_files[j]
maps_per_cl_gen=uf.maps_per_cl(distribution=0)
maps_per_cl=maps_per_cl_gen.compute_maps_per_cl([tau[0][0]],n_train=n_per_r,n_train_fix=n_per_r)

y_test=np.ones((n_test,1))
n_pix=hp.nside2npix(Nside_red)
x_test=np.ones((n_test,n_pix,pol*n_channels))

for i in range(len(tau)):
    data=uf.generate_cl(n_spectra=1,Nside=Nside,Nside_red=Nside_red,tau_interval=tau[i],r_interval=r[0],raw=1,verbose=0)
    noise_maps=uf.generate_noise_maps(n_train=n_per_r,n_channels=n_channels,nside=Nside_red,pol=2,
                                              sensitivity=sensitivity,input_files=input_files)
    noise_E,noise_B=uf.convert_to_EB(noise_maps)

    mappe,y_tau=uf.generate_maps(data, r=[tau[i][0]],n_train=n_per_r,nside=Nside_red, beam_w=2*res, noise_maps=noise_E,
                                map_per_cl=maps_per_cl, kind_of_map="EE", raw=1 , n_channels=n_channels,beam_yes=1 , verbose=0)
    y_test[i*n_per_r:(i+1)*n_per_r]=y_tau
    x_test[i*n_per_r:(i+1)*n_per_r]=mappe
masking=True
if masking:
    x_test=uf.mask_it(x_test,path="/home/amorelli/HFI_Mask_GalPlane-apo0_2048_R2.00.fits",
                  field=1,nside_low=nside,nside_high=2048)
    #masks: 0=20% , 1=40% , 2=60%, 3=70, 4=80, 5=90, 6=97, 7=99 20% means that 20% of sky is visible
np.savez(outfile_name,x_test=x_test,y_test=y_test) 
