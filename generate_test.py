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

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
Nside=512
Nside_red=16
lmax=3*Nside_red-1
l_gen=4*Nside

const=1.88 * 10**(-9)
tau=0.06
r=0.06
seed_test=70
np.random.seed(seed_test)# i set a seed for the generation of the maps and the a_lm. I use a seed for reproducibility.
outfile_name="test_data_r00"+str(int(r*100))+"_t00"+str(int(tau*100))+"_"+str(seed_test)

pars.set_cosmology(H0=67.32, ombh2=0.02237, omch2=0.1201, mnu=0.06, omk=0, tau=tau)
pars.InitPower.set_params(As=const*np.exp(2*tau), ns=0.9651, r=r)
pars.set_for_lmax(l_gen, lens_potential_accuracy=0)#i generate the cl up to l_gen>>lmax
pars.WantTensors=True #i tell camb to compute the tensor perturbations to the maps

results = camb.get_results(pars)
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK',raw_cl=True)#spectra are multiplied by l*(l+1)/2pi
totCL=powers['total']
d=[totCL[0:lmax,0],totCL[0:lmax,1], totCL[0:lmax,2], totCL[0:lmax,3]] #i keep only the cl with l<lmax

def convert_to_df(totCL):
    #d={"l":ls}
    D={}
    D["TT"]=totCL[0]
    D["EE"]=totCL[1]
    D["BB"]=totCL[2]
    D["TE"]=totCL[3]
    df=pd.DataFrame(D)
    return(df)
d=convert_to_df(d)# i convert the list of spectra in a dataframe with TT,EE,BB,TE as indexes

n_test=10000
n_channels=2
high_nside = 512
low_nside= 16
window=hp.pixwin(low_nside,lmax=lmax) #i generate a window function 
res=hp.nside2resol(low_nside) 
beam=hp.gauss_beam(2*res, lmax=lmax) #i generate a gaussian beam window function with fwhm=2*res where res is the pixel size in radians
n_pix=hp.nside2npix(low_nside)
sensitivity=4 #muK-arcmin
mu, sigma = 0, sensitivity*np.deg2rad(1./60.)/res 
smooth=window*beam

mappe_B=np.zeros((n_test,n_pix,2)) #i prepare an array of 10000 pair of maps
for i in range(n_test):
    alm = hp.synalm((cl.TT, cl.EE, cl.BB, cl.TE), lmax=lmax, new=True) # i generate the alm from the cl
    alm_wb = np.array([hp.almxfl(each,smooth) for each in alm]) #i multiply the alm by the beam and window functions a_lm*Bl*w_l
    alm_B=alm_wb[2] #i take a_lm^B from the array (a_lm^T, a_lm^E, a_lm^B) 
    B_map=hp.alm2map(alm_B, nside=low_nside, pol=False, lmax=lmax)# i fourier transform the alm_B directly (pol=false tell alm2map that 
    #i am not giving it multiple maps to combine in T,Q,U maps)
    for j in range(0,2*n_channels,2): #i sum a different noise realization to each map in a set of two maps -> noise uncorrelated
        noise = np.random.normal(mu, sigma, n_pix) #i generate gaussian noise for each pixel of the map
        mappe_B[i,:,int(j/2)]=B_map+noise #i sum the noise directly on the map

np.savez(outfile_name,x_test=mappe_B,y_test=r) 

