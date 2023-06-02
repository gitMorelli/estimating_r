#!/usr/bin/env python
# coding: utf-8

import sys
import h5py
import numpy as np
import healpy as hp
import random as python_random
import pandas as pd
import camb
#from loss_functions import sigma_loss, sigma2_loss,sigma_batch_loss,sigma_norm_loss,sigma_log_loss,mse_tau,mse_sigma, mse_batch, sigma_f_loss
import math
import random

def generate_cl(n_spectra,Nside,Nside_red,tau_interval,r_interval,raw=0,verbose=0):
    ''' generate a matrix of cl^TT cl^EE cl^BB cl^TE. 
    input
        Nside: the high_nside i use to generate the Cl
        Nside_red: the low_nside that determines the lmax of the map
        n_spectra: the number of spectra to generate
        r_or_tau: 
            0 use tau to generate the Cls
            1 use r to generate the Cls
        verbose:
            0 do nothing
            1 it prints something every 100 spectra generated
        raw
            0 return spectra multiplied by l(l+1)/2pi
            1 return raw spectra
        output:
            a matrix of the kind Cl^TT Cl^EE Cl^BB Cl^TE
            the output cl are not normalized
        
    '''
    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    lmax=3*Nside_red-1
    l_gen=4*Nside #genero i Cl fino a lgen e poi salvo solo i primi lmax. Il risultato è diverso rispetto a generare direttamente fino a lmax
    if r_interval[0]==r_interval[1]:
        tau=np.linspace(tau_interval[0],tau_interval[1], n_spectra) #genero n_spectra spettri nell'intervallo
        r=np.ones(len(tau))*r_interval[0]
    else:
        r=np.linspace(r_interval[0],r_interval[1], n_spectra) #genero n_spectra spettri nell'intervallo
        tau=np.ones(len(r))*tau_interval[0]
    const=1.88 * 10**(-9)
    to_iter=range(n_spectra) #preparo l'array di indici da 0 a len(r)-1 per alleggerire la notazione
    data=[]
    
    for i in to_iter: 
        #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
        pars.set_cosmology(H0=67.32, ombh2=0.02237, omch2=0.1201, mnu=0.06, omk=0, tau=tau[i])
        pars.InitPower.set_params(As=const*np.exp(2*tau), ns=0.9651, r=r[i]) #setto il tensor to scalar ratio e As (calcolato da tau)
        pars.set_for_lmax(l_gen, lens_potential_accuracy=0)#imposto l fino a cui genereare i Cl (come detto prendo l_gen>l_max)
        pars.WantTensors=True #dico a camb che deve calcolare i Cl usando anche le perturbazioni tensoriali
        results = camb.get_results(pars)
        if raw==1:
            powers = results.get_cmb_power_spectra(pars, CMB_unit='muK',raw_cl=True) #get dictionary of CAMB power spectra; 
        else:
            powers = results.get_cmb_power_spectra(pars, CMB_unit='muK',raw_cl=False) #get dictionary of CAMB power spectra;
        #spectra are multiplied by l*(l+1)/2pi
        totCL=powers['total']
        d=[totCL[0:lmax,0],totCL[0:lmax,1], totCL[0:lmax,2], totCL[0:lmax,3]] #seleziono solo i Cl fino a l=l_max. In ordine d contiene C^TT 
        #C^EE C^BB C^TE
        data.append(d)
        if i%100==0 and verbose==1:
            print("number: ",i) #ogni 100 spettri stampo un messaggio di output 

    data=np.asarray(data)#converto i dati in un tensore numpy e li salvo in un file npz
    return data

def unison_sorted_copies(a, b):#this function sort the first array and reorder the second so that the pairing between elements
    #is maintained
    assert len(a) == len(b)
    p=np.argsort(a,axis=0)
    a_out=np.empty_like(a)# this intermediate step is necessary to prevent shape change in the arrays
    b_out=np.empty_like(b)
    for i in range(len(a)):
        a_out[i]=a[p[i]]
        b_out[i]=b[p[i]]
    return a_out, b_out

def adjust_map_per_cl(num_maps,n_train):
    to_remove=np.sum(num_maps)-n_train #the actual n_train is bigger than the initial n_train because i use the ceil function to round the numbers
    #print(to_remove)
    if(to_remove !=0 ):
        to_remove_abs=np.abs(to_remove)
        sign=to_remove/to_remove_abs
        indexes=np.random.randint(0,len(num_maps),to_remove_abs) 
        for i in indexes:
            num_maps[i]-=1*sign
        
    return num_maps

class maps_per_cl:
    def __init__(self,distribution):
        self.distribution=distribution
    def compute_maps_per_cl(self,r,n_train,n_train_fix):
        map_per_cl=np.empty_like(r)
        if self.distribution==0:
            map_per_cl=self.map_per_cl_uniform(r,n_train)
        elif self.distribution==1:
            #print(n_train,n_train_fix)
            map_per_cl=self.map_per_cl_linear(r,n_train,n_train_fix) #i generate map_per_cl for every value of r
            #in the compute_map function
        return map_per_cl
    def map_per_cl_linear(self,r,n,n_fix): # this function determine how many maps i need to generate for each Cl based on n_train and n_train_fix
        #from each cl n_train_fix/n_cl maps are generated at least. The other n_train-n_train_fix are spread on the interval so that more maps
        #are generated as we go from r_half to r=r_max or r=r_min. I use a linear relation map_per_cl=m*r+q
        #print(n,n_fix)
        n_cl=len(r) #the number of cls i use to generate the maps
        dr=r[1]-r[0] #difference between subcessive r values
        r_mean=r[-1]/2#middle r of the interval
        #r_temp=np.abs(r-r_mean)
        #closest=np.sort(r_temp)[0]+r_mean
        #r_mean=closest
        s_min=math.ceil(n_fix/n_cl) #the number of maps generated for Cl(r=r_half), this is the minimum number of maps generated for a Cl
        z=r_mean*n_cl/2 + dr*(n_cl/2)*(n_cl/2+1)/2
        m=(n/2-s_min*(n_cl/2+1))/(z-n_cl/2*r_mean) #formula for the angular coefficient of the formula that generates the maps_per_cl
        q=s_min-m*r_mean 
        num_maps=np.empty_like(r,dtype=np.int32)
        q_1=q+2*m*r_mean #for generation of maps_per_cl on the left of the middle value i need to change the sign of m and compute a new q based
        #on r_half and q
        for i,x in enumerate(r):
            if x>=r_mean: #for points on the right of r_half i use one formula, for points on the left the symmetric formula
                num_maps[i]=math.ceil(m*x+q)
            else:
                num_maps[i]=math.ceil(-m*x+q_1)
        num_maps=adjust_map_per_cl(num_maps,n)
        return(num_maps)

    def map_per_cl_uniform(self,r,n):
        size=n/len(r)
        num_maps=np.empty_like(r,dtype=int)
        for i in range(len(num_maps)):
            num_maps[i]=np.ceil(size)
        num_maps=adjust_map_per_cl(num_maps,n)
        return(num_maps)

def normalize_cl(input_cl): #this is the function to divide each cl of a given spectra by l(l+1)/2pi
    output_cl=np.zeros(len(input_cl)) # i prepare the output array
    for i in range(1,len(input_cl)):
        output_cl[i]=input_cl[i]/i/(i+1)*2*np.pi # i divide each element by l(l+1)/2pi
    return output_cl

def unison_shuffled_copies(a, b): # this function shuffle the first array and reorder the second so that elements that had same index
    #before the shuffling have same indexes after
    assert len(a) == len(b)
    p = np.random.permutation(len(a)) #this give a random permutation of the indexes of a
    return a[p], b[p] 

def generate_noise_maps(n_train,n_channels,nside,kind_of_map,input_file=None,sensitivity=0):
    '''
    assume input_file is an npz file with a single array and that the input dimension is n_maps, n_pix,n_channels
    if noise is QU noise last dimension is 2*n_channels, i have noiseQ1 noiseU1 noiseQ2 etc..'''
    low_nside= nside
    n_pix=hp.nside2npix(low_nside)
    if input_file==None:
        noise=np.zeros((n_train,n_pix,n_channels))
        res=hp.nside2resol(low_nside, arcmin=False)
        mu, sigma = 0, sensitivity*np.deg2rad(1./60.)/res
        for i in range(n_train):
            for k in range(n_channels):
                noise[i,:,k]=np.random.normal(mu, sigma, n_pix)
        return noise
    else:
        f_ = np.load(input_file) 
        label=f_.files[0]
        data=f_[label]
        noise=np.zeros((n_train,n_pix,data.shape[-1])) #same shape of input file but n_train fixed by user
        n_input=len(data)
        if n_input<n_train:
            diff=n_train-n_input
            indexes=np.random.randint(0,n_input,diff)
            noise[:n_input]=data
            for i,ind in enumerate(indexes):
                noise[n_input+i]=data[ind]
            p = np.random.permutation(n_train) 
            noise=noise[p]
        elif n_input>n_train:
            indexes=np.random.randint(0,n_input,n_train)
            noise=data[indexes]
        else:
            noise=data
        return noise

def generate_maps(data, r,n_train,nside, beam_w, noise_maps, map_per_cl, kind_of_map="TT", raw=0 , n_channels=1,beam_yes=1 , verbose=0):
    ''' You can use this function to generate a custom number of maps for each input Cl. You can generate maps with noise, beam and window function. You can generate EE,BB, TT, QU maps.
    input:
        data: expects a matrix of cl (cl^TT,cl^EE,cl^BB,cl^ET) 
        raw:
            0 -> the input data are Cl to be normalized
            1 -> input is already normalized
        r: a one dimensional array that contains the parameters used to generate the cl_in
        n_train: the total number of maps to generate
        n_pix: the dimension of the maps
        kind of map:
            "BB" for BB map
            "EE" for EE map
            "QU" for Q and U maps
        n_channels: the number of different noise realization of the same map
        sensitivity: the sigma of the noise in uK-arcmin
        beam_yes
            1 if you want beam and window function
            0 if you don't
        beam_w: the dimension of the beam in radians
        window:
        verbose:
            0 it prints out nothing
            1 it prints out the dimensions of the output array
    returns: 
        list element 0: an array of maps ordered in increasing value of the parameter that you vary to generate the maps
        list element 1: the sorted array of parameters that generated the maps
    '''
    high_nside = 512
    low_nside= nside
    lmax=len(data[0,0,:])
    window=hp.pixwin(low_nside,lmax=lmax) 
    beam=hp.gauss_beam(beam_w, lmax=lmax)
    n_pix=hp.nside2npix(low_nside)
    smooth=window*beam
    
    r, data=unison_sorted_copies(r, data)
    if kind_of_map!="QU":
        mappe=np.zeros((n_train,n_pix,n_channels)) # i prepare an array of n_train pairs of output maps
    else:
        mappe=np.zeros((n_train,n_pix,2*n_channels)) # i prepare an array of n_train pairs of output maps
    y_r=np.zeros((n_train,1)) # i prepare the array for the corresponding output r
    previous=0 #this is an index to keep track of the position in mappe_B as i generate the maps for the CL
    #i need an index cause i generate a variable number of maps for each Cl
    for i,cl_in in enumerate(data): # i iterate on the input spectra
        cl=np.empty_like(cl_in)
        if raw==0:
            for j in range(len(cl_in)):
                cl[j]=normalize_cl(cl_in[j]) # i normalize the cl
        else:
            cl=cl_in
        for k in range(map_per_cl[i]): #i have map_per_cl maps to generate for each spectra -> i iterate on the same spectra map_per_cl times
            index=previous+k #this is the index (in mappe_B) of the kth map that i am generating for this spectra
            y_r[index]=r[i]
            alm = hp.synalm((cl[0], cl[1], cl[2], cl[3]), lmax=lmax, new=True) # i generate the a_lm. In order the cl[i] are TT,EE,BB,TE
            
            if beam_yes==0:
                smooth=np.ones(len(smooth))
            alm_wb = np.array([hp.almxfl(each,smooth) for each in alm]) #i multiply the alm by the window functions
            if kind_of_map=="BB":
                alm_scalar=alm_wb[2]#i select the alm^B
            elif kind_of_map=="EE":
                alm_scalar=alm_wb[1]#i select the alm^E
            elif kind_of_map=="TT":
                alm_scalar=alm_wb[0]#i select the alm^T
            
            if kind_of_map!="QU":
                mappa=hp.alm2map(alm_scalar, nside=low_nside, pol=False, lmax=lmax)#i make the harmonic transform to obtain the B map
            else:
                mappa=hp.alm2map(alm_wb, nside=low_nside, pol=True, lmax=lmax)
            
            for j in range(0,2*n_channels,2): # i generate two maps+noise (two independent channels) from each of the map_per_cl B_maps
                #the map is the same for the channels, the noise is different (same magnitude)
                if kind_of_map!="QU":
                    mappe[index,:,int(j/2)]=mappa+noise_maps[index,:,int(j/2)]
                else:
                    mappe[index,:,j]=mappa[1]+noise_maps[index,:,j]
                    mappe[index,:,j+1]=mappa[2]+noise_maps[index,:,j+1]
        previous+=map_per_cl[i] #after i have generated all the maps for a Cl i add to the index the number of maps i have 
        #generated for that Cl
    return mappe,y_r

def convert_to_EB(mappe_Q, mappe_U):
    ''' it expects mappeQ to be of shape n_maps,n_pix,n_channels'''
    n_maps, n_pix, n_channels = (mappe_Q[0].shape[k] for k in range(3))
    nside=hp.npix2nside(npix)
    print(n_maps,n_pix,nside,n_channels)
    E_maps=np.zeros((n_maps,n_pix,n_channels)) # i prepare an array of n_train pairs of output maps
    B_maps=np.zeros((n_maps,n_pix,n_channels)) # i prepare an array of n_train pairs of output maps
    mappe_placeholder=np.zeros((n_maps,n_pix))
    for i in range(n_maps):
        for k in range(n_channels):
            alm=hp.map2alm([mappe_placeholder[i,:],mappe_Q[i,:,k],mappe_U[i,:,k]], pol=True, verbose=True)
            E_maps[i,:,k]=hp.alm2map(alm[1], nside, pol=False, verbose=True)
            B_maps[i,:,k]=hp.alm2map(alm[2], nside, pol=False, verbose=True)
    return E_maps, B_maps
def check_y(y_train):
    y_train=np.sort(y_train,axis=0)
    y_count=[]
    y_red=[]
    prev_index=0
    for i in range(1,len(y_train)):
        if y_train[i] != y_train[i-1]:
            y_count.append(i-prev_index)
            prev_index=i
            y_red.append(y_train[i-1])
        else:
            pass
    return y_count, y_red 