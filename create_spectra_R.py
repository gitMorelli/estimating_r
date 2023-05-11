#!/usr/bin/env python
# coding: utf-8

#importo le librerie necessarie
import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import camb
import pandas as pd
from camb import model, initialpower

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
Nside=512
Nside_red=16
lmax=3*Nside_red-1
l_gen=4*Nside #genero i Cl fino a lgen e poi salvo solo i primi lmax. Il risultato è diverso rispetto a generare direttamente fino a lmax

import random
n_spectra=1100
r=np.linspace(0.0,0.06, n_spectra) #genero n_spectra spettri nell'intervallo
seed=10 #imposto un seme da cui partire per la generazione casuale in modo che il codice sia riproducibile
np.random.seed(seed)
np.random.shuffle(r)
tau=0.06
const=1.88 * 10**(-9)
to_iter=range(len(r)) #preparo l'array di indici da 0 a len(r)-1 per alleggerire la notazione

data=[]
for i in to_iter: 
    #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=67.32, ombh2=0.02237, omch2=0.1201, mnu=0.06, omk=0, tau=tau)
    pars.InitPower.set_params(As=const*np.exp(2*tau), ns=0.9651, r=r[i]) #setto il tensor to scalar ratio e As (calcolato da tau)
    pars.set_for_lmax(l_gen, lens_potential_accuracy=0)#imposto l fino a cui genereare i Cl (come detto prendo l_gen>l_max)
    pars.WantTensors=True #dico a camb che deve calcolare i Cl usando anche le perturbazioni tensoriali
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK') #get dictionary of CAMB power spectra; 
    #spectra are multiplied by l*(l+1)/2pi
    totCL=powers['total']
    d=[totCL[0:lmax,0],totCL[0:lmax,1], totCL[0:lmax,2], totCL[0:lmax,3]] #seleziono solo i Cl fino a l=l_max. In ordine d contiene C^TT 
    #C^EE C^BB C^TE
    data.append(d)
    if i%100==0:
        print("number: ",i) #ogni 100 spettri stampo un messaggio di output 

data=np.asarray(data)#converto i dati in un tensore numpy e li salvo in un file npz
np.savez("outfile_R_000_006_seed="+str(seed),data=data,r=r) #I Cl vanno sotto la label "data", Gli r che generano i Cl corrispondenti vanno
#sotto la label "r". 


