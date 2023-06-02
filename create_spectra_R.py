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
import useful functions as uf
import random

Nside=512
Nside_red=16
n_spectra=1100
r=[0,0.06]
seed=10 #imposto un seme da cui partire per la generazione casuale in modo che il codice sia riproducibile
np.random.seed(seed)
tau=[0.06,0.06]
const=1.88 * 10**(-9)

data=uf.generate_cl(n_spectra,Nside,Nside_red,tau_interval,r_interval,raw=0,verbose=1)

np.savez("outfile_R_000_006_seed="+str(seed),data=data,r=r) #I Cl vanno sotto la label "data", Gli r che generano i Cl corrispondenti vanno
#sotto la label "r". 


