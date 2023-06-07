#!/usr/bin/env python
# coding: utf-8

#i import the necessary libraries
import sys
import h5py
import numpy as np
import healpy as hp
import tensorflow as tf
import random as python_random
import nnhealpix.layers
from tensorflow.keras import metrics
import pandas as pd
from loss_functions import sigma_loss, sigma2_loss,sigma_batch_loss,sigma_norm_loss,sigma_log_loss,mse_tau,mse_sigma, mse_batch, sigma_f_loss
import math
import useful_functions as uf
import NN_functions as nuf
import os, shutil

seed_train=33
np.random.seed(seed_train)# i set a random seed for the generation of the maps for reproducibility

#map gen
nside = 16
n_train=100000 #the total number of training+validation pair of maps that i will generate
n_train_fix=100000 #the total number of of training maps i will spread on all the r interval -> for each r value i generate n_train_fix/len(r) maps 
kind_of_map="EE"
n_channels=2
pol=1
res=hp.nside2resol(nside, arcmin=False) 
sensitivity=4

name='7_6_23'
base_dir='/home/amorelli/QU_foreground_tau/'+name+'/'
# callbacks
reduce_lr_on_plateau = True
p_stopping=20
p_reduce=5
f_reduce=0.5
stopping_monitor="val_loss"
reduce_monitor="val_loss"
metrics=[sigma_loss, sigma_batch_loss,mse_tau,mse_sigma, sigma_f_loss, mse_batch]# these are the different loss functions i have used. I use them as metrics

#network structure
one_layer=True # this is to switch between one dense layer or two dense layer
drop=0.2
n_layer_0=48
n_layer_1=64
n_layer_2=16
if kind_of_map!="QU": 
    n_inputs=n_channels
else:
    n_inputs=pol*n_channels

#train and val
batch_size = 16
max_epochs = 200
lr=0.0003 
fval=0.1 # this is the fraction of data that i use for validation, computed on n_train_fix
training_loss="mse_tau+sigma_tau"
loss_training_tau=mse_tau # this is the loss i use for the training
loss_training_sigma=mse_sigma # this is the loss i use for the training
shuffle=False

f_ = np.load('/home/amorelli/cl_generator/outfile_l_47_complete.npz') 
#print("outfile_R:",f_.files) #give the keiwords for the stored arrays
labels=f_.files
data=f_[labels[0]]
r=f_[labels[1]]
r, data=uf.unison_sorted_copies(r, data)
#indexes=np.linspace(0,len(r)-1,10,dtype=int)
#r=r[indexes]
#data=data[indexes]

input_folder="/home/amorelli/foreground_noise_maps/noise_generation"
input_files=os.listdir(input_folder)
for i in range(len(input_files)):
    input_files[i]=input_folder+"/"+input_files[i]
noise_maps=uf.generate_noise_maps(n_train,n_channels,nside,pol=2,sensitivity=sensitivity,input_files=input_files)

noise_E,noise_B=uf.convert_to_EB(noise_maps)

maps_per_cl_gen=uf.maps_per_cl(distribution=1)
maps_per_cl=maps_per_cl_gen.compute_maps_per_cl(r,n_train,n_train_fix)

mappe_B,y_r=uf.generate_maps(data, r,n_train=n_train,nside=nside, map_per_cl=maps_per_cl, 
                             noise_maps=noise_E, beam_w=2*res, kind_of_map=kind_of_map, raw=0 , n_channels=n_channels,beam_yes=1 , verbose=0)

x_train,y_train,x_val,y_val = nuf.prepare_data(y_r,mappe_B,r,n_train,n_train_fix,fval,maps_per_cl
                                               , batch_size, batch_ordering=True)

np.savez(base_dir+"check_r_distribution",y_train=y_train,y_val=y_val) 

model=nuf.build_network_tau(n_inputs,nside,drop,n_layer_0,n_layer_1,n_layer_2,one_layer)

history=nuf.compile_and_fit(model, x_train, y_train, x_val, y_val, batch_size, max_epochs, 
                            stopping_monitor,p_stopping,reduce_monitor,f_reduce, p_reduce,base_dir, 
                            loss_training,lr,metrics,shuffle=shuffle, verbose=2)

model.save(base_dir+'test_model_tau')

predictions_tau=model.predict(mappe_B)

y_sigma=(predictions-y_r)**2

x_train,y_train,x_val,y_val = nuf.prepare_data(y_sigma,mappe_B,r,n_train,n_train_fix,fval,maps_per_cl
                                               , batch_size, batch_ordering=False)
#-----------------------------------------
hyperparameters={}
hyperparameters["name"]=name
hyperparameters["loss"]=training_loss
hyperparameters["noise"]=sensitivity
hyperparameters["p_stopping"]=p_stopping
hyperparameters["p_reduce"]=p_reduce
hyperparameters["f_reduce"]=f_reduce
hyperparameters["stop-reduce"]=stopping_monitor+"-"+reduce_monitor
hyperparameters["lr"]=lr
hyperparameters["batch_size"]=batch_size
hyperparameters["n_layers"]=one_layer
if one_layer:
    hyperparameters["nodes_layers"]=n_layer_0
else:
    hyperparameters["nodes_layers"]=str(n_layer_1)+"-"+str(n_layer_2)
hyperparameters["comments"]=" "
hyperparameters = {k:[v] for k,v in hyperparameters.items()}
output=pd.DataFrame(hyperparameters)
output.to_csv(base_dir+'output.txt', index=False, sep=' ')