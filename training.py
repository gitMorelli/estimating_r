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

seed_train=40
np.random.seed(seed_train)# i set a random seed for the generation of the maps for reproducibility

#map gen
nside = 16
n_train=100000 #the total number of training+validation pair of maps that i will generate
n_train_fix=100000 #the total number of of training maps i will spread on all the r interval -> for each r value i generate n_train_fix/len(r) maps 
kind_of_map="BB"
n_channels=2
pol=1
res=hp.nside2resol(nside, arcmin=False) 
sensitivity=4

name='results_12c_6_23'
base_dir='/home/amorelli/r_estimate/B_maps_white_noise/'+name+'/'
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
n_output=2

#train and val
batch_size = 16
max_epochs = 200
lr=0.0003 
fval=0.1 # this is the fraction of data that i use for validation, computed on n_train_fix
training_loss="sigma_batch_loss"
loss_training=sigma_batch_loss # this is the loss i use for the training
shuffle=False
norm=True
map_norm=True
batch_ordering=True
distr=0

f_ = np.load('/home/amorelli/cl_generator/outfile_R_000_001_seed=67.npz') 
#print("outfile_R:",f_.files) #give the keiwords for the stored arrays
labels=f_.files
data=f_[labels[0]]
r=f_[labels[1]]
r, data=uf.unison_sorted_copies(r, data)
#indexes=np.linspace(0,len(r)-1,10,dtype=int)
#r=r[indexes]
#data=data[indexes]

#input_folder="/home/amorelli/foreground_noise_maps/noise_generation"
#input_files=os.listdir(input_folder)
#for i in range(len(input_files)):
   # input_files[i]=input_folder+"/"+input_files[i]
noise_maps=uf.generate_noise_maps(n_train,n_channels,nside,pol=1,sensitivity=sensitivity,input_files=None)

#noise_E,noise_B=uf.convert_to_EB(noise_maps)

maps_per_cl_gen=uf.maps_per_cl(distribution=distr)
maps_per_cl=maps_per_cl_gen.compute_maps_per_cl(r,n_train,n_train_fix)

mappe_B,y_r=uf.generate_maps(data, r,n_train=n_train,nside=nside, map_per_cl=maps_per_cl, 
                             noise_maps=noise_maps, beam_w=2*res, kind_of_map=kind_of_map, 
                             raw=0 , n_channels=n_channels,beam_yes=1 , verbose=0)


x_train,y_train,x_val,y_val = nuf.prepare_data(y_r,mappe_B,r,n_train,n_train_fix,fval,maps_per_cl
                                               , batch_size, batch_ordering=batch_ordering)

if norm:
    y_train=nuf.normalize_data(y_train,r)
    y_val=nuf.normalize_data(y_val,r)
np.savez(base_dir+"check_r_distribution",y_train=y_train,y_val=y_val) 
#rand_indexes=np.random.randint(0,len(y_train)-1,10000)
#np.savez(base_dir+"check_train_maps",y_train=y_train[rand_indexes], x_train=x_train[rand_indexes])

if map_norm:
    for i in range(len(x_train)):
        for j in range(n_inputs):
            x=x_train[i,:,j]
            x_train[i,:,j]=nuf.normalize_data(x,x)
    for i in range(len(x_val)):
        for j in range(n_inputs):
            x=x_val[i,:,j]
            x_val[i,:,j]=nuf.normalize_data(x,x)

model=nuf.build_network(n_inputs,nside,drop,n_layer_0,n_layer_1,n_layer_2,one_layer,
                        num_output=n_output,use_normalization=[False,False,False],use_drop=False)

history=nuf.compile_and_fit(model, x_train, y_train, x_val, y_val, batch_size, max_epochs, 
                            stopping_monitor,p_stopping,reduce_monitor,f_reduce, p_reduce,base_dir, 
                            loss_training,lr,metrics,shuffle=shuffle, verbose=2,callbacks=[True,True,True,True])

print('Saving model to disk')
model.save(base_dir+'test_model')

predictions=model.predict(x_train)

np.savez(base_dir+"predictions",y_train=y_train, pred=predictions, norm=r)

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
