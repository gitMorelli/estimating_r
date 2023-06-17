#!/usr/bin/env python
# coding: utf-8
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
from tensorflow import keras
from keras import metrics
from keras import layers
from keras import models
import pandas as pd
from loss_functions import sigma_loss, sigma2_loss,sigma_batch_loss,sigma_norm_loss,sigma_log_loss,mse_tau,mse_sigma, mse_batch, sigma_f_loss
import math
import useful_functions as uf
import NN_functions as nuf
import os, shutil

seed_train=400
np.random.seed(seed_train)# i set a random seed for the generation of the maps for reproducibility
tf.random.set_seed(seed_train)

#map gen
nside = 16
n_train=100 #the total number of training+validation pair of maps that i will generate
n_train_fix=100 #the total number of of training maps i will spread on all the r interval -> for each r value i generate n_train_fix/len(r) maps 
kind_of_map="BB"
n_channels=2
pol=1
res=hp.nside2resol(nside, arcmin=False) 
sensitivity=4

base_dir='/home/amorelli/pipeline/test_double/sigma/'
base_dir_tau='/home/amorelli/pipeline/test_double/'
test_model_folder="test_model"
# callbacks
reduce_lr_on_plateau = True
p_stopping=[20,20]
p_reduce=[5,5]
f_reduce=[0.5,0.5]
stopping_monitor="val_loss"
reduce_monitor="val_loss"
metrics=[]

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
max_epochs = [2,2]
lr=[0.001, 0.0003]
fval=0.1 # this is the fraction of data that i use for validation, computed on n_train_fix
loss_training=tf.keras.losses.MeanSquaredError() # this is the loss i use for the training
shuffle=True
norm=True
map_norm=False
batch_ordering=False
distr=0

f_ = np.load('/home/amorelli/cl_generator/outfile_R_000_001_seed=67.npz') 
#print("outfile_R:",f_.files) #give the keiwords for the stored arrays
labels=f_.files
data=f_[labels[0]]
r=f_[labels[1]]
r, data=uf.unison_sorted_copies(r, data)
indexes=np.linspace(0,len(r)-1,10,dtype=int)
r=r[indexes]
data=data[indexes]

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
#np.savez(base_dir+"check_r_distribution",y_train=y_train,y_val=y_val) 
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
f_train=np.load(base_dir_tau+"predictions.npz")
normalizer=f_train["norm"]
model_tau = keras.models.load_model(base_dir_tau+test_model_folder) 
predictions_train=model_tau.predict(x_train)
predictions_val=model_tau.predict(x_val)

if norm:
    y_train_sigma=(predictions_train-y_train)**2 * np.std(r)**2
    y_val_sigma=(predictions_val-y_val)**2 * np.std(r)**2
    count,red=uf.check_y(y_train_sigma)
    y_train_sigma=nuf.normalize_data(y_train_sigma,red)
    y_val_sigma=nuf.normalize_data(y_val_sigma,red)
else:
    y_train_sigma=(predictions_train-y_train)**2 
    y_val_sigma=(predictions_val-y_val)**2 
    red=None
model_sigma=model_tau
model_sigma.trainable=True
set_layer=False
for layer in model_sigma.layers:
    if layer.name == "flatten":
        set_layer=True
    if set_layer:
        layer.trainable = True
    else:
        layer.trainable = False
nuf.compile_and_fit(model_sigma, x_train, y_train_sigma, x_val, y_val_sigma, batch_size, max_epochs[0], stopping_monitor,p_stopping[0],
                    reduce_monitor,f_reduce[0], p_reduce[0],base_dir, 
                    loss_training,lr[0],metrics,shuffle=True,verbose=2,callbacks=[True,True,True,True],append=False)
model_sigma.trainable=True
set_layer=False
for layer in model_sigma.layers:
    if layer.name == "conv1d_3":
        set_layer=True
    if set_layer:
        layer.trainable = True
    else:
        layer.trainable = False
nuf.compile_and_fit(model_sigma, x_train, y_train_sigma, x_val, y_val_sigma, batch_size, max_epochs[1], stopping_monitor,p_stopping[1],
                    reduce_monitor,f_reduce[1], p_reduce[1],base_dir, 
                    loss_training,lr[1],metrics,shuffle=True,verbose=2,callbacks=[True,True,True,True],append=True)
predictions_sigma=model_sigma.predict(x_train)
np.savez(base_dir+"predictions",y_train=y_train_sigma, pred=predictions_sigma, norm=red)
model_sigma.save(base_dir+'test_model_sigma')