#!/usr/bin/env python
# coding: utf-8
#!/usr/bin/env python
# coding: utf-8

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

seed_train=4
np.random.seed(seed_train)# i set a random seed for the generation of the maps for reproducibility
tf.random.set_seed(seed_train)

#map gen
nside = 16
n_train=100000 #the total number of training+validation pair of maps that i will generate
n_train_fix=100000 #the total number of of training maps i will spread on all the r interval -> for each r value i generate n_train_fix/len(r) maps 
kind_of_map="EE"
n_channels=2
pol=1
res=hp.nside2resol(nside, arcmin=False) 
sensitivity=4

base_dir='/home/amorelli/E_foreground/29_6_23/'
base_dir_tau='/home/amorelli/E_foreground/25_6_23/'
test_model_folder="test_model"

# callbacks
p_stopping=20
p_reduce=5
f_reduce=0.5
stopping_monitor="val_loss"
reduce_monitor="val_loss"
metrics=[]#
metrics_tau=[sigma_loss, sigma_batch_loss,mse_tau,mse_sigma, sigma_f_loss, mse_batch]

#network structure
drop=[0.2,0.2,0.2]
activation_dense="relu"
kernel_initializer="glorot_uniform"
use_drop=[False,True,True]
use_normalization=[False,False,False]
n_layers=1
nodes_per_layer=[64,256,256]
if kind_of_map!="QU": 
    n_inputs=n_channels
else:
    n_inputs=pol*n_channels
n_output=1
n_output_tau=2

#train and val
#train and val
batch_size = 16
max_epochs = 200
lr=0.0001 
fval=0.1 # this is the fraction of data that i use for validation, computed on n_train_fix
training_loss="mse"
loss_training=tf.keras.losses.MeanSquaredError() # this is the loss i use for the training
loss_training_tau=sigma_batch_loss
shuffle=True
norm=True
map_norm=True
batch_ordering=False
distr=0
n_optimizer=0
callbacks=[True,True,True,True,False]

f_ = np.load('/home/amorelli/cl_generator/outfile_l_47_complete.npz') 
#print("outfile_R:",f_.files) #give the keiwords for the stored arrays
labels=f_.files
data=f_[labels[0]]
r=f_[labels[1]]
r, data=uf.unison_sorted_copies(r, data)
#indexes=np.linspace(0,len(r)-1,10,dtype=int)
#r=r[indexes]
#data=data[indexes]

input_folder="/home/amorelli/foreground_noise_maps/noise_maps_d1s1_train"
input_files=os.listdir(input_folder)
for i in range(len(input_files)):
    input_files[i]=input_folder+"/"+input_files[i]
noise_maps=uf.generate_noise_maps(n_train,n_channels,nside,pol=2,sensitivity=sensitivity,input_files=None)

noise_E,noise_B=uf.convert_to_EB(noise_maps)

maps_per_cl_gen=uf.maps_per_cl(distribution=distr)
maps_per_cl=maps_per_cl_gen.compute_maps_per_cl(r,n_train,n_train_fix)

mappe_B,y_r=uf.generate_maps(data, r,n_train=n_train,nside=nside, map_per_cl=maps_per_cl, 
                             noise_maps=noise_E, beam_w=2*res, kind_of_map=kind_of_map, 
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
model_tau = keras.models.load_model(
    base_dir_tau+test_model_folder,  custom_objects={'loss_training' : loss_training_tau, 'metrics' : metrics_tau}, 
    compile=False
) #i restore the model from the test_model folder. I need to specify the custom objects and recompile the model with the custom
#objects, thus the metrics and the loss functions
model_tau.compile(loss=loss_training_tau,optimizer=tf.optimizers.Adam(), metrics=metrics_tau)

if n_output_tau==1:
    predictions_train=model_tau.predict(x_train)
    predictions_val=model_tau.predict(x_val)
else:
    predictions_train=model_tau.predict(x_train)[:,0].reshape((len(y_train), 1))
    predictions_val=model_tau.predict(x_val)[:,0].reshape((len(y_val), 1))

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

model_sigma=nuf.build_network(n_inputs,nside,n_layers=n_layers,layer_nodes=nodes_per_layer,
                        num_output=n_output,use_normalization=use_normalization,
                        use_drop=use_drop,drop=drop,
                        activation_dense=activation_dense,kernel_initializer=kernel_initializer)

history=nuf.compile_and_fit(model_sigma, x_train, y_train_sigma, x_val, y_val_sigma, batch_size, max_epochs, 
                            stopping_monitor,p_stopping,reduce_monitor,f_reduce, p_reduce,base_dir, 
                            loss_training,lr,metrics,shuffle=shuffle, verbose=2,callbacks=callbacks,n_optimizer=n_optimizer)

predictions_sigma=model_sigma.predict(x_train)
np.savez(base_dir+"predictions",y_train=y_train_sigma, pred=predictions_sigma, norm=red)
model_sigma.save(base_dir+'test_model_sigma')

