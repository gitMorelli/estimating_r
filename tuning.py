#!/usr/bin/env python
# coding: utf-8

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
import keras_tuner

seed_train=40
np.random.seed(seed_train)# i set a random seed for the generation of the maps for reproducibility
tf.random.set_seed(seed_train)#the seed for tensorflow operation is different from the seed for numpy operations

nside = 16
n_train=100000 
n_train_fix=40000 #the total number of of training maps i will spread on all the r interval -> for each r value i generate n_train_fix/len(r) maps 
kind_of_map="BB"
n_channels=2
pol=1
res=hp.nside2resol(nside, arcmin=False) 
sensitivity=4

#train and val
fval=[0.1,0.2] # this is the fraction of data that i use for validation, computed on n_train_fix
norm=True
map_norm=True
batch_ordering=False
batch_size=16
n_inputs=2

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
maps_per_cl=[]
mappe_B=[]
y_r=[]
x_train=[]
y_train=[]
x_val=[]
y_val=[]
for i in range(2):
    maps_per_cl_gen=uf.maps_per_cl(distribution=i)
    maps_per_cl.append(maps_per_cl_gen.compute_maps_per_cl(r,n_train,n_train_fix))
    mappe,y_mappe=uf.generate_maps(data, r,n_train=n_train,nside=nside, map_per_cl=maps_per_cl[i], 
                             noise_maps=noise_maps, beam_w=2*res, kind_of_map=kind_of_map, 
                             raw=0 , n_channels=n_channels,beam_yes=1 , verbose=0)
    mappe_B.append(mappe)
    y_r.append(y_mappe)
    x_t,y_t,x_v,y_v = nuf.prepare_data(y_r[i],mappe_B[i],r,n_train,n_train_fix,fval[i],maps_per_cl[i]
                                               , batch_size, batch_ordering=batch_ordering)
    x_train.append(x_t)
    y_train.append(y_t)
    x_val.append(x_v)
    y_val.append(y_v)
    
    if norm:
        y_train[i]=nuf.normalize_data(y_train[i],r)
        y_val[i]=nuf.normalize_data(y_val[i],r)
    if map_norm:
        for k in range(len(x_train[i])):
            for j in range(n_inputs):
                x=x_train[i][k,:,j]
                x_train[i][k,:,j]=nuf.normalize_data(x,x)
        for k in range(len(x_val[i])):
            for j in range(n_inputs):
                x=x_val[i][k,:,j]
                x_val[i][k,:,j]=nuf.normalize_data(x,x)
#-----------------------------------------------------------------------------------------------------------

class MyHyperModel(keras_tuner.HyperModel): # i define an hypermodel class. 
    #it has a build function to compile the model with tunable hyperparameters and to fit it
    #using a class is the only way to tune a NN written in functional form
    def build(self, kt):
        lr=kt.Float("lr", min_value=1e-5, max_value=1e-2, step=10, sampling="log")#this means that the lr is a parameter that 
        drop_rate=kt.Choice("drop_rate", [0.2,0.5])
        activation_funct=kt.Choice("activation_funct", ["relu", "swish"]) 
        kernel_initializer=kt.Choice("kernel_initializer", ["glorot_uniform", "he_normal"])
        use_normalization=kt.Choice("use_normalization",[True,False])

        model=nuf.build_network(n_inputs=2,nside=16,n_layers=2,layer_nodes=[256,128],
                        num_output=1,use_normalization=[False,use_normalization,use_normalization],
                        use_drop=[False,True,True],drop=[drop_rate,drop_rate,drop_rate],
                        activation_dense=activation_funct,kernel_initializer=kernel_initializer)
        
        metrics=[tf.keras.losses.MeanSquaredError()]
        loss_funct=0
        optimizers=[tf.optimizers.Adam(learning_rate=lr),tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9,
 nesterov=True)]
        optimizer_index=kt.Choice("optimizer_index",[0,1])
        model.compile(loss=metrics[loss_funct], optimizer=optimizers[optimizer_index],metrics=[]) # i compile the model
        return model
    
    def fit(self,kt,model, x, y, validation_data, batch_size=None, shuffle=None , **kwargs):
        #in kwargs the keras oracle (that drives the execution) pass some callbacks to the function (eg the callback for checkpoints)
        batch_size=16
        p_stopping=10
        stop_to_monitor="val_loss"
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor=stop_to_monitor,patience=p_stopping,restore_best_weights=True) 
        self.callbacks = kwargs.get('callbacks')+[early_stopping] #fit has an object callbacks (passed through the kwargs). 
        train_distr=kt.Choice("train_distr",[0,1])
        
        n_train=len(y[train_distr])
        n_val=len(validation_data[1][train_distr])
        f_tune=0.1
        
        index_train=np.random.randint(0,n_train,math.ceil(n_train*f_tune))
        index_val=np.random.randint(0,n_val,math.ceil(n_val*f_tune))
        
        x_train=x[train_distr][index_train]
        x_val=validation_data[0][train_distr][index_val]
        y_train=y[train_distr][index_train]
        y_val=validation_data[1][train_distr][index_val]
        print("train length:",len(x_train),len(x_val))
        return model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=batch_size, shuffle=True, **kwargs)

home_dir='/home/amorelli/r_estimate/B_maps_white_noise/tuning'
project_name="24_6_23"
tuner = keras_tuner.RandomSearch( # i create the tuner object. 
    MyHyperModel(),#i define the model i will use (with tunable hyperparameters)
    objective="val_loss",
    max_trials=1000, 
    #seed=2,
    executions_per_trial=1, #this is the number of times a certain set of hyperparameters is tested
    directory=home_dir,
    #max_retries_per_trial=2,
    overwrite=False, #if the tuning is interrupted before n_trials=max_trials it restart the tuning from last trial and add to the folder 
    #without cancelling previous trials
    project_name=project_name #this is the directory in which all the results of the tuning (checkpoints, summmary of results, etc..) will be saved
)
print("tuner search space summary:","/n",tuner.search_space_summary())
max_epochs=400
tuner.search(x=x_train, y=y_train, validation_data=(x_val,y_val), epochs=max_epochs, verbose=2)#it launch the oracle -> select max_trial times a set of hyperparameters and 
#pass it to the fit function
for i in range(2):
    np.savez(home_dir+"/"+project_name+"/"+"check_r_distribution_"+str(i),y_train=y_train[i],y_val=y_val[i]) 
