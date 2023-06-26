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

seed_train=44
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
max_epochs=400

n_layers=2
nodes_per_layer=[256,128]
n_output=1
use_drop=[False,True,True]

stopping_monitor="val_loss"
reduce_monitor="val_loss"
p_stopping=10
f_reduce=1
p_reduce=5
callbacks=[True,False,False,False,False]
base_dir="/home/amorelli/r_estimate/B_maps_white_noise/tuning/25_6_23"
loss_training=tf.keras.losses.MeanSquaredError()
metrics=[]

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
parameters={
    "lr":[0.01 * (0.1)**i for i in range(5)],
    "kernel":["he_normal","glorot_uniform"],
    "optimizers":[0,1],
    "use_normalization":[True,False],
    "drop":[0.2,0.5],
    "distr":[0,1],
    "activation":["relu","swish"]
}
n_param=len(parameters)
all_configurations=[]
def select_indices(n_param, parameters, configuration, depth):
    key=list(parameters.keys())[depth]
    dim=len(parameters[key])
    if depth==n_param-1:
        for i in range(dim):
            configuration[depth]=i
            all_configurations.append(list(configuration))
        #print(all_configurations)
        return None
    else:
        for i in range(dim):
            configuration[depth]=i
            select_indices(n_param, parameters, configuration, depth+1)
select_indices(n_param, parameters, [0 for i in range(n_param)],0)
st=0
for index,list_index in enumerate(all_configurations[st:]):
    lr=parameters["lr"][list_index[0]]
    kernel=parameters["kernel"][list_index[1]]
    n_optimizer=parameters["optimizers"][list_index[2]]
    use_normalization=parameters["use_normalization"][list_index[3]]
    drop=parameters["drop"][list_index[4]]
    train_distr=parameters["distr"][list_index[5]]
    activation=parameters["activation"][list_index[6]]
    
    n_train=len(y_train[train_distr])
    n_val=len(y_val[train_distr])
    f_tune=0.1

    index_train=np.random.randint(0,n_train,math.ceil(n_train*f_tune))
    index_val=np.random.randint(0,n_val,math.ceil(n_val*f_tune))

    x_t=x_train[train_distr][index_train]
    x_v=x_val[train_distr][index_val]
    y_t=y_train[train_distr][index_train]
    y_v=y_val[train_distr][index_val]
    
    model=nuf.build_network(n_inputs,nside,n_layers=n_layers,layer_nodes=nodes_per_layer,
                        num_output=n_output,use_normalization=[False,use_normalization,use_normalization],
                        use_drop=use_drop,drop=[drop,drop,drop],
                        activation_dense=activation,kernel_initializer=kernel)
    history=nuf.compile_and_fit(model, x_t, y_t, x_v, y_v, batch_size, max_epochs, 
                                stopping_monitor,p_stopping,reduce_monitor,f_reduce, p_reduce,base_dir, 
                                loss_training,lr,metrics,shuffle=True, verbose=2,callbacks=callbacks,n_optimizer=n_optimizer)
    history_d=history.history
    val_loss=history_d["val_loss"][-10]
    last_epoch=len(history_d["val_loss"])
    lista=[str(index+st),str(val_loss),str(last_epoch)]
    print(index+st,val_loss,last_epoch)
    print("-------------------------------------------------")
    with open(base_dir+"/"+'history.txt',"a") as f:
        f.write("\n")
        f.write(" ".join(lista))