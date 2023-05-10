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
from loss_functions import sigma_loss, sigma2_loss,sigma_batch_loss,sigma_norm_loss,sigma_log_loss,mse_tau,mse_sigma, sigma_f_loss

# i set the training parameters
nside = 16
reduce_lr_on_plateau = True
batch_size = 512
one_layer=True # this is to switch between one dense layer or two dense layer
max_epochs = 200
p_stopping=20
p_reduce=5
f_reduce=0.5
seed_train=21
lr=0.0003 
stopping_monitor="val_loss"
reduce_monitor="val_loss"
fval=0.1 # this is the fraction of data that i use for validation 
drop=0.2
n_layer_0=48
n_layer_1=64
n_layer_2=16
name='6b_5_23'
base_dir='/home/amorelli/r_estimate/B_maps_white_noise/results_'+name+'/'
loss_training=sigma_batch_loss # this is the loss i use for the training
metrics=[sigma_loss, sigma_batch_loss,mse_tau,mse_sigma, sigma_f_loss]# these are the different loss functions i have used. I use them as metrics
np.random.seed(seed_train)# i set a random seed for the generation of the maps for reproducibility
n_train=100000 #th enumber of training+validation pair of maps that i will generate

def normalize_cl(input_cl): #this is the function to divide each cl of a given spectra by l(l+1)/2pi
    output_cl=np.zeros(len(input_cl)) # i prepare the output array
    for i in range(1,len(input_cl)):
        output_cl[i]=input_cl[i]/i/(i+1)*2*np.pi # i divide each element by l(l+1)/2pi
    return output_cl

f_ = np.load('/home/amorelli/cl_generator/outfile_R_000_006.npz') 
print("outfile_R:",f_.files) #give the keiwords for the stored arrays
data=f_["data"]
r=f_["r"]
lmax=len(data[0][0])-1
print("lmax: ",lmax)
ell = np.arange(0,lmax+1)
n_map=len(data)
map_per_cl=int(n_train/n_map) # i use each spectra to generate some maps. Eg if i have 10 spectra with variable r and i want
#to generate 1000 maps i will generate 10 maps per spectra
n_channels=2

#i generate noise, window function and beam function
high_nside = 512
low_nside= 16
window=hp.pixwin(low_nside,lmax=lmax)
res=hp.nside2resol(low_nside, arcmin=False) 
beam=hp.gauss_beam(2*res, lmax=lmax)
n_pix=hp.nside2npix(low_nside)
sensitivity=4 #muK-arcmin
mu, sigma = 0, sensitivity*np.deg2rad(1./60.)/res
smooth=window*beam

mappe_B=np.zeros((n_train,n_pix,2)) # i prepare an array of n_train pairs of output maps
y_r=np.zeros((n_train,1)) # i prepare the array for the corresponding output r
for i,cl_in in enumerate(data): # i iterate on the input spectra
    cl=normalize_cl(cl_in) # i normalize the cl
    for k in range(map_per_cl): #i have map_per_cl maps to generate for each spectra -> i iterate on the same spectra map_per_cl times
        index=i*map_per_cl+k #this is the index (in mappe_B) of the kth map that i am generating for this spectra
        y_r[index]=r[i]
        alm = hp.synalm((cl.TT, cl.EE, cl.BB, cl.TE), lmax=lmax, new=True) # i generate the a_lm
        alm_wb = np.array([hp.almxfl(each,smooth) for each in alm]) #i multiply the alm by the window functions
        alm_B=alm_wb[2]#i select the alm^B
        B_map=hp.alm2map(alm_B, nside=low_nside, pol=False, lmax=lmax)#i make the harmonic transform to obtain the B map
        for j in range(0,2*n_channels,2): # i generate two maps+noise (two independent channels) from each of the map_per_cl B_maps
            #the map is the same for the channels, the noise is different (same magnitude)
            noise = np.random.normal(mu, sigma, n_pix)
            mappe_B[index,:,int(j/2)]=B_map+noise

n_val=int(n_train*fval)#this is the number of vlaidation maps i will use
def unison_shuffled_copies(a, b): # this function shuffle the first array and reorder the second so that elements that had same index
    #before the shuffling have same indexes after
    assert len(a) == len(b)
    p = np.random.permutation(len(a)) #this give a random permutation of the indexes of a
    return a[p], b[p] 
mappe_B, y_r = unison_shuffled_copies(mappe_B, y_r) # i shuffle the maps and the r values
tot=n_train-n_val # this is the number of maps used for the actual training
R=tot%batch_size # i compute the reminder with the batch size so that tot-R (the number of maps i will use for actual training)
#is divisible by the batch_size
x_train=mappe_B[:tot-R]
x_val=mappe_B[tot-R:]
y_train=y_r[:tot-R]
y_val=y_r[tot-R:]
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
if loss_training==sigma_batch_loss: #if i use the sigma_batch_loss i also need to use a particular ordering of the training set 
    #this is why i make this check
    y_train, x_train = unison_sorted_copies(y_train, x_train)# i sort the training set according to the r used to generate the maps
    list_length=int(len(y_train)/batch_size)
    lista=np.zeros(shape=(list_length,batch_size,y_train.shape[1])) # i generate an array that has same shape of y_r but has blocks of 
    #batch_size -> i will store the y_r here dividing y_r in block of batch_size
    #the lengt of this container will be len(y_train)/batch_size of course
    lista_2=np.zeros(shape=(list_length,batch_size,x_train.shape[1],x_train.shape[2]))# same but with B-maps
    for i in range(list_length):
        for j in range(batch_size):
            lista[i,j]=y_train[batch_size*i+j]# i consider the ith elment of the list_length elements of the container
            #the ith element contains batc_size elements form y, namely the elements from batch_size*i to batch_size(i+1)
            lista_2[i,j]=x_train[batch_size*i+j] # i do the same for mappe-B
    lista , lista_2 = unison_shuffled_copies(lista, lista_2) # i shuffle the container so that the elements in the batch_size block are not
    #shuffled, only the blocks are shuffled
    for i in range(list_length):
        for j in range(batch_size):
            y_train[batch_size*i+j]=lista[i,j]# i put again the elements from the container to the y_r array
            x_train[batch_size*i+j]=lista_2[i,j]#same with mappe B
    #notice that there is no need to sort the validation dataset

def compile_and_fit(model, X_train, y_train, X_valid, y_valid): # function to compile and run the model
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=stopping_monitor,
                                                      patience=p_stopping,
                                                      mode='min')
    #this callback stops the training if the monitor metric doesn't get better in p_stopping epochs
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=reduce_monitor, 
                                                     factor=f_reduce,
                                                     patience=p_reduce)
    #this callback reduce the lr if the monitor metric doesn't get better in p_reduce epochs
    
    csv_logger=tf.keras.callbacks.CSVLogger(base_dir+'log', separator=" ", append=False)
    #this callback print the value of the loss and metrics of each epoch to a log file
    
    checkpoint_filepath = base_dir+'checkpoints/saved-weights-{epoch:02d}-{val_loss:.5f}.hdf5' 
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True, save_freq='epoch',save_best_only=False)
    #this callback save the weights of the model at each epoch of the training and save them in .hdf5 files
        
    model.compile(loss=loss_training,
                  optimizer=tf.optimizers.Adam(learning_rate=lr),
                  metrics=metrics)
    #i define the loss function, optimizer and metrics that i am using in the training

    if loss_training==sigma_batch_loss: #if i use sigma_batch_loss i need the model to use the custom sorting, if i don't i can shuffle the
        #data before training 
        history = model.fit(x=X_train, y=y_train, 
                            batch_size=batch_size, epochs=max_epochs,
                            validation_data=(X_valid, y_valid),
                            callbacks=[reduce_lr,early_stopping,model_checkpoint_callback,csv_logger],shuffle=False,verbose=2)
    else:
        history = model.fit(x=X_train, y=y_train, 
                            batch_size=batch_size, epochs=max_epochs,
                            validation_data=(X_valid, y_valid),
                            callbacks=[reduce_lr,early_stopping,model_checkpoint_callback,csv_logger],verbose=2)
    return history

#the structure of the neural network
shape = (hp.nside2npix(nside), 2)
inputs = tf.keras.layers.Input(shape)
# nside 16 -> 8
x=inputs
for k in range(4):
    x = nnhealpix.layers.ConvNeighbours(nside/2**k, filters=32, kernel_size=9)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = nnhealpix.layers.Dgrade(nside//2**k, nside//2**(k+1))(x) # i use 4 convolutional layers, for each layer i decrease the number of pixels by 1/2
# dropout
x = tf.keras.layers.Dropout(drop)(x)
x = tf.keras.layers.Flatten()(x)
if one_layer==True:# depending on the state os one_layer i create a NN with one layer or with two layers
    x = tf.keras.layers.Dense(n_layer_0)(x)
    x = tf.keras.layers.Activation('relu')(x)
    out = tf.keras.layers.Dense(2)(x)
else:
    x = tf.keras.layers.Dense(n_layer_1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(drop)(x)
    x = tf.keras.layers.Dense(n_layer_2)(x)
    x = tf.keras.layers.Activation('relu')(x)
    out = tf.keras.layers.Dense(2)(x)
tf.keras.backend.clear_session()
model = tf.keras.models.Model(inputs=inputs, outputs=out)

history = compile_and_fit(model, x_train, y_train, x_val, y_val)
print('Saving model to disk')
model.save(base_dir+'test_model')

#-----------------------------------------
hyperparameters={} #i save all the hyperparameters in a dictionary 
hyperparameters["name"]=name
hyperparameters["noise"]=sensitivity
hyperparameters["lr"]=lr
hyperparameters["n_layers"]=one_layer
if one_layer:
    hyperparameters["nodes_layers"]=n_layer_0
else:
    hyperparameters["nodes_layers"]=str(n_layer_1)+"-"+str(n_layer_2)
hyperparameters["batch_size"]=batch_size
hyperparameters["p_reduce"]=p_reduce
hyperparameters["f_reduce"]=f_reduce
hyperparameters["p_stopping"]=p_stopping
hyperparameters["stopping_monitor"]=stopping_monitor
hyperparameters["reduce_monitor"]=reduce_monitor
hyperparameters["comments"]="this is fine tuned on best model from tuning 1-5-23"
hyperparameters = {k:[v] for k,v in hyperparameters.items()}
output=pd.DataFrame(hyperparameters) # i convert the hyperparameters 
output.to_csv(base_dir+'output.txt', index=False, sep=' ')