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

# i set the training parameters
nside = 16
reduce_lr_on_plateau = True
batch_size = 16
one_layer=True # this is to switch between one dense layer or two dense layer
max_epochs = 200
p_stopping=20
p_reduce=5
f_reduce=0.5
seed_train=3
lr=0.0003 
stopping_monitor="val_mse_batch"
reduce_monitor="val_loss"
fval=0.10 # this is the fraction of data that i use for validation 
drop=0.2
n_layer_0=48
n_layer_1=64
n_layer_2=16
name='14_5_23'
base_dir='/home/amorelli/r_estimate/B_maps_white_noise/results_'+name+'/'
training_loss="new_sigma_batch"
loss_training=sigma_batch_loss # this is the loss i use for the training
metrics=[sigma_loss, sigma_batch_loss,mse_tau,mse_sigma, sigma_f_loss, mse_batch]# these are the different loss functions i have used. I use them as metrics
np.random.seed(seed_train)# i set a random seed for the generation of the maps for reproducibility
n_train=100000 #the total number of training+validation pair of maps that i will generate
n_train_fix=40000 #the total number of of training maps i will spread on all the r interval -> for each r value i generate n_train_fix/len(r) maps 

def normalize_cl(input_cl): #this is the function to divide each cl of a given spectra by l(l+1)/2pi
    output_cl=np.zeros(len(input_cl)) # i prepare the output array
    for i in range(1,len(input_cl)):
        output_cl[i]=input_cl[i]/i/(i+1)*2*np.pi # i divide each element by l(l+1)/2pi
    return output_cl
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
def unison_shuffled_copies(a, b): # this function shuffle the first array and reorder the second so that elements that had same index
    #before the shuffling have same indexes after
    assert len(a) == len(b)
    p = np.random.permutation(len(a)) #this give a random permutation of the indexes of a
    return a[p], b[p] 
def compute_map_per_cl(r,n,n_fix): # this function determine how many maps i need to generate for each Cl based on n_train and n_train_fix
    #from each cl n_train_fix/n_cl maps are generated at least. The other n_train-n_train_fix are spread on the interval so that more maps
    #are generated as we go from r_half to r=r_max or r=r_min. I use a linear relation map_per_cl=m*r+q
    n_cl=len(r) #the number of cls i use to generate the maps
    dr=r[1]-r[0] #difference between subcessive r values
    r_mean=r[-1]/2#middle r of the interval
    s_min=int(n_fix/n_cl) #the number of maps generated for Cl(r=r_half), this is the minimum number of maps generated for a Cl
    z=r_mean*n_cl/2 + dr*(n_cl/2)*(n_cl/2+1)/2
    m=(n/2-s_min*(n_cl/2+1))/(z-n_cl/2*r_mean) #formula for the angular coefficient of the formula that generates the maps_per_cl
    q=s_min-m*r_mean 
    num_maps=np.empty_like(r,dtype=int)
    q_1=q+2*m*r_mean #for generation of maps_per_cl on the left of the middle value i need to change the sign of m and compute a new q based
    #on r_half and q
    for i,x in enumerate(r):
        if x>=r_mean: #for points on the right of r_half i use one formula, for points on the left the symmetric formula
            num_maps[i]=math.ceil(m*x+q)
        else:
            num_maps[i]=math.ceil(-m*x+q_1)
    return(num_maps)

f_ = np.load('/home/amorelli/cl_generator/outfile_R_000_001_seed=67.npz') 
print("outfile_R:",f_.files) #give the keiwords for the stored arrays
data=f_["data"]
r=f_["r"]
lmax=len(data[0][0])-1
print("lmax: ",lmax)
r, data=unison_sorted_copies(r, data)
map_per_cl=np.empty_like(r)
size=len(map_per_cl)
map_per_cl=compute_map_per_cl(r,n_train,n_train_fix) #i generate map_per_cl for every value of r
n_train=np.sum(map_per_cl) #the actual n_train is bigger than the initial n_train because i use the ceil function to round the numbers
#in the compute_map function

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
previous=0 #this is an index to keep track of the position in mappe_B as i generate the maps for the CL
#i need an index cause i generate a variable number of maps for each Cl
for i,cl_in in enumerate(data): # i iterate on the input spectra
    cl=np.empty_like(cl_in)
    for j in range(len(cl_in)):
        cl[j]=normalize_cl(cl_in[j]) # i normalize the cl
    for k in range(map_per_cl[i]): #i have map_per_cl maps to generate for each spectra -> i iterate on the same spectra map_per_cl times
        index=previous+k #this is the index (in mappe_B) of the kth map that i am generating for this spectra
        y_r[index]=r[i]
        alm = hp.synalm((cl[0], cl[1], cl[2], cl[3]), lmax=lmax, new=True) # i generate the a_lm. In order the cl[i] are TT,EE,BB,TE
        alm_wb = np.array([hp.almxfl(each,smooth) for each in alm]) #i multiply the alm by the window functions
        alm_B=alm_wb[2]#i select the alm^B
        B_map=hp.alm2map(alm_B, nside=low_nside, pol=False, lmax=lmax)#i make the harmonic transform to obtain the B map
        for j in range(0,2*n_channels,2): # i generate two maps+noise (two independent channels) from each of the map_per_cl B_maps
            #the map is the same for the channels, the noise is different (same magnitude)
            noise = np.random.normal(mu, sigma, n_pix)
            mappe_B[index,:,int(j/2)]=B_map+noise
    previous+=map_per_cl[i] #after i have generated all the maps for a Cl i add to the index the number of maps i have 
    #generated for that Cl
    
n_val=math.ceil(n_train_fix/len(r)*fval)#this is the number of validation maps i take for each value of r (it is the fraction fval of the 
#total number of maps that are distributed on all r values
x_val=np.empty_like(mappe_B)[:int(n_val*len(r))] #i take n_val element for each r to build the validation set -> it will have this dimension
y_val=np.empty_like(y_r)[:int(n_val*len(r))]
tot=n_train-n_val*len(r) # this is the number of maps used for the actual training
x_train=np.empty_like(mappe_B)[:tot] #i take n_val element for each r to build the validation set -> it will have this dimension
y_train=np.empty_like(y_r)[:tot]
R=tot%batch_size # i compute the reminder with the batch size so that tot-R (the number of maps i will use for actual training)
#is divisible by the batch_size

previous=0
previous_val=0 #this index and the next have the same scope of previous but keep track of position in y_val and y_train (not in y_r)
previous_train=0
for i in range(len(r)):
    for k in range(n_val):
        y_val[previous_val+k]=y_r[previous+k]
        x_val[previous_val+k]=mappe_B[previous+k]
    previous_val+=n_val
    for k in range(map_per_cl[i]-n_val):
        y_train[previous_train+k]=y_r[previous+n_val+k]
        x_train[previous_train+k]=mappe_B[previous+n_val+k]
    previous_train+=map_per_cl[i]-n_val
    previous+=map_per_cl[i]
x_train=x_train[:len(x_train)-R]# i remove the last R maps from the training set ( i simply remove them because it is easier 
#than finding a way to pass R uniformly chosen maps to validation set and R<<n_train)
y_train=y_train[:len(y_train)-R]

if loss_training==sigma_batch_loss: #if i use the sigma_batch_loss i also need to use a particular ordering of the training set 
    #this is why i make this check
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
np.savez("check_r_distribution",y_train=y_train,y_val=y_val) 

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
    x = nnhealpix.layers.ConvNeighbours(nside//2**k, filters=32, kernel_size=9)(x)
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
