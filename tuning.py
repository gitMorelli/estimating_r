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
from loss_functions import sigma_loss, sigma2_loss,sigma_batch_loss,sigma_norm_loss,sigma_log_loss,mse_tau,mse_sigma
import keras_tuner

nside = 16
seed_train=222
fval=0.1
n_train=100000
f_tune=0.1
loss_training=sigma_batch_loss
def convert_to_df(totCL):
    #d={"l":ls}
    d={}
    d["TT"]=totCL[0]
    d["EE"]=totCL[1]
    d["BB"]=totCL[2]
    d["TE"]=totCL[3]
    df=pd.DataFrame(d)
    return(df)

f_ = np.load('/home/amorelli/cl_generator/outfile_R_000_006.npz') #the cCls are in the other directory!
#https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html if i have multiple npz files

print("outfile_R:",f_.files) #give the keiwords for the stored arrays
data=f_["data"]
r=f_["r"]
print("data_shape ",data.shape)
lmax=len(data[0][0])-1
print("lmax: ",lmax)
ell = np.arange(0,lmax+1)
n_map=len(data)
map_per_cl=int(n_train/n_map)
n_channels=2
np.random.seed(seed_train)

high_nside = 512
low_nside= 16
window=hp.pixwin(low_nside,lmax=lmax)
res=hp.nside2resol(low_nside, arcmin=False) #false give output in radians
beam=hp.gauss_beam(2*res, lmax=lmax)
n_pix=hp.nside2npix(low_nside)
sensitivity=4 #muK-arcmin
mu, sigma = 0, sensitivity*np.deg2rad(1./60.)/res
smooth=window*beam

mappe_B=np.zeros((n_train,n_pix,2))
y_r=np.zeros((n_train,1))
for i,cl_in in enumerate(data):
    #print(len(ell),len(cl_in[1]))
    input_cl=convert_to_df(cl_in)
    cl=input_cl.reindex(np.arange(1, lmax+1))
    #print(cl.head())#,input_cl.tail())
    cl = cl.divide(cl.index * (cl.index+1) / (np.pi*2), axis="index")
    cl=cl.reindex(np.arange(0, lmax+1))
    cl=cl.fillna(0)
    for k in range(map_per_cl):
        index=i*map_per_cl+k
        y_r[index]=r[i]
        alm = hp.synalm((cl.TT, cl.EE, cl.BB, cl.TE), lmax=lmax, new=True) 
    #smooth=np.ones(len(window))
        alm_wb = np.array([hp.almxfl(each,smooth) for each in alm])
        #cmb_map = hp.alm2map(alm_wb, nside=low_nside, pol=True, lmax=lmax)
        alm_B=alm_wb[2]
        B_map=hp.alm2map(alm_B, nside=low_nside, pol=False, lmax=lmax)
        for j in range(0,2*n_channels,2):
            noise = np.random.normal(mu, sigma, n_pix)
            mappe_B[index,:,int(j/2)]=B_map+noise

print(mappe_B.shape, y_r.shape)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
def unison_sorted_copies(a, b):
    assert len(a) == len(b)
    p=np.argsort(a,axis=0)
    a_out=np.empty_like(a)
    b_out=np.empty_like(b)
    for i in range(len(a)):
        a_out[i]=a[p[i]]
        b_out[i]=b[p[i]]
    return a_out, b_out
mappe_B, y_r = unison_shuffled_copies(mappe_B, y_r)
x_tot=mappe_B[:]
y_tot=y_r[:]

#-----------------------------------------------------------------------------------------------------------

class MyHyperModel(keras_tuner.HyperModel): # i define an hypermodel class. 
    #it has a build function to compile the model with tunable hyperparameters and to fit it
    #using a class is the only way to tune a NN written in functional form
    def build(self, kt):
        one_layer = True
        lr=kt.Float("lr", min_value=2e-5, max_value=2*1e-3, step=2, sampling="log")#this means that the lr is a parameter that 
        #varies in the interval 2e-5 - 2e-3. Each time a new model is tested one value in this interval is chosen (2e-5, 4e-5, 8e-5, ..)
        drop_rate=kt.Float("drop_rate", min_value=0.1, max_value=0.6, step=0.1)# same for drop rate
        
        #activation_funct=kt.Choice("activation_funct", ["relu", "tanh"]) #activation _function is relu or tanh. .choice enable you to 
        #build a phase space that has strings or a set of custom numbers instead of an interval

        shape = (hp.nside2npix(nside), 2)
        inputs = tf.keras.layers.Input(shape)
        # nside 16 -> 8
        x = nnhealpix.layers.ConvNeighbours(nside, filters=32, kernel_size=9)(inputs)
        x = tf.keras.layers.Activation('relu')(x)
        x = nnhealpix.layers.Dgrade(nside, nside//2)(x)
        # nside 8 -> 4
        x = nnhealpix.layers.ConvNeighbours(nside//2, filters=32, kernel_size=9)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = nnhealpix.layers.Dgrade(nside//2, nside//4)(x)
        # nside 4 -> 2
        x = nnhealpix.layers.ConvNeighbours(nside//4, filters=32, kernel_size=9)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = nnhealpix.layers.Dgrade(nside//4, nside//8)(x)
        # nside 2 -> 1
        x = nnhealpix.layers.ConvNeighbours(nside//8, filters=32, kernel_size=9)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = nnhealpix.layers.Dgrade(nside//8, nside//16)(x)
        # dropout
        x = tf.keras.layers.Dropout(drop_rate)(x)
        x = tf.keras.layers.Flatten()(x)
        
        #for i in range(kt.Int("num_layers", 1, 2)): #this is the code to use if i want to have a variable number of layers each with number of nodes that varies in an interval (same interval)
            #x = tf.keras.layers.Dense(kt.Int(f"n_layers_{i}", min_value=8, max_value=256, step=2, sampling="log"))(x)#steps=5
            #x = tf.keras.layers.Activation('relu')(x)
            
        if one_layer==True:# depending on the state os one_layer i create a NN with one layer or with two layers
            x = tf.keras.layers.Dense(kt.Int("n_layers_0", min_value=16, max_value=256, step=2, sampling="log"))(x)
            x = tf.keras.layers.Activation('relu')(x)
            out = tf.keras.layers.Dense(2)(x)
        else:
            x = tf.keras.layers.Dense(kt.Int("n_layers_0", min_value=16, max_value=256, step=2, sampling="log"))(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Dropout(drop_rate)(x)
            x = tf.keras.layers.Dense(kt.Int("n_layers_1", min_value=16, max_value=256, step=2, sampling="log"))(x)
            x = tf.keras.layers.Activation('relu')(x)
            out = tf.keras.layers.Dense(2)(x)

        tf.keras.backend.clear_session()
        model = tf.keras.models.Model(inputs=inputs, outputs=out)
        metrics=[sigma_loss, sigma2_loss,sigma_batch_loss,sigma_norm_loss,sigma_log_loss,mse_tau,mse_sigma]
        loss_funct=2 #kt.Choice("loss_funct", [0, 2]) #i can fix the loss function to a certain metric from loss_functions library
        #or i can create a list of metrics to test with .choice
        model.compile(loss=metrics[loss_funct], optimizer=tf.optimizers.Adam(learning_rate=lr),metrics=metrics[0]) # i compile the model
        return model
    
    def fit(self,kt,model, x, y, validation_data=None, batch_size=None, shuffle=None , **kwargs):
        #in kwargs the keras oracle (that drives the execution) pass some callbacks to the function (eg the callback for checkpoints)
        n_tune=int(n_train*f_tune)#this is the number of data used for the tuning (in this case 1/10 of the complete training set of 100000 maps)
        n_val=int(n_tune*fval) #fraction of the tuning set used for validation
        batch_size=kt.Int("batch_size", 16, 128, step=2, sampling="log" )
        p_stopping=20
        p_reduce=kt.Int("p_reduce", min_value=2, max_value=20, step=2)
        f_reduce=kt.Float("f_reduce", min_value=0.1, max_value=0.8, step=0.1)
        stop_to_monitor="val_loss"
        reduce_to_monitor="val_loss"
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor=stop_to_monitor,patience=p_stopping) 
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=reduce_to_monitor, factor=f_reduce,patience=p_reduce)
        self.callbacks = kwargs.get('callbacks')+[early_stopping,reduce_lr] #fit has an object callbacks (passed through the kwargs). 
        #I set it to the callbacks i pass to the fit function + the callbacks i have defined above
        x,y=unison_shuffled_copies(x, y)
        #i subtract R to n_tune so that the training set used in the tuning is divisible by the batch_size
        tot=n_tune-n_val
        R=tot%batch_size
        x_train=x[:tot-R]
        x_val=x[tot-R:n_tune+1]
        y_train=y[:tot-R]
        y_val=y[tot-R:n_tune+1]
        if loss_training==sigma_batch_loss: #if i am using the sigma_loss as loss function in the tuning i need to give the 
            #samples ordered in a specific way to the .fit function. This is the code to do this
            y_train, x_train = unison_sorted_copies(y_train, x_train)
            list_length=int(len(y_train)/batch_size)
            lista=np.zeros(shape=(list_length,batch_size,y_train.shape[1]))
            lista_2=np.zeros(shape=(list_length,batch_size,x_train.shape[1],x_train.shape[2]))
            for i in range(list_length):
                for j in range(batch_size):
                    lista[i,j]=y_train[batch_size*i+j]
                    lista_2[i,j]=x_train[batch_size*i+j]
            lista , lista_2 = unison_shuffled_copies(lista, lista_2)
            for i in range(list_length):
                for j in range(batch_size):
                    y_train[batch_size*i+j]=lista[i,j]
                    x_train[batch_size*i+j]=lista_2[i,j]
            return model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=batch_size, shuffle=False, **kwargs)
        else: #if i use another loss function i can shuffle the input data in the normal way
            return model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=batch_size, **kwargs)
# In[13]:
home_dir='/home/amorelli/r_estimate/B_maps_white_noise/tuning'
tuner = keras_tuner.BayesianOptimization( # i create the tuner object. 
    MyHyperModel(),#i define the model i will use (with tunable hyperparameters)
    objective="val_loss",
    max_trials=50, #this is the number of sets of hyperparameters tested
    executions_per_trial=3, #this is the number of times a certain set of hyperparameters is tested
    directory=home_dir,
    overwrite=False, #if the tuning is interrupted before n_trials=max_trials it restart the tuning from last trial and add to the folder 
    #without cancelling previous trials
    project_name="3_5_23" #this is the directory in which all the results of the tuning (checkpoints, summmary of results, etc..) will be saved
)
print("tuner search space summary:","/n",tuner.search_space_summary())
max_epochs=100
tuner.search(x=x_tot, y=y_tot, epochs=max_epochs, verbose=0)#it launch the oracle -> select max_trial times a set of hyperparameters and 
#pass it to the fit function