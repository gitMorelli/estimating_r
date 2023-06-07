import sys
import h5py
import numpy as np
import healpy as hp
import tensorflow as tf
import random as python_random
import nnhealpix.layers
from tensorflow.keras import metrics
import pandas as pd
import useful_functions as uf
from loss_functions import sigma_loss, sigma2_loss,sigma_batch_loss,sigma_norm_loss,sigma_log_loss,mse_tau,mse_sigma, mse_batch, sigma_f_loss
import math

def prepare_data(y_r,mappe_B,r,n_train,n_train_fix,fval,map_per_cl, batch_size,batch_ordering=False):
    ''' assume y_r and mappe_B are already sorted '''
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

    if batch_ordering: #if i use the sigma_batch_loss i also need to use a particular ordering of the training set 
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
        lista , lista_2 = uf.unison_shuffled_copies(lista, lista_2) # i shuffle the container so that the elements in the batch_size block are not
        #shuffled, only the blocks are shuffled
        for i in range(list_length):
            for j in range(batch_size):
                y_train[batch_size*i+j]=lista[i,j]# i put again the elements from the container to the y_r array
                x_train[batch_size*i+j]=lista_2[i,j]#same with mappe B
        #notice that there is no need to sort the validation dataset
    return x_train,y_train,x_val,y_val

def compile_and_fit(model, x_train, y_train, x_val, y_val, batch_size, max_epochs, stopping_monitor,p_stopping,reduce_monitor,f_reduce, p_reduce,base_dir, loss_training,lr,metrics,shuffle=True,verbose=2,callbacks=[True,True,True,True]): # function to compile and run the model
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
    callbacks_all=[early_stopping,reduce_lr,csv_logger,model_checkpoint_callback]
    callbacks=[]
    for i,c in enumerate(callbacks):
        if c:
            callbacks.append(callbacks_all[i])
        
    model.compile(loss=loss_training,
                  optimizer=tf.optimizers.Adam(learning_rate=lr),
                  metrics=metrics)
    #i define the loss function, optimizer and metrics that i am using in the training
    history = model.fit(x=x_train, y=y_train, 
                            batch_size=batch_size, epochs=max_epochs,
                            validation_data=(x_val, y_val),
                            callbacks=callbacks,shuffle=shuffle,verbose=verbose)
    return history

def build_network(n_inputs,nside,drop,n_layer_0,n_layer_1,n_layer_2,one_layer=True):
    #the structure of the neural network
    shape = (hp.nside2npix(nside), n_inputs)
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
    return model

def build_network_tau(n_inputs,nside,drop,n_layer_0,n_layer_1,n_layer_2,one_layer=True):
    #the structure of the neural network
    shape = (hp.nside2npix(nside), n_inputs)
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
        out = tf.keras.layers.Dense(1)(x)
    tf.keras.backend.clear_session()
    model = tf.keras.models.Model(inputs=inputs, outputs=out)
    return model

def build_network_sigma(n_inputs,nside,drop,n_layer_0,n_layer_1,n_layer_2,one_layer=True):
    #the structure of the neural network
    shape = (hp.nside2npix(nside), n_inputs)
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
        out = tf.keras.layers.Dense(1)(x)
    tf.keras.backend.clear_session()
    model = tf.keras.models.Model(inputs=inputs, outputs=out)
    return model