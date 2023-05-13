import sys
import h5py
import numpy as np
import healpy as hp
import tensorflow as tf
import random as python_random
import nnhealpix.layers
from tensorflow.keras import metrics
import pandas as pd

def sigma_loss(y_true, y_pred):
    squared_residual = tf.math.square(y_true[:,0] - y_pred[:,0]) 
    sigma = y_pred[:,1] 
    loss = tf.math.reduce_sum(squared_residual) #reduce sum make the sum of the argument on the whole batch sum_i^batch_size
    #this is the mse for y_pred[0]
    loss += tf.math.reduce_sum( tf.math.square(squared_residual -  sigma) )
    #this is added to the mse. It trains the second output y_pred[1].
    return loss

def sigma2_loss(y_true, y_pred):
    squared_residual = tf.math.square(y_true[:,0] - y_pred[:,0])
    sigma =tf.math.square( y_pred[:,1])
    loss = tf.math.reduce_sum(squared_residual)
    loss += tf.math.reduce_sum( tf.math.square(squared_residual -  sigma) )
    #same as before but sigma is y_pred[0]^2
    return loss
def sigma_batch_loss(y_true, y_pred):
    squared_residual = tf.math.square(y_true[:,0] - y_pred[:,0])
    size=tf.shape(y_true,out_type=tf.dtypes.int32)[0] #this gives the size of the batch
    #recall that during the training the steps are as follows: the loss is computed on one batch, the weights are updated
    #this is done for all batches in an epoch. -> if batch_size=32 the input shape will always be 32 during the training
    size=tf.cast(size, dtype=tf.dtypes.float32)#i cast it to float since size is int by default
    var_batch=tf.math.divide(tf.math.reduce_sum(squared_residual),size)
    sigma_batch=tf.math.sqrt(var_batch)
    sigma = y_pred[:,1]
    loss = tf.math.reduce_sum(squared_residual)
    loss += tf.math.reduce_sum( tf.math.square(sigma_batch -  sigma) )
    #the sigma is trained on the std of the batch elements
    return loss

def sigma_norm_loss(y_true, y_pred):
    squared_residual = tf.math.square(y_true[:,0] - y_pred[:,0])
    sigma = y_pred[:,1]
    loss = tf.math.reduce_sum(squared_residual)
    loss += tf.math.reduce_sum( tf.math.square(squared_residual - tf.math.multiply(sigma,tf.math.square(y_true[:,0])) ) )
    return loss

def sigma_log_loss(y_true, y_pred):
    squared_residual = tf.math.square(y_true[:,0] - y_pred[:,0])
    squared_sigma = tf.math.square(y_pred[:,1])
    loss = tf.math.log(tf.math.reduce_sum(squared_residual)+1)
    loss += tf.math.log(tf.math.reduce_sum( tf.math.square(squared_residual -  squared_sigma ) )+1)
    return loss

def mse_tau(y_true, y_pred):
    squared_residual = tf.math.square(y_true[:,0] - y_pred[:,0])
    loss = tf.math.reduce_sum(squared_residual)
    #this computes only the mse on y_pred[0]. I use this as a metric
    return loss

def mse_sigma(y_true, y_pred):
    squared_residual = tf.math.square(y_true[:,0] - y_pred[:,0])
    sigma = y_pred[:,1]
    loss = tf.math.reduce_sum( tf.math.square(squared_residual -  sigma) )
    #this loss is = sigma_loss - mse_tau. I use this as a metric for sigma
    return loss

def sigma_f_loss(y_true, y_pred):
    squared_residual = tf.math.square(y_true[:,0] - y_pred[:,0])
    sigma = y_pred[:,1]
    f=10.0**8
    loss = tf.math.reduce_sum(squared_residual)
    loss += f*tf.math.reduce_sum( tf.math.square(squared_residual -  sigma) )
    #in this loss i multiply the sigma part of the loss by a factor f
    return loss

def mse_batch(y_true, y_pred):
    squared_residual = tf.math.square(y_true[:,0] - y_pred[:,0])
    #size=tf.math.reduce_sum(tf.math.divide(y_true[:,0],y_true[:,0]))
    size=tf.shape(y_true,out_type=tf.dtypes.int32)[0]
    size=tf.cast(size, dtype=tf.dtypes.float32)
    #tf.print(size)
    var_batch=tf.math.divide(tf.math.reduce_sum(squared_residual),size)
    sigma_batch=tf.math.sqrt(var_batch)
    sigma = y_pred[:,1]
    #squared_sigma = tf.math.square(y_pred[:,1])
    loss = tf.math.reduce_sum( tf.math.square(sigma_batch -  sigma) )
    #loss += tf.math.reduce_sum( tf.math.square(squared_residual -  squared_sigma) )
    return loss