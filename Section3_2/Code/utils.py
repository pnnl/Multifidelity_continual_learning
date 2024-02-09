"""
Created 2023
@author: Amanda Howard (amanda.howard@pnnl.gov)

"""

import jax.numpy as np
from jax import random, jit
from jax.example_libraries import optimizers
from jax.nn import relu, elu, selu, swish

from functools import partial
from torch.utils import data



    
class DataGenerator_h(data.Dataset):
    def __init__(self, u, s,  u_l, 
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u
        self.s = s
        self.u_l = u_l
        
        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs,  u_l = self.__data_generation(subkey)
        return inputs, outputs,  u_l

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        s = self.s[idx,:,:]
        u = self.u[idx,:,:]
        u_l = self.u_l[idx,:,:]

        # Construct batch
        inputs = (u, u_l)
        outputs = s
        return inputs, outputs
    

def DNN(branch_layers, activation=np.tanh):

    def xavier_init_j(key, d_in, d_out):
        glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * random.normal(key, (d_in, d_out))
        b = np.zeros(d_out)
        return W, b
    def init(rng_key1):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            W, b = xavier_init_j(k1, d_in, d_out)
            return W, b
        key1, *keys1 = random.split(rng_key1, len(branch_layers))
        branch_params = list(map(init_layer, keys1, branch_layers[:-1], branch_layers[1:]))
        return branch_params
        
    def apply(params, u):
      #  print(u.shape)
        for k in range(len(branch_layers)-2):
            W_b, b_b = params[k]
            
            u = activation(np.dot(u, W_b) + b_b)

        W_b, b_b = params[-1]
        u = np.dot(u, W_b) + b_b
      #  print(u.shape)

        return u

    return init, apply



    
def nonlinear_DNN(branch_layers, activation=np.tanh):

    def xavier_init_j(key, d_in, d_out):
        glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * random.normal(key, (d_in, d_out))
        b = np.zeros(d_out)
        return W, b
    def init(rng_key1):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            W, b = xavier_init_j(k1, d_in, d_out)
            return W, b
        key1, *keys1 = random.split(rng_key1, len(branch_layers))
        branch_params = list(map(init_layer, keys1, branch_layers[:-1], branch_layers[1:]))
        return branch_params
        
    def apply(params, u):
      #  print(u.shape)
        for k in range(len(branch_layers)-2):
            W_b, b_b = params[k]
            u = activation(np.dot(u, W_b) + b_b)
        W_b, b_b = params[-1]
        u = (np.dot(u, W_b) + b_b)
        return u
        
    def weight_norm(params):
    
        loss = 0

        for k in range(len(branch_layers)-1):
            W_b, b_b = params[k]
            
            loss += np.sum(W_b**2)
            loss += np.sum(b_b**2)

        return loss
    
    return init, apply, weight_norm

def linear_DNN(branch_layers):

    def xavier_init_j(key, d_in, d_out):
        glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * random.normal(key, (d_in, d_out))
        b = np.zeros(d_out)
        return W, b
    def init(rng_key1):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            W, b = xavier_init_j(k1, d_in, d_out)
            return W, b
        key1, *keys1 = random.split(rng_key1, len(branch_layers))
        branch_params = list(map(init_layer, keys1, branch_layers[:-1], branch_layers[1:]))
        return (branch_params )
        
    def apply(params, u):
        branch_params = params
        for k in range(len(branch_layers)-1):
            W_b, b_b = branch_params[k]

            u = (np.dot(u, W_b) + b_b)
        

        return u

    return init, apply


    
    
