import os
import math
import jax
import jax.numpy as np
from jax import random, vmap, jit
from jax.nn import relu, elu, selu, swish
from jax.config import config

from functools import partial
from torch.utils import data

class DataGenerator(data.Dataset):
    def __init__(self, u, s, 
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u
        self.s = s
        
        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        s = self.s[idx, :]
        u = self.u[idx, :]
        # Construct batch
        inputs = u
        outputs = s
        return inputs, outputs


class DataGenerator_res(data.Dataset):
    def __init__(self, u_coords,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.coords = u_coords
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        u = self.coords[0] + (self.coords[1]-self.coords[0])*random.uniform(key, shape=[self.batch_size, 1])
        s = np.zeros([self.batch_size, 1])
        # Construct batch
        inputs = u
        outputs = s
        return inputs, outputs

class DataGenerator_RDPS(data.Dataset):
    def __init__(self, u_coords, res_pts, res_val, batch_size_res=20, 
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.coords = u_coords
        self.batch_size = batch_size
        self.batch_size_res = batch_size_res
        self.res_pts = res_pts
        self.N = res_pts.shape[0]
        self.res_val = res_val
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        random_batch_size = self.batch_size-self.batch_size_res
        u = self.coords[0] + (self.coords[1]-self.coords[0])*random.uniform(key, shape=[random_batch_size, 1])
        idx = random.choice(key, self.N, (self.batch_size_res,), p=self.res_val, replace=False)
        u_res = self.res_pts[idx]
        u = np.concatenate([u, u_res])
        s = np.zeros([self.batch_size, 1])
        # Construct batch
        inputs = u
        outputs = s
        return inputs, outputs

    
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
    

def DNN(branch_layers, activation=swish):

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



    
def nonlinear_DNN(branch_layers, activation=swish):

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


    
    
