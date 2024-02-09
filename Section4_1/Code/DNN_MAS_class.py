"""
Created 2023
@author: Amanda Howard (amanda.howard@pnnl.gov)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import jax
import jax.numpy as np
from jax import random, grad, vmap, jit
from jax.example_libraries import optimizers
from jax.nn import relu
from jax.config import config
from jax import lax
from jax.flatten_util import ravel_pytree

import itertools
from functools import partial
from tqdm import trange, tqdm
from copy import deepcopy

######################################################################
#######################  Standard DNN ##########################
######################################################################


def DNN(branch_layers, activation=relu):

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
        return u

    return init, apply

class DNN_MAS:
    
    # Initialize the class
    def __init__(self, layers_branch_low, lr, restart = 0, params_t = 0, params_i = 0): 


        #Network initialization 
        self.init_low, self.apply_low = DNN(layers_branch_low)
        self.params_t = params_t
        if restart ==1:
            params_low = params_i
        else:
            params_low = self.init_low(random.PRNGKey(10))
        params = (params_low)
        
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(lr)
     
        self.opt_state = self.opt_init(params)

        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)

        self.itercount = itertools.count()

        # building loss function
        self.error_training_log = []
        self.error_test_log = []


    # =============================================
    # evaluation
    # =============================================
    
    def predict(self, params, image):
      return self.apply_low(params, image)

    def batched_predict(self, params, images):
        batched_predict = vmap(self.predict, in_axes=(None, 0))(params, images)
        return batched_predict

    def loss(self, params, params_t, batch_low, F, lam):
        u, y = batch_low

        preds = self.batched_predict(params, u)
        l =  np.mean((y.flatten()- preds.flatten())**2)

        count = 0
        s = 0.0
        n = len(params)

        for j in range(len(lam)):
            for k in range(len(params)):
                s += lam[j]/2 * np.sum(F[count]*(params[k][0]-params_t[n*j + k][0])**2)
                count += 1
                s += lam[j]/2 * np.sum(F[count]*(params[k][1]-params_t[n*j + k][1])**2)
                count += 1
        l += s


        return l 

    def error(self,params, batch):
      images, targets = batch
      preds = self.batched_predict(params, images)
      l =  np.mean((targets.flatten()- preds.flatten())**2)
      return l
        
        # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, params_t, batch_low,  F, lam):
        params = self.get_params(opt_state)

        g = grad(self.loss)(params, params_t, batch_low, F, lam)
        return self.opt_update(i, g, opt_state)
    

    # Optimize parameters in a loop
    def train(self, train_dataset, test_dataset, F,  nIter = 10000, lam = 0.0 ):
        train_data = iter(train_dataset)
        test_data = iter(test_dataset)

            
        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Fetch data
            train_batch= next(train_data)
            test_batch= next(test_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, self.params_t, 
                                       train_batch, F, lam)
            
            if it % 100 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                train_err = self.error(params, train_batch)
                test_err = self.error(params, test_batch)

                # Store losses
                self.error_training_log.append(train_err)
                self.error_test_log.append(test_err)

                pbar.set_postfix({'Train': train_err, 'Test': test_err})
                    

    
    def predict_full(self, params, U_star):
        s_pred = self.batched_predict(params, U_star)
        return s_pred


    
    @partial(jit, static_argnums=(0,))
    def predict_log(self, params, U_star):
        pred = self.batched_predict(params, U_star)
        return np.sum(pred**2)
    
    
    def compute_MAS(self, params, u, key,  num_samples=200, plot_diffs=True, disp_freq=1, scaled=False):
    
        branch_layers = len(params)
        # initialize Fisher information for most recent task
        F_accum = []
        for k in range(branch_layers):
            F_accum.append(np.zeros(params[k][0].shape))
            F_accum.append(np.zeros(params[k][1].shape))

        for i in range(num_samples):
            # select random input image
            key, subkey = random.split(key)
    
            idx = random.choice(subkey, u.shape[0], (1,), replace=False)
          #  print(idx)
            ders = grad(self.predict_log)(params, u[idx,  :])
            for k in range(branch_layers):
                F_accum[2*k] += np.abs(ders[k][0])
                F_accum[2*k+1] += np.abs(ders[k][1])

    
        for v in range(len(F_accum)):
            F_accum[v] /= (num_samples)
         #   print(F_accum[v])
        if scaled:
            flat_MAS, _  = ravel_pytree(F_accum)
            q = np.quantile(flat_MAS, q=.99)
            for v in range(len(F_accum)):
                      F_accum[v] /= q
        return F_accum


    

