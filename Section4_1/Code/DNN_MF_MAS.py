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
from jax import lax
from jax.flatten_util import ravel_pytree

import itertools
from functools import partial
from tqdm import trange, tqdm
from copy import deepcopy

def nonlinear_DNN(branch_layers, activation=relu):

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
        return branch_params
        
    def apply(params, u):
      #  print(u.shape)
        for k in range(len(branch_layers)-1):
            W_b, b_b = params[k]
       #     print(u.shape)
       #     print(W_b.shape)
       #    print(b_b.shape)
            u = (np.dot(u, W_b) + b_b)
        return u

    return init, apply



class  MF_DNN_MAS:
    
    # Initialize the class
    def __init__(self, layers_branch_l,
                     layers_branch_nl, lr, restart = 0, params_t = 0): 
        
        self.init_nl, self.apply_nl, self.weight_nl = nonlinear_DNN(layers_branch_nl)
        self.init_l, self.apply_l = linear_DNN(layers_branch_l)

        self.params_t = params_t


        if restart == 1:
            params_nl = params_t[0]
            params_l = params_t[1]
        else:
            params_nl = self.init_nl(random.PRNGKey(13))
            params_l = self.init_l(random.PRNGKey(12345))
        
        params = (params_nl, params_l)
        
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(lr)
        
        self.opt_state = self.opt_init(params)

        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)

        self.itercount = itertools.count()

        # building loss function
        self.loss_best = 1e5
        self.error_training_log = []
        self.error_test_log = []
        self.loss_total_log = []

                
    # =============================================
    # evaluation
    # =============================================

    def predict(self, params, image, u_A):
      # per-example predictions
      params_nl, params_l = params

      logits_nl = self.apply_nl(params_nl, image)
      logits_l = self.apply_l(params_l, u_A)
      logits_l = logits_l[:, 0]
      pred = logits_nl + logits_l 
      return pred

    @partial(jit, static_argnums=(0,))
    def batched_predict(self, params, images, u_A):
        batched_predict = vmap(self.predict, in_axes=(None, 0, 0))(params, images, u_A)
        return batched_predict

    @partial(jit, static_argnums=(0,))
    def loss(self, params, params_t, batch_low, F, lam):
        inputs, targets = batch_low
        images, u_A = inputs
        params_nl, params_l = params

        preds = self.batched_predict(params, images, u_A)

        l =  np.mean((targets.flatten()- preds.flatten())**2)+  .0001*(self.weight_nl(params_nl)) 
   
        count = 0
        s = 0.0

        for j in range(len(lam)):
            for k in range(len(params[0])):
                s += lam[j]/2 * np.sum(F[count]*(params[0][k][0]-params_t[2*j + 0][k][0])**2)
                count += 1
                s += lam[j]/2 * np.sum(F[count]*(params[0][k][1]-params_t[2*j + 0][k][1])**2)
                count += 1
            for k in range(len(params[1])):
                s += lam[j]/2 * np.sum(F[count]*(params[1][k][0]-params_t[2*j + 1][k][0])**2)
                count += 1
                s += lam[j]/2 * np.sum(F[count]*(params[1][k][1]-params_t[2*j + 1][k][1])**2)
                count += 1
        l += s
        return l 
    
    

    def loss_high(self, params, batch_low):
       inputs, targets = batch_low
       images, u_A = inputs
       preds = self.batched_predict(params, images, u_A)

       return np.mean((targets.flatten()- preds.flatten())**2)
    
        # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, params_t,  batch, MAS, lam):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, params_t, batch, MAS, lam)
        return self.opt_update(i, g, opt_state)
    

    # Optimize parameters in a loop
    def train(self, train_dataset, test_dataset, MAS, lam, nIter = 10000):
        low_data = iter(train_dataset)
        test_data = iter(test_dataset)

        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Fetch data
            low_batch= next(low_data)
            test_batch= next(test_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, self.params_t, low_batch, MAS, lam)
            
            if it % 100 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses

                
                
                train_acc = self.loss_high(params, low_batch)
                test_acc = self.loss_high(params, test_batch)
                loss_t = self.loss(params, self.params_t, low_batch, MAS, lam)

                # Store losses
                self.error_training_log.append(train_acc)
                self.error_test_log.append(test_acc)
                self.loss_total_log.append(loss_t)

                pbar.set_postfix({'Train': train_acc, 'Test': test_acc,
                                      'Loss total': loss_t})

                    
                    
                # Print losses

                

    def predict_full(self, params, U_star, Ulin):
        s_pred = self.batched_predict(params, U_star, Ulin)
        return s_pred

    def predict_nl(self, params, image):
      # per-example predictions
      params_nl, params_l = params

      logits_nl = self.apply_nl(params_nl, image)
      return logits_nl

    @partial(jit, static_argnums=(0,))
    def batched_predict_nl(self, params, images):
        batched_predict = vmap(self.predict_nl, in_axes=(None, 0))(params, images)
        return batched_predict
    
    def predict_l(self, params,  u_A):
      # per-example predictions
      params_nl, params_l = params

      logits_l = self.apply_l(params_l, u_A)
      logits_l = logits_l[:, 0]
      return logits_l

    @partial(jit, static_argnums=(0,))
    def batched_predict_l(self, params,  u_A):
        batched_predict = vmap(self.predict_l, in_axes=(None, 0))(params, u_A)
        return batched_predict    
    
    @partial(jit, static_argnums=(0,))
    def predict_log_nl(self, params, U_star):
        pred = self.batched_predict_nl(params, U_star)
        return np.sum(pred[0][0][0]**2)
    
    
    @partial(jit, static_argnums=(0,))
    def predict_log_l(self, params, Ulin):
        pred = self.batched_predict_l(params,  Ulin)
        return np.sum(pred[0][0]**2)
    
    
    
    def compute_MAS(self, params, u, ulin, key,  num_samples=200, plot_diffs=True, disp_freq=1, scale=False):
    
        # initialize Fisher information for most recent task
        F_accum = []
        for k in range(len(params[0])):
            F_accum.append(np.zeros(params[0][k][0].shape))
            F_accum.append(np.zeros(params[0][k][1].shape))
        for k in range(len(params[1])):
            F_accum.append(np.zeros(params[1][k][0].shape))
            F_accum.append(np.zeros(params[1][k][1].shape))

    
        for i in range(num_samples):
            # select random input image
            key, subkey = random.split(key)
    
            idx = random.choice(subkey, u.shape[0], (1,), replace=False)

            ders_nl = grad(self.predict_log_nl)(params, u[idx,  :])
            ders_l = grad(self.predict_log_l)(params, ulin[idx,  :])
            count = 0
            for k in range(len(params[0])):
                F_accum[count] += np.abs(ders_nl[0][k][0])
                count += 1
                F_accum[count] += np.abs(ders_nl[0][k][1])
                count += 1
            for k in range(len(params[1])):
                F_accum[count] += np.abs(ders_l[1][k][0])
                count += 1
                F_accum[count] += np.abs(ders_l[1][k][1])
                count += 1

    
        for v in range(len(F_accum)):
            F_accum[v] /= (num_samples)
         
        
        if scale: 
            flat_MAS, _  = ravel_pytree(F_accum)

            q = np.quantile(flat_MAS, q=.99) + 1e-12
            for v in range(len(F_accum)):
                     F_accum[v] /= q
             
        
        return F_accum
