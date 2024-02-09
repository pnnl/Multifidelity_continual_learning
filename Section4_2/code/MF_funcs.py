#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created October 2023
@author: Amanda Howard (amanda.howard@pnnl.gov)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import jax
import jax.numpy as np
from jax import random, grad, vmap, jit
from jax.example_libraries import optimizers
from jax.nn import relu, elu,selu
#from jax.ops import index_update, index
from jax.flatten_util import ravel_pytree

import itertools
from functools import partial
from tqdm import trange
from copy import deepcopy

import numpy as onp

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




class DNN_class:
    
    # Initialize the class
    def __init__(self, layers_branch_low, lr , restart =0, params_t = 0, activation_func=np.tanh): 


        #Network initialization 
        self.init_low, self.apply_low = DNN(layers_branch_low, activation=activation_func)
        self.params_t = params_t
        if restart ==1:
            params_low = self.params_t[-1]
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
        self.loss_training_log = []
        self.loss_testing_log = []


    # =============================================
    # evaluation
    # =============================================

    # Define DeepONet architecture
    def operator_net(self, params, x):
        B = self.apply_low(params, x)
        return B[0]

  

    def loss_data(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u = inputs
        

        # Compute forward pass
        s1_pred =vmap(self.operator_net, (None, 0))(params, u)
        # Compute loss

        loss_s1 = np.mean((outputs.flatten() - s1_pred.flatten())**2)

        loss = loss_s1
        return loss
    

    
    # Define total loss
    def loss(self, params, params_t, data_batch,  F, lam):
        loss_data = self.loss_data(params, data_batch)

        loss =  loss_data
        
        count = 0
        s = 0.0

        for j in range(len(lam)):
            count = 0
            for k in range(len(params)):
                s += lam[j]/2 * np.sum(F[j][count]*(params[k][0]-params_t[j][k][0])**2)
                count += 1
                s += lam[j]/2 * np.sum(F[j][count]*(params[k][1]-params_t[j][k][1])**2)
                count += 1
        loss += s



                
        return loss

    
    
    
        # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, params_t, data_batch, F, lam):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, params_t, data_batch, F, lam)

        return self.opt_update(i, g, opt_state)
    

    # Optimize parameters in a loop
    def train(self, train_dataset, test_dataset, nIter = 10000, F = 0, lam = []):
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
            
            if it % 1000 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, self.params_t, train_batch,  F, lam)
                
                #train_value = self.loss_data(params, train_batch)
                test_value = self.loss_data(params, test_batch)

                # Store losses
                self.loss_training_log.append(loss_value)
                self.loss_testing_log.append(test_value)

                # Print losses
                pbar.set_postfix({'Loss': "{0:.4f}".format(loss_value), 
                                  'Test': "{0:.4f}".format(test_value)})

  #  @partial(jit, static_argnums=(0,))
    def predict_u(self, params, U_star):

        
      #  s_pred = vmap(self.apply_low, (None, 0, 0))(params_low, U_star, Y_star)
        s_pred =vmap(self.operator_net, (None, 0))(params, U_star)
        return s_pred


    
    @partial(jit, static_argnums=(0,))
    def predict_log(self, params, U_star):
        pred = self.predict_u(params, U_star)

        return np.sum(pred[0]**2)    
    
    def compute_MAS(self, params, u, key,  num_samples=200, plot_diffs=True, disp_freq=1):
    
        branch_layers = len(params)
        # initialize Fisher information for most recent task
        F_accum = []
        for k in range(branch_layers):
            F_accum.append(np.zeros(params[k][0].shape))
            F_accum.append(np.zeros(params[k][1].shape))
    
        if(plot_diffs):
            # track differences in mean Fisher info
            F_prev = deepcopy(F_accum)
            mean_diffs = np.zeros(0)
    
        for i in range(num_samples):
            # select random input image
            key, subkey = random.split(key)
    

            idx = random.choice(subkey, u.shape[0], (1,), replace=False)
            
            
            ders = grad(self.predict_log)(params, u[idx,  :])
            for k in range(branch_layers):
                F_accum[2*k] += np.abs(ders[k][0])
                F_accum[2*k+1] += np.abs(ders[k][1])
    
            if(plot_diffs):
                if i % disp_freq == 0 and i > 0:
                    # recording mean diffs of F
                    F_diff = 0
                    for v in range(len(F_accum)):
                        F_diff += np.sum(np.absolute(F_accum[v]/(i+1) - F_prev[v]))
                    mean_diff = np.mean(F_diff)
                    mean_diffs = np.append(mean_diffs, mean_diff)
                    print(mean_diff)

                    for v in range(len(F_accum)):
                        F_prev[v] = F_accum[v]/(i+1)
                        
                        
 #       i = num_samples -1
  #      plt.semilogy(range(disp_freq+1, i+2, disp_freq), mean_diffs)
   #     plt.xlabel("Number of samples")
    #    plt.ylabel("Mean absolute Fisher difference")
    
        # divide totals by number of samples
    
        for v in range(len(F_accum)):
            F_accum[v] /= (num_samples)
         #   print(F_accum[v])
        
        return F_accum



class MF_DNN_class:
    
    # Initialize the class
    def __init__(self, layers_branch_nl, layers_branch_l, layers_branch_lf, lr ,
                 params_A, restart =0, params_t = []): 

        self.init_nl, self.apply_nl, self.weight_nl = nonlinear_DNN(layers_branch_nl)
        self.init_l, self.apply_l = linear_DNN(layers_branch_l)
        self.init_lf, self.apply_lf = DNN(layers_branch_lf)


        self.params_t = params_t

        self.params_A = params_A

        if restart == 1 and len(self.params_t) > 0:
            params_nl = self.params_t[-1][0]
            params_l = self.params_t[-1][1]
        else:
            params_nl = self.init_nl(random.PRNGKey(13))
            params_l = self.init_l(random.PRNGKey(12345))
        params = (params_nl, params_l)
        self.restart = restart

        
        
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(lr)
     
        self.opt_state = self.opt_init(params)

        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)

        self.itercount = itertools.count()


        # building loss function
        self.loss_training_log = []
        self.loss_testing_log = []
        self.loss_training_train_log =[]

    # =============================================
    # evaluation
    # =============================================

    # Define DeepONet architecture
    def operator_net(self, params, u):

        ul = self.apply_lf(self.params_A, u)
        for i in onp.arange(len(self.params_t)): 
            paramsB_nl =  self.params_t[i][0]
            paramsB_l =  self.params_t[i][1]
            y = np.hstack([u, ul])
            B_lin = self.apply_l(paramsB_l, ul)
            B_nonlin = self.apply_nl(paramsB_nl, y)

            ul = B_nonlin + B_lin 
        
        params_nl, params_l = params
        y = np.hstack([u, ul])

        logits_nl = self.apply_nl(params_nl, y)
        logits_l = self.apply_l(params_l, ul)
        logits_l = logits_l
        pred = logits_nl + logits_l 

        
        return pred[0]
  

    def loss_data(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u = inputs
        

        # Compute forward pass
        s1_pred =vmap(self.operator_net, (None, 0))(params, u)
        # Compute loss

        loss_s1 = np.mean((outputs.flatten() - s1_pred.flatten())**2)

        loss = loss_s1
        return loss
    

    
    # Define total loss
    def loss(self, params, data_batch,  F, lam):
        loss_data = self.loss_data(params, data_batch)

        loss =  loss_data
        
        params_nl, params_l = params
        
        
        count = 0
        s = 0.0
        n = len(params)

        for j in range(len(lam)):
            count = 0
            for k in range(len(params[0])):
                s += lam[j]/2 * np.sum(F[j][count]*(params[0][k][0]-self.params_t[j][0][k][0])**2)
                count += 1
                s += lam[j]/2 * np.sum(F[j][count]*(params[0][k][1]-self.params_t[j][0][k][1])**2)
                count += 1
            for k in range(len(params[1])):
                s += lam[j]/2 * np.sum(F[j][count]*(params[1][k][0]-self.params_t[j][1][k][0])**2)
                count += 1
                s += lam[j]/2 * np.sum(F[j][count]*(params[1][k][1]-self.params_t[j][1][k][1])**2)
                count += 1
        
        loss += s
        loss += .00001*(self.weight_nl(params_nl)) 

        return loss

    
        # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, data_batch, F, lam):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, data_batch, F, lam)

        return self.opt_update(i, g, opt_state)
    

    # Optimize parameters in a loop
    def train(self, train_dataset, test_dataset, nIter = 10000, F = 0, lam = []):
        train_data = iter(train_dataset)
        test_data = iter(test_dataset)

        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Fetch data
            train_batch= next(train_data)
            test_batch= next(test_data)


            self.opt_state = self.step(next(self.itercount), self.opt_state, 
                                       train_batch, F, lam)
            
            if it % 1000 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, train_batch,  F, lam)
                
                #train_value = self.loss_data(params, train_batch)
                test_value = self.loss_data(params, test_batch)
                train_value = self.loss_data(params, train_batch)

                # Store losses
                self.loss_training_log.append(loss_value)
                self.loss_training_train_log.append(train_value)
                self.loss_testing_log.append(test_value)

                # Print losses
                pbar.set_postfix({'Loss': "{0:.4f}".format(loss_value), 
                                  'Test': "{0:.4f}".format(test_value),
                                  'Train': "{0:.4f}".format(train_value)})

  #  @partial(jit, static_argnums=(0,))
    def predict_u(self, params, U_star):

        
      #  s_pred = vmap(self.apply_low, (None, 0, 0))(params_low, U_star, Y_star)
        s_pred =vmap(self.operator_net, (None, 0))(params, U_star)
        return s_pred


    def operator_net_nl(self, params, u):

        ul = self.apply_lf(self.params_A, u)
        for i in onp.arange(len(self.params_t)): 
            paramsB_nl =  self.params_t[i][0]
            paramsB_l =  self.params_t[i][1]
            y = np.hstack([u, ul])
            B_lin = self.apply_l(paramsB_l, ul)
            B_nonlin = self.apply_nl(paramsB_nl, y)

            ul = B_nonlin + B_lin 
        
        params_nl, params_l = params
        y = np.hstack([u, ul])

        logits_nl = self.apply_nl(params_nl, y)

        return logits_nl[0]
  
    def operator_net_l(self, params, u):

        ul = self.apply_lf(self.params_A, u)
        for i in onp.arange(len(self.params_t)): 
            paramsB_nl =  self.params_t[i][0]
            paramsB_l =  self.params_t[i][1]
            y = np.hstack([u, ul])
            B_lin = self.apply_l(paramsB_l, ul)
            B_nonlin = self.apply_nl(paramsB_nl, y)

            ul = B_nonlin + B_lin 
        
        params_nl, params_l = params
        y = np.hstack([u, ul])

        logits_l = self.apply_l(params_l, ul)

        return logits_l[0]
  
    def predict_u_nl(self, params, U_star):
        s_pred =vmap(self.operator_net_nl, (None, 0))(params, U_star)
        return s_pred
    def predict_u_l(self, params, U_star):
        s_pred =vmap(self.operator_net_nl, (None, 0))(params, U_star)
        return s_pred


    
    @partial(jit, static_argnums=(0,))
    def predict_log_nl(self, params, U_star):
        pred = self.predict_u_nl(params, U_star)
        return np.sum(pred[0]**2)
    
    
    @partial(jit, static_argnums=(0,))
    def predict_log_l(self, params, U_star):
        pred = self.predict_u_l(params,  U_star)
        return np.sum(pred[0]**2)
    
    
    
    def compute_MAS(self, params, u, key,  num_samples=200, plot_diffs=True, disp_freq=1, scale=False):
    
        # initialize Fisher information for most recent task
        F_accum = []
        for k in range(len(params[0])):
            F_accum.append(np.zeros(params[0][k][0].shape))
            F_accum.append(np.zeros(params[0][k][1].shape))
        for k in range(len(params[1])):
            F_accum.append(np.zeros(params[1][k][0].shape))
            F_accum.append(np.zeros(params[1][k][1].shape))
            
    
        if(plot_diffs):
            # track differences in mean Fisher info
            F_prev = deepcopy(F_accum)
            mean_diffs = np.zeros(0)
    
        for i in range(num_samples):
            # select random input image
            key, subkey = random.split(key)
    
            idx = random.choice(subkey, u.shape[0], (1,), replace=False)

            ders_nl = grad(self.predict_log_nl)(params, u[idx,  :])
            ders_l = grad(self.predict_log_l)(params, u[idx,  :])
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
                
                    
            if(plot_diffs):
                if i % disp_freq == 0 and i > 0:
                    # recording mean diffs of F
                    F_diff = 0
                    for v in range(len(F_accum)):
                        F_diff += np.sum(np.absolute(F_accum[v]/(i+1) - F_prev[v]))
                    mean_diff = np.mean(F_diff)
                    print(mean_diff)
                    mean_diffs = np.append(mean_diffs, mean_diff)
                    for v in range(len(F_accum)):
                        F_prev[v] = F_accum[v]/(i+1)
                        
     #   if(plot_diffs):
                
     #       i = num_samples -1
     #       plt.semilogy(range(disp_freq+1, i+2, disp_freq), mean_diffs)
     #       plt.xlabel("Number of samples")
     #       plt.ylabel("Mean absolute Fisher difference")
        
        # divide totals by number of samples
    
        for v in range(len(F_accum)):
            F_accum[v] /= (num_samples)
         #   print(F_accum[v])
         
        
        if scale: 
            flat_EWC, _  = ravel_pytree(F_accum)

            q = np.quantile(flat_EWC, q=.99) + 1e-12
            for v in range(len(F_accum)):
                     F_accum[v] /= q
             
        
        return F_accum
