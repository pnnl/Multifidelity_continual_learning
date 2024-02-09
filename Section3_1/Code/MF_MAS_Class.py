#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created October 2023
@author: Amanda Howard (amanda.howard@pnnl.gov)
"""

import os

#import numpy as np
import scipy.io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import time
from utils import  nonlinear_DNN, linear_DNN, DNN
import math
import jax
import jax.numpy as np
from jax import random, grad, vmap, jit, hessian
from jax.example_libraries import optimizers
from jax.experimental.ode import odeint
from jax.nn import relu, elu, log_softmax, softmax
from jax.config import config
#from jax.ops import index_update, index
from jax import lax
from jax.flatten_util import ravel_pytree

import itertools
from functools import partial
from torch.utils import data
from tqdm import trange, tqdm
#import matplotlib.pyplot as plt
#import pandas as pd
#import matplotlib.pyplot as plt
from copy import deepcopy


class MF_class_MAS:
    
    # Initialize the class
    def __init__(self, layers_branch_l,layers_trunk_l,
                     layers_branch_nl, layers_A, 
                     ics_weight, res_weight, data_weight, pen_weight, params_A, params_prev, lr): 

        self.init_nl, self.apply_nl, self.weight_nl = nonlinear_DNN(layers_branch_nl)
        self.init_l, self.apply_l = linear_DNN(layers_branch_l)
        self.init_A, self.apply_A = DNN(layers_A)


        
        params_nl = self.init_nl(random.PRNGKey(13))
        params_l = self.init_l(random.PRNGKey(23))
        self.params_A = params_A 
        self.params_prev = params_prev

        if len(params_prev) > 0:
            paramsB_nl= params_prev[-2]
            paramsB_l = params_prev[-1]
            params_nl = paramsB_nl
            params_l = paramsB_l
        params = (params_nl, params_l)

        self.ics_weight = ics_weight
        self.res_weight = res_weight
        self.data_weight = data_weight
        self.pen_weight = pen_weight
        
        
        
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(lr)
     
        self.opt_state = self.opt_init(params)

        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)

        self.itercount = itertools.count()

        self.ics_weight = ics_weight
        self.res_weight = res_weight
        self.data_weight = data_weight


        # building loss function
        self.loss_training_log = []
        self.loss_res_log = []
        self.loss_ics_log = []
        self.loss_data_log = []

    # =============================================
    # evaluation
    # =============================================

    # Define DeepONet architecture
    def operator_net(self, params, u):
        params_nl, params_l = params
        ul = self.apply_A(self.params_A, u) 

        l = int(len(self.params_prev)/2)
        for i in range(l): 
            paramsB_nl =  self.params_prev[2*i]
            paramsB_l =  self.params_prev[2*i+1]
            in_u = np.hstack([u, ul])

            B_lin = self.apply_l(paramsB_l, ul)
           # B_lin = self.apply_l(paramsB_l,in_u)
            B_nonlin = self.apply_nl(paramsB_nl, in_u)
            ul = B_lin+B_nonlin

        in_u=np.hstack([u, ul])
        B_lin = self.apply_l(params_l, ul)
        B_nonlin = self.apply_nl(params_nl,in_u)


        
     #   print("opp net")
     #   print(B.shape)
#
        s1 = B_lin[:1]+ B_nonlin[:1]
        s2 = B_lin[1:]+ B_nonlin[1:]
        
      #  print(s1.shape)

        return s1, s2


    # Define ODE residual
    def residual_net(self, params, u):

        s1, s2 = self.operator_net(params, u)
     #   print(s1.shape)
     #   print(u.shape)


        def s1_fn(params, u):
          s1_fn, _ = self.operator_net(params, u)
       #   print(s1_fn.shape)
          return s1_fn[0]
        
        def s2_fn(params, u):
          _, s2_fn  = self.operator_net(params, u)
          return s2_fn[0]

        s1_y = grad(s1_fn, argnums= 1)(params, u)
        s2_y = grad(s2_fn, argnums= 1)(params, u)

        res_1 = s1_y - s2
        res_2 = s2_y + 0.05 * s2 + 9.81 * np.sin(s1)

        return res_1, res_2


    def loss_data(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u = inputs
        
        s1 = outputs[:, 0:1]
        s2 = outputs[:, 1:2]

        # Compute forward pass
        s1_pred, s2_pred =vmap(self.operator_net, (None, 0))(params, u)
        # Compute loss

        loss_s1 = np.mean((s1.flatten() - s1_pred.flatten())**2)
        loss_s2 = np.mean((s2.flatten() - s2_pred.flatten())**2)

        loss = loss_s1 + loss_s2
        return loss
    
    # Define residual loss
    def loss_res(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u = inputs

        # Compute forward pass
        res1_pred, res2_pred = vmap(self.residual_net, (None, 0))(params, u)
        # Compute loss

        loss_res1 = np.mean((res1_pred)**2)
        loss_res2 = np.mean((res2_pred)**2)
        loss_res = loss_res1 + loss_res2
        return loss_res   

    # Define total loss
    def loss(self, params, params_prev, ic_batch, res_batch, val_batch, lam, F):
        loss_ics = self.loss_data(params, ic_batch)
        loss_res = self.loss_res(params, res_batch)
        loss_data = self.loss_data(params, val_batch)
        loss =  self.ics_weight*loss_ics + self.res_weight*loss_res + self.data_weight*loss_data
        params_nl, params_l = params

        count = 0
        s = 0.0
        
        l = len(params)

        for j in range(len(lam)):
            for k in range(len(params[0])):
                s += lam[j]/2 * np.sum(F[count]*(params[0][k][0]-params_prev[l*j + 0][k][0])**2)
                count += 1
                s += lam[j]/2 * np.sum(F[count]*(params[0][k][1]-params_prev[l*j + 0][k][1])**2)
                count += 1
            for k in range(len(params[1])):
                s += lam[j]/2 * np.sum(F[count]*(params[1][k][0]-params_prev[l*j + 1][k][0])**2)
                count += 1
                s += lam[j]/2 * np.sum(F[count]*(params[1][k][1]-params_prev[l*j + 1][k][1])**2)
                count += 1

                                
        loss =  self.ics_weight*loss_ics + self.res_weight*loss_res +\
            self.data_weight*loss_data+ self.pen_weight*(self.weight_nl(params_nl)) + s
            
        return loss 
    

    
        # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, ic_batch, res_batch, val_batch, lam, F, params_prev):
        params = self.get_params(opt_state)

        g = grad(self.loss)(params, params_prev, ic_batch, res_batch, val_batch, lam, F)
        return self.opt_update(i, g, opt_state)
    

    # Optimize parameters in a loop
    def train(self, ic_dataset, res_dataset, val_dataset, lam, F, params_prev, nIter = 10000):
        res_data = iter(res_dataset)
        ic_data = iter(ic_dataset)
        val_data = iter(val_dataset)

        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Fetch data
            res_batch= next(res_data)
            ic_batch= next(ic_data)
            val_batch= next(val_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, 
                                       ic_batch, res_batch, val_batch, lam, F, params_prev)
            
            if it % 1000 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params,params_prev, ic_batch, res_batch, val_batch, lam, F)
                res_value = self.loss_res(params, res_batch)
                ics__value = self.loss_data(params, ic_batch)
                data_value = self.loss_data(params, val_batch)

                # Store losses
                self.loss_training_log.append(loss_value)
                self.loss_res_log.append(res_value)
                self.loss_ics_log.append(ics__value)
                self.loss_data_log.append(data_value)

                # Print losses
                pbar.set_postfix({'Loss': "{0:.4f}".format(loss_value), 
                                  'Res': "{0:.4f}".format(res_value), 
                                  'ICS': "{0:.4f}".format(ics__value),
                                  'Data': "{0:.4f}".format(data_value)})

    # Evaluates predictions at test points  
  #  @partial(jit, static_argnums=(0,))
    @partial(jit, static_argnums=(0,))
    def predict_full(self, params, u):
        params_nl, params_l = params

                              
        ul =vmap( self.apply_A, (None, 0))(self.params_A, u)

        l = int(len(self.params_prev)/2)

        for i in range(l): 
            paramsB_nl =  self.params_prev[2*i]
            paramsB_l =  self.params_prev[2*i+1]
            in_u = np.hstack([u, ul])
            B_lin =vmap(self.apply_l, (None, 0))(paramsB_l, ul).reshape([-1, 2])
            B_nonlin = self.apply_nl(paramsB_nl, in_u).reshape([-1, 2])
            ul = B_lin+B_nonlin
        in_u=np.hstack([u, ul])

        B_lin =vmap(self.apply_l, (None, 0))(params_l, ul).reshape([-1, 2])
        B_nonlin = self.apply_nl(params_nl, in_u).reshape([-1, 2])
        pred = B_lin+B_nonlin


        return pred

    def predict_res(self, params, u):
        res1, res2 = vmap(self.residual_net, (None, 0))(params, u)
        loss_res1 = np.mean((res1)**2, axis=1)
        loss_res2 = np.mean((res2)**2, axis=1)
        loss_res = loss_res1 + loss_res2
        return loss_res
    
    
    def predict_class_log2(self, params, u):

        params_nl, params_l = params

                              
        ul =vmap( self.apply_A, (None, 0))(self.params_A, u).reshape([-1])


        l = int(len(self.params_prev)/2)

    
        for i in range(l): 
            paramsB_nl =  self.params_prev[2*i]
            paramsB_l =  self.params_prev[2*i+1]
            in_u=np.hstack([u, ul])
            B_lin =self.apply_l(paramsB_l, ul)
            B_nonlin = self.apply_nl(paramsB_nl, in_u).reshape([-1])

            ul = B_lin+B_nonlin
        
        in_u=np.hstack([u, ul])
 

        B_lin =self.apply_l(params_l, ul)
        B_nonlin = self.apply_nl(params_nl, in_u).reshape([-1, 2])
        pred = B_lin+B_nonlin

        return np.sum(pred**2)
    
    
    def predict_class_log_nl(self, params, u):

        params_nl, params_l = params

                              
        ul =vmap( self.apply_A, (None, 0))(self.params_A, u).reshape([-1])


        l = int(len(self.params_prev)/2)

    
        for i in range(l): 
            paramsB_nl =  self.params_prev[2*i]
            paramsB_l =  self.params_prev[2*i+1]
            in_u=np.hstack([u, ul])
            B_lin =self.apply_l(paramsB_l, ul)
            B_nonlin = self.apply_nl(paramsB_nl, in_u).reshape([-1])

            ul = B_lin+B_nonlin
        
        in_u=np.hstack([u, ul])
 

        B_lin =self.apply_l(params_l, ul)
        B_nonlin = self.apply_nl(params_nl, in_u).reshape([-1, 2])
        pred = B_nonlin

        return np.sum(pred**2)
    

    
    def predict_class_log_l(self, params, u):

        params_nl, params_l = params

                              
        ul =vmap( self.apply_A, (None, 0))(self.params_A, u).reshape([-1])


        l = int(len(self.params_prev)/2)

    
        for i in range(l): 
            paramsB_nl =  self.params_prev[2*i]
            paramsB_l =  self.params_prev[2*i+1]
            in_u=np.hstack([u, ul])
            B_lin =self.apply_l(paramsB_l, ul)
            B_nonlin = self.apply_nl(paramsB_nl, in_u).reshape([-1])

            ul = B_lin+B_nonlin
        
        in_u=np.hstack([u, ul])
 

        B_lin =self.apply_l(params_l, ul)
        B_nonlin = self.apply_nl(params_nl, in_u).reshape([-1, 2])
        pred = B_lin

        return np.sum(pred**2)
    
    



    def compute_MAS(self, params, coords, key,  num_samples=200, plot_diffs=True, disp_freq=1, scale=False):
        
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
            
            key, subkey = random.split(key)

            u = coords[0] + (coords[1]-coords[0])*random.uniform(key, shape=[num_samples, 1])

            for i in range(num_samples):
                # select random input image
                dersnl = grad(self.predict_class_log_nl)(params,  u[i,  :])
                dersl = grad(self.predict_class_log_l)(params,  u[i,  :])
                
                count = 0
                for k in range(len(params[0])):
                        F_accum[count] += np.abs(dersnl[0][k][0])
                        count += 1
                        F_accum[count] += np.abs(dersnl[0][k][1])
                        count += 1
                for k in range(len(params[1])):
                        F_accum[count] += np.abs(dersl[1][k][0])
                        count += 1
                        F_accum[count] += np.abs(dersl[1][k][1])
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
                            
            if(plot_diffs):
                
                i = num_samples -1
           #     plt.semilogy(range(disp_freq+1, i+2, disp_freq), mean_diffs)
           #     plt.xlabel("Number of samples")
           #     plt.ylabel("Mean absolute Fisher difference")
        
            # divide totals by number of samples
            
            
            


            for v in range(len(F_accum)):
                F_accum[v] /= (num_samples)
             #   print(F_accum[v])
            if scale:
                flat_MAS, _  = ravel_pytree(F_accum)
                q = np.quantile(flat_MAS, q=.999)
                for v in range(len(F_accum)):
                          F_accum[v] /= q
            
            return F_accum


