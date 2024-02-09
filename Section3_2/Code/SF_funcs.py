"""
Created 2023
@author: Amanda Howard (amanda.howard@pnnl.gov)

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from utils import DNN
import jax
import jax.numpy as np
from jax import random, grad, vmap, jit
from jax.example_libraries import optimizers
from jax.nn import relu, selu
from jax.flatten_util import ravel_pytree

import itertools
from functools import partial
from torch.utils import data
from tqdm import trange
from copy import deepcopy



class DataGenerator_res(data.Dataset):
    def __init__(self, dim, coords,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.dim = dim
        self.coords = coords
        
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs = self.__data_generation(subkey)
        return inputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        x = self.coords[0:1,:] + (self.coords[1:2,:]-self.coords[0:1,:])*random.uniform(key, shape=[self.batch_size, self.dim])
        inputs = x
        return inputs

class DataGenerator_bcs(data.Dataset):
    def __init__(self, dim, coords1, coords2,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.dim = dim
        self.coords1 = coords1
        self.coords2 = coords2
        
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs = self.__data_generation(subkey)
        return inputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        xval = random.uniform(key, shape=[self.batch_size, self.dim])
        x1 = self.coords1[0:1,:] + (self.coords1[1:2,:]-self.coords1[0:1,:])*xval
        x2 = self.coords2[0:1,:] + (self.coords2[1:2,:]-self.coords2[0:1,:])*xval
        inputs = x1, x2
        return inputs

class DataGenerator_ICS(data.Dataset):
    def __init__(self, dim, coords, model, params, 
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.dim = dim
        self.coords = coords
        self.params = params
        self.model = model
        
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
        x = self.coords[0:1,:] + (self.coords[1:2,:]-self.coords[0:1,:])*random.uniform(key, shape=[self.batch_size, self.dim])
        y = self.model.predict_u(self.params, x)
        inputs = x
        outputs = y
        return inputs, outputs
    

class DataGenerator_ICS_A(data.Dataset):
    def __init__(self, dim, coords, func, 
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.dim = dim
        self.coords = coords
        self.func = func
        
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
        x = self.coords[0:1,:] + (self.coords[1:2,:]-self.coords[0:1,:])*random.uniform(key, shape=[self.batch_size, self.dim])
        y = self.func(x)
        inputs = x
        outputs = y
        return inputs, outputs
    
# Define the exact solution and its derivatives
def u0(x):
    """
    :param x: x = (t, x)
    """
    t = x[:,0:1]
    x = x[:,1:2]
    return x*x*np.cos(np.pi * x) 


class DNN_class:
    
    # Initialize the class
    def __init__(self, layers_branch_low, ics_weight, res_weight, ut_weight, lr , restart =0, params_t = 0, params_i = 0): 

        #Network initialization 
        self.init_low, self.apply_low = DNN(layers_branch_low, activation=np.tanh)
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

        self.ics_weight = ics_weight
        self.res_weight = res_weight
        self.ut_weight = ut_weight


        # building loss function
        self.loss_training_log = []
        self.loss_res_log = []
        self.loss_ics_log = []
        self.loss_ut_log = []

    # =============================================
    # evaluation
    # =============================================

    # Define DNN architecture
    def operator_net(self, params, x, t):
        y = np.stack([t,x])
        B = self.apply_low(params, y)
        return B[0]
    
      # Define ODE residual
    def residual_net(self, params, u):
          x = u[1]
          t = u[0]
          
          s_xx = grad(grad(self.operator_net, argnums= 1), argnums= 1)(params, x, t)
          s_t = grad(self.operator_net, argnums= 2)(params, x, t)
          s = self.operator_net(params, x, t)
    
          res = s_t - 0.0001*s_xx+5*s**3-5.0*s
          return res
    
    def ux_net(self, params, u):
          x = u[1]
          t = u[0]
          
          s_t = grad(self.operator_net, argnums= 1)(params, x, t)
          return s_t
      
      # Define initial loss
    def loss_ics(self, params, batch):
          # Fetch data
          inputs, outputs = batch
          u = inputs
          x = u[:, 1]
          t = u[:, 0]

          # Compute forward pass
          s1_pred =vmap(self.operator_net, (None, 0, 0))(params, x, t)
          # Compute loss
          loss = np.mean((outputs.flatten() - s1_pred.flatten())**2)
          return loss
      
      # Define residual loss
    def loss_res(self, params, batch):
          # Fetch data
          inputs = batch
          u = inputs
    
          # Compute forward pass
          res1_pred  = vmap(self.residual_net, (None, 0))(params, u)
          loss_res = np.mean((res1_pred)**2)
          return loss_res   
    
    def loss_bcs(self, params, batch):
          # Fetch data
          inputs = batch
          u1, u2 = inputs
          x1 = u1[:, 1]
          t1 = u1[:, 0]
          x2 = u2[:, 1]
          t2 = u2[:, 0]
          
          # Compute forward pass
          s1_pred =vmap(self.operator_net, (None, 0, 0))(params, x1, t1)
          s2_pred =vmap(self.operator_net, (None, 0, 0))(params, x2, t2)
    
          # Compute loss
          loss_s_bc = np.mean((s1_pred - s2_pred)**2)
          return loss_s_bc
    
    def loss_bcs_x(self, params, batch):
          # Fetch data
          inputs = batch
          u1, u2 = inputs
    
          # Compute forward pass
          s_x_bc1_pred = vmap(self.ux_net, (None, 0))(params, u1)
          s_x_bc2_pred = vmap(self.ux_net, (None, 0))(params, u2)
    
          # Compute loss
          loss_s_x_bc = np.mean((s_x_bc1_pred - s_x_bc2_pred)**2)
          return loss_s_x_bc  

    # Define total loss
    def loss(self, params, params_t, ics_batch, bc1_batch, bc2_batch, res_batch, F, lam):
        loss_ics = self.loss_ics(params, ics_batch)
        loss_bcs = self.loss_bcs(params, bc1_batch)
        loss_bcs_x = self.loss_bcs_x(params, bc2_batch)
        loss_res = self.loss_res(params, res_batch)

        loss =  self.ics_weight*(loss_ics)\
                + self.res_weight*loss_res \
                + self.ut_weight*(loss_bcs + loss_bcs_x)
        count = 0
        s = 0.0
        n = len(params)

        for j in range(len(lam)):
            for k in range(len(params)):
                s += lam[j]/2 * np.sum(F[count]*(params[k][0]-params_t[n*j + k][0])**2)
                count += 1
                s += lam[j]/2 * np.sum(F[count]*(params[k][1]-params_t[n*j + k][1])**2)
                count += 1
        loss += s
                
        return loss

    
        # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, params_t, ics_batch, bc1_batch, bc2_batch, res_batch, F, lam):
        params = self.get_params(opt_state)

        g = grad(self.loss)(params, params_t, ics_batch, bc1_batch, bc2_batch, res_batch, F, lam)
        return self.opt_update(i, g, opt_state)
    

    # Optimize parameters in a loop
    def train(self, ics_dataset, bc1_dataset, bc2_dataset, res_dataset, nIter = 10000, F = 0, lam = []):
        res_data = iter(res_dataset)
        ics_data = iter(ics_dataset)
        bc1_data = iter(bc1_dataset)
        bc2_data = iter(bc2_dataset)

        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Fetch data
            res_batch= next(res_data)
            ics_batch= next(ics_data)
            bc1_batch= next(bc1_data)
            bc2_batch= next(bc2_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, self.params_t, ics_batch, bc1_batch, bc2_batch, res_batch, F, lam)
            
            if it % 1000 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, self.params_t, ics_batch, bc1_batch, bc2_batch, res_batch, F, lam)
                
                res_value = self.loss_res(params, res_batch)
                ics_value = self.loss_ics(params, ics_batch)
                bcs_value = self.loss_bcs(params, bc1_batch)+self.loss_bcs_x(params, bc2_batch)

                # Store losses
                self.loss_training_log.append(loss_value)
                self.loss_res_log.append(res_value)
                self.loss_ics_log.append(ics_value)
                self.loss_ut_log.append(bcs_value)

                # Print losses
                pbar.set_postfix({'Loss': "{0:.4f}".format(loss_value), 
                                  'Res': "{0:.4f}".format(res_value), 
                                  'ICS': "{0:.4f}".format(ics_value),
                                  'BCS': "{0:.4f}".format(bcs_value)})

    # Evaluates predictions at test points  
  #  @partial(jit, static_argnums=(0,))
    def predict_u(self, params, U_star):
        x = U_star[:, 1]
        t = U_star[:, 0]
        s_pred =vmap(self.operator_net, (None, 0, 0))(params, x, t)
        return s_pred

    def predict_ut(self, params, U_star):
        s_pred =vmap(self.ut_net, (None, 0))(params, U_star)
        return s_pred
    
    def predict_res(self, params, U_star):
        s_pred =vmap(self.residual_net, (None, 0))(params,  U_star)
        return s_pred
    
    @partial(jit, static_argnums=(0,))
    def predict_l2(self, params, U_star):
        pred = self.predict_u(params, U_star)

        return np.sum(pred**2)
    
    def compute_MAS(self, params, coords, key,  num_samples=200, plot_diffs=True, disp_freq=1, scale=False):
    
        branch_layers = len(params)


        F_accum = []
        for k in range(branch_layers):
            F_accum.append(np.zeros(params[k][0].shape))
            F_accum.append(np.zeros(params[k][1].shape))
    

        for i in range(num_samples):
            # select random input image
            key, subkey = random.split(key)
    
            u = coords[0:1,:] + (coords[1:2,:]- coords[0:1,:])*random.uniform(key, shape=[1, 2])
            ders = grad(self.predict_l2)(params, u)
            for k in range(branch_layers):
                F_accum[2*k] += np.abs(ders[k][0])
                F_accum[2*k+1] += np.abs(ders[k][1])

        for v in range(len(F_accum)):
            F_accum[v] /= (num_samples)
 
        flat_MAS, _  = ravel_pytree(F_accum)
        if scale: 
            q = np.quantile(flat_MAS, q=.999)
            for v in range(len(F_accum)):
                        F_accum[v] /= q
    
        return F_accum


                    
