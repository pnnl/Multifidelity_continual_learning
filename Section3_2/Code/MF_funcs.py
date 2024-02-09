"""
Created 2023
@author: Amanda Howard (amanda.howard@pnnl.gov)

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from utils import DNN, nonlinear_DNN, linear_DNN
import jax
import jax.numpy as np
from jax import random, grad, vmap, jit, hessian
from jax.example_libraries import optimizers
from jax.flatten_util import ravel_pytree

import itertools
from functools import partial
from tqdm import trange
from copy import deepcopy


# Define the exact solution and its derivatives
def u0(x):
    """
    :param x: x = (t, x)
    """
    t = x[:,0:1]
    x = x[:,1:2]
    return x*x*np.cos(np.pi * x) 



class MF_DNN_class:
    
    # Initialize the class
    def __init__(self, layers_branch_nl, layers_branch_l, ics_weight, res_weight, 
                 ut_weight, lr , func_prev, restart =0, params_t = 0): 

        self.init_nl, self.apply_nl, self.weight_nl = nonlinear_DNN(layers_branch_nl)
        self.init_l, self.apply_l = linear_DNN(layers_branch_l)

        self.prior_net = func_prev
        self.params_t = params_t


        if restart == 1:
            params_nl = params_t[0]
            params_l = params_t[1]
        else:
            params_nl = self.init_nl(random.PRNGKey(139))
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
        y = y.reshape([-1, 2])
        yprev = self.prior_net(y)[0]
        y = np.stack([t,x, yprev])
        
        params_nl, params_l = params

        logits_nl = self.apply_nl(params_nl, y)
        logits_l = self.apply_l(params_l, yprev)
        logits_l = logits_l[:, 0]
        pred = logits_nl + logits_l 

        return pred[0]
    
    def operator_net_nl(self, params, x, t):
        y = np.stack([t,x])
        y = y.reshape([-1, 2])
        yprev = self.prior_net(y)[0]
        y = np.stack([t,x, yprev])

        params_nl, params_l = params

        logits_nl = self.apply_nl(params_nl, y)
        pred = logits_nl

        return pred[0]
    
    def operator_net_l(self, params, x, t):
        y = np.stack([t,x])
        y = y.reshape([-1, 2])
        yprev = self.prior_net(y)[0]

        params_nl, params_l = params

        logits_l = self.apply_l(params_l, yprev)
        logits_l = logits_l[:, 0]
        pred = logits_l 

        return pred[0]


    def residual_net(self, params, x, t):        
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
        
        s1 = outputs

        # Compute forward pass
        s1_pred =vmap(self.operator_net, (None, 0, 0))(params, x, t)
        # Compute loss
        loss = np.mean((s1.flatten() - s1_pred.flatten())**2)

        return loss

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
          u1, u2= inputs

          # Compute forward pass
    
          s_x_bc1_pred = vmap(self.ux_net, (None, 0))(params, u1)
          s_x_bc2_pred = vmap(self.ux_net, (None, 0))(params, u2)
    
          # Compute loss
          loss_s_x_bc = np.mean((s_x_bc1_pred - s_x_bc2_pred)**2)
    
          return loss_s_x_bc  
    
    # Define residual loss
    def loss_res(self, params, batch):
        # Fetch data
        inputs = batch
        u = inputs
        x = u[:, 1]
        t = u[:, 0]
        
        # Compute forward pass
        res1_pred  = vmap(self.residual_net, (None, 0, 0))(params, x, t)
        loss_res = np.mean((res1_pred)**2)
        return loss_res   

    
    # Define total loss
    @partial(jit, static_argnums=(0,))
    def loss(self, params, params_t, ics_batch, bc1_batch, bc2_batch, res_batch, F, lam):
        loss_ics = self.loss_ics(params, ics_batch)
        loss_bc1 = self.loss_bcs(params, bc1_batch)
        loss_bcs_x = self.loss_bcs_x(params, bc2_batch)
        loss_res = self.loss_res(params, res_batch)


        loss =  self.ics_weight*(loss_ics)\
                + self.res_weight*loss_res \
                + self.ut_weight*(loss_bc1+loss_bcs_x)
                
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
        loss += s
        params_nl, params_l = params

        loss += 1.0e-5*self.weight_nl(params_nl)
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
        if 0:
            pbar = trange(5000)
            # Main training loop
            for it in pbar:
                # Fetch data
                res_batch= next(res_data)
                ics_batch= next(ics_data)
                bc1_batch= next(bc1_data)
                bc2_batch= next(bc2_data)
                    
                if it % 1000 == 0:
                    params = self.get_params(self.opt_state)
    
                    # Compute losses
                    loss_value = self.loss_ICS_full(params, self.params_t, ics_batch, bc1_batch, bc2_batch, res_batch, F, lam)
                    
                    res_value = self.loss_res(params, res_batch)
                    ics_value = self.loss_ics(params, ics_batch)
                    bcs_value = self.loss_bcs(params, bc1_batch)+self.loss_bcs_x(params, bc2_batch)

                    # Store losses
                    self.loss_training_log.append(loss_value)
                    self.loss_res_log.append(res_value)
                    self.loss_ics_log.append(ics_value)
                    self.loss_ut_log.append(bcs_value)
    
                    # Print losses
                    pbar.set_postfix({'Loss': loss_value, 
                                      'Res': res_value, 
                                      'ICS':ics_value,
                                      'BCS': bcs_value})
                

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
                pbar.set_postfix({'Loss': loss_value, 
                                  'Res': res_value, 
                                  'ICS':ics_value,
                                  'BCS': bcs_value})

                

    # Evaluates predictions at test points  
    def predict_u(self, params, U_star):
        x = U_star[:, 1]
        t = U_star[:, 0]
        
      #  s_pred = vmap(self.apply_low, (None, 0, 0))(params_low, U_star, Y_star)
        s_pred =vmap(self.operator_net, (None, 0, 0))(params, x, t)
        return s_pred


    def predict_u_nl(self, params, U_star):
        x = U_star[:, 1]
        t = U_star[:, 0]
        
      #  s_pred = vmap(self.apply_low, (None, 0, 0))(params_low, U_star, Y_star)
        s_pred =vmap(self.operator_net_nl, (None, 0, 0))(params, x, t)
        return s_pred


    def predict_u_l(self, params, U_star):
        x = U_star[:, 1]
        t = U_star[:, 0]
        
      #  s_pred = vmap(self.apply_low, (None, 0, 0))(params_low, U_star, Y_star)
        s_pred =vmap(self.operator_net_l, (None, 0, 0))(params, x, t)
        return s_pred

    def predict_res(self, params, U_star):

        x = U_star[:, 1]
        t = U_star[:, 0]
        
        s_pred =vmap(self.residual_net, (None, 0, 0))(params,  x, t)
        return s_pred
    
    def predict_ut(self, params, U_star):
        s_pred =vmap(self.ut_net, (None, 0))(params, U_star)
        return s_pred
    
    
    @partial(jit, static_argnums=(0,))
    def predict_l2_nl(self, params, U_star):
        pred = self.predict_u_nl(params, U_star)
        return pred[0]**2
    
    @partial(jit, static_argnums=(0,))
    def predict_l2_l(self, params, U_star):
        pred = self.predict_u_l(params, U_star)
        return pred[0]**2
    
    def compute_MAS(self, params, coords, key,  num_samples=200, plot_diffs=True, disp_freq=1, scale=False):
    
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
    
            u = coords[0:1,:] + (coords[1:2,:]- coords[0:1,:])*random.uniform(key, shape=[1, 2])

            ders_nl = grad(self.predict_l2_nl)(params, u)
            ders_l = grad(self.predict_l2_l)(params, u)
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
        flat_MAS, _  = ravel_pytree(F_accum)

        if scale:
            q = np.quantile(flat_MAS, q=.999)
            for v in range(len(F_accum)):
                    F_accum[v] /= q
        
        return F_accum                    
