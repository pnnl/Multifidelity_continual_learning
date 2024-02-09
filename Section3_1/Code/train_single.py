#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created October 2023
@author: Amanda Howard (amanda.howard@pnnl.gov)

Usage: trains one single fidelity PINN in the full domain [0, 10]
"""

import os

#import numpy as np
import scipy.io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import jax
import jax.numpy as np
from utils import DataGenerator, DataGenerator_res
from jax.example_libraries import optimizers
from jax.flatten_util import ravel_pytree


from DNN_MAS_Class import DNN_class_MAS

def save_data(model, params,  save_results_to, save_prfx):
    # ====================================
    # Saving model
    # ====================================
    t_train_range = np.linspace(0, 50, 2000)
    u_res = t_train_range.reshape([len(t_train_range), 1])
    flat_params, _  = ravel_pytree(model.get_params(model.opt_state))
    np.save(save_results_to + 'params_' + save_prfx + '.npy', flat_params)

    S_pred =  model.predict_low(params, u_res)

    fname= save_results_to +"beta_test.mat"
    scipy.io.savemat(fname, {'U_res':u_res,'S_pred':S_pred})
    
    scipy.io.savemat(save_results_to +"losses.mat", 
                     {'training_loss':model.loss_training_log,
                      'res_loss':model.loss_res_log,
                      'ics_loss':model.loss_ics_log,
                      'data_loss':model.loss_data_log})

    
# =============================================
# =============================================
def run_SF():
    
    ics_weight = 1.0
    res_weight = 10.0
    data_weight  = 0.0

    batch_size = 200

    epochs = 50000
    lr = optimizers.exponential_decay(1e-3, decay_steps=2000, decay_rate=0.95)
    N_low = 200
    layers_branch_low = [1, N_low,N_low, N_low, 2]
    
    min_A = 0
    min_B = 10

    data_range_A = np.arange(0,int(2*min_B))

    u_bc = np.asarray([0]).reshape([1, -1])
    s_bc = np.asarray([1, 1]).reshape([1, -1])
    u_bc = jax.device_put(u_bc)
    s_bc = jax.device_put(s_bc)
    d_vx = scipy.io.loadmat("../data.mat")
    t_data_full, s_data_full = (d_vx["u"].astype(np.float32), 
               d_vx["s"].astype(np.float32))
    ic_dataset = DataGenerator(u_bc, s_bc, 1)

    # ====================================
    # saving settings
    # ====================================
    results_dir = "../results_single/"
    save_results_to = results_dir
    if not os.path.exists(save_results_to):
        os.makedirs(save_results_to)
        
    # ====================================
    # Train A
    # ====================================
    u_bc = np.asarray([0]).reshape([1, -1])
    s_bc = np.asarray([1, 1]).reshape([1, -1])
    u_bc = jax.device_put(u_bc)
    s_bc = jax.device_put(s_bc)

    t_data = jax.device_put(t_data_full[:, data_range_A].reshape([-1, 1]))
    s_data = jax.device_put(s_data_full[data_range_A, :].reshape([-1, 2]))

    # Create data set
    coords = [min_A, min_B]
    ic_dataset = DataGenerator(u_bc, s_bc, 1)
    res_dataset = DataGenerator_res(coords, batch_size)
    data_dataset = DataGenerator(t_data, s_data, len(t_data))


    # Create and train model
    model_A = DNN_class_MAS(layers_branch_low, ics_weight, res_weight, data_weight, [], lr)

    model_A.train(ic_dataset, res_dataset, data_dataset, [], [], [], nIter=epochs)
    params_A = model_A.get_params(model_A.opt_state)
    save_data(model_A, params_A,  results_dir, 'A')   


# =============================================
# =============================================
if __name__ == "__main__":
    
    run_SF() #Train one single fidelity network
    


    


    
    
    
    
    
    
    
    
    
