#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created October 2023
@author: Amanda Howard (amanda.howard@pnnl.gov)

Usage: training the single fidelity results
"""

import os
import scipy.io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import jax
import jax.numpy as np
from utils import DataGenerator, DataGenerator_RDPS, DataGenerator_res
from jax import random, grad, vmap, jit
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
    scipy.io.savemat(fname, {'U_res':u_res,'S_pred':S_pred}, format='4')
    
    scipy.io.savemat(save_results_to +"losses.mat", 
                     {'training_loss':model.loss_training_log,
                      'res_loss':model.loss_res_log,
                      'ics_loss':model.loss_ics_log,
                      'data_loss':model.loss_data_log}, format='4')

    
# =============================================
# =============================================
def run_SF(replay, MAS, RDPS, scaled, MASl=[0], N_low=100):
    
    
    save_suff = "SF_" + str(N_low)
    Npts = 1
    if replay:
        save_suff = save_suff + "_replay"
    if MAS:
        save_suff = save_suff + "_MAS"
    if scaled:
        save_suff = save_suff + "scaled"
    if RDPS:
        save_suff = save_suff + "_RDPS"
        Npts = 10000
         
    ics_weight = 1.0
    res_weight = 10.0
    data_weight  = 0.0

    batch_size = 100
    batch_size_res = 0
    if RDPS:
        batch_size_res = int(batch_size/2)
        
        
    cases = ['A',  'C', 'D', 'E', 'F']
    MAS_num_samples = 10
    if MAS:
        MAS_num_samples = 1000

    k = 2
    c = 0 
    epochs = 50000
    lr = optimizers.exponential_decay(1e-3, decay_steps=2000, decay_rate=0.95)
    
    layers_branch_low = [1, N_low, N_low, N_low, N_low, N_low, 2]
    
    reloadA = False
    reloadC = False
    reloadD = False
    reloadE = False
    reloadF = False

    min_A = 0
    min_B = 2
    min_C = 4
    min_D = 6
    min_E = 8
    min_F = 10

    
    data_range_A = np.arange(0,int(2*min_B))
    data_range_C = np.arange(0, int(2*min_C))
    data_range_D = np.arange(0, int(2*min_D))
    data_range_E = np.arange(0, int(2*min_E))
    data_range_F = np.arange(0, int(2*min_F))

    
    u_bc = np.asarray([0]).reshape([1, -1])
    s_bc = np.asarray([1, 1]).reshape([1, -1])
    u_bc = jax.device_put(u_bc)
    s_bc = jax.device_put(s_bc)
    d_vx = scipy.io.loadmat("../data.mat")
    t_data_full, s_data_full = (d_vx["u"].astype(np.float32), 
               d_vx["s"].astype(np.float32))
    ic_dataset = DataGenerator(u_bc, s_bc, 1)

    for l in MASl: 
        save_str = str(l)
    
        # ====================================
        # saving settings
        # ====================================
        for i in range(len(cases)):
            results_dir = "../results_"+cases[i] + "/" + save_suff + "_l_" +save_str+"/"
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
        
        
    
    
        lam= []
        F = []
        results_dir = "../results_A/" + save_suff  + "_l_" +save_str+"/"
    
        model_A = DNN_class_MAS(layers_branch_low, ics_weight, res_weight, data_weight, [], lr)
        if reloadA:
            params_A = model_A.unravel_params(np.load(results_dir + '/params_A.npy'))
                
            t = model_A.compute_MAS(params_A, coords, random.PRNGKey(12345), 
                                  num_samples=0, plot_diffs=False, disp_freq=10, scale=scaled)
            _, unravel  = ravel_pytree(t)
            d_vx = scipy.io.loadmat(results_dir +"/MAS.mat")
            MAS_A  = unravel(d_vx["MAS"][0, :])
        
        else:
            model_A.train(ic_dataset, res_dataset, data_dataset, lam, F, [], nIter=epochs)
            params_A = model_A.get_params(model_A.opt_state)
            save_data(model_A, params_A,  results_dir, 'A')   
    
            MAS_A =model_A.compute_MAS(params_A, coords, random.PRNGKey(12345), 
                                          num_samples=MAS_num_samples, plot_diffs=True, disp_freq=100, scale=scaled)
            flat_MAS, _  = ravel_pytree(MAS_A)
    
            scipy.io.savemat(results_dir+"/MAS.mat",  {'MAS':flat_MAS}, format='4')
    
        t = model_A.compute_MAS(params_A, coords, random.PRNGKey(12345), 
                              num_samples=0, plot_diffs=False, disp_freq=100, scale=scaled)
        _, unravel  = ravel_pytree(t)
            
        # ====================================
        # Train C
        # ====================================
        
        coords = [min_B, min_C]
        if replay:
            coords = [0, min_C]
            
        u_bc = np.asarray(coords[0]).reshape([1, -1])
        s_bc = model_A.predict_low(params_A, u_bc)
        ic_dataset = DataGenerator(u_bc, s_bc, 1)
        res_pts = coords[0] + (coords[1]-coords[0])*random.uniform(random.PRNGKey(1234), shape=[Npts, 1])
        res_val = model_A.predict_res(params_A, res_pts)
        err = res_val**k/np.mean( res_val**k) + c
        err_norm = err/np.sum(err)
        res_dataset = DataGenerator_RDPS(coords, res_pts, err_norm, batch_size_res, batch_size)
        
        t_data = jax.device_put(t_data_full[:, data_range_C].reshape([-1, 1]))
        s_data = jax.device_put(s_data_full[data_range_C, :].reshape([-1, 2]))
        data_dataset = DataGenerator(t_data, s_data, len(t_data))
        
        
    
        lam= [l]
        results_dir = "../results_C/" + save_suff  + "_l_" +save_str+"/"
    
        model_C = DNN_class_MAS(layers_branch_low, ics_weight, res_weight, data_weight,  params_A, lr)
        if reloadC:
            params_C = model_C.unravel_params(np.load(results_dir + '/params_C.npy'))
            d_vx = scipy.io.loadmat(results_dir +"/MAS.mat")
            MAS_C  = unravel(d_vx["MAS"][0, :])
        else:
            model_C.train(ic_dataset, res_dataset, data_dataset, lam, MAS_A, params_A, nIter=epochs)
            params_C = model_C.get_params(model_C.opt_state)
            save_data(model_C, params_C,  results_dir, 'C')  
    
            MAS_C =model_C.compute_MAS(params_C, coords, random.PRNGKey(12345), 
                                          num_samples=MAS_num_samples, plot_diffs=True, disp_freq=100, scale=scaled)
            flat_MAS, _  = ravel_pytree(MAS_C)
    
            scipy.io.savemat(results_dir+"/MAS.mat",  {'MAS':flat_MAS}, format='4')
    
    
        # ====================================
        # Train D
        # ====================================
        
        coords = [min_C, min_D]
        if replay:
            coords = [0, min_D]
        
        u_bc = np.asarray(coords[0]).reshape([1, -1])
        s_bc = model_C.predict_low(params_C, u_bc)
        ic_dataset = DataGenerator(u_bc, s_bc, 1)
        res_pts = coords[0] + (coords[1]-coords[0])*random.uniform(random.PRNGKey(1234), shape=[Npts, 1])
        res_val = model_C.predict_res(params_C, res_pts)
        err = res_val**k/np.mean( res_val**k) + c
        err_norm = err/np.sum(err)
        res_dataset = DataGenerator_RDPS(coords, res_pts, err_norm, batch_size_res, batch_size)
        
        t_data = jax.device_put(t_data_full[:, data_range_D].reshape([-1, 1]))
        s_data = jax.device_put(s_data_full[data_range_D, :].reshape([-1, 2]))
        data_dataset = DataGenerator(t_data, s_data, len(t_data))
        
    
        lam= [l, l]
        MAS= MAS_A + MAS_C
        params_prev = params_A + params_C
        results_dir = "../results_D/" + save_suff  + "_l_" +save_str+"/"
    
        model_D = DNN_class_MAS(layers_branch_low, ics_weight, res_weight, data_weight, params_C, lr)
        if reloadD:
            params_D = model_D.unravel_params(np.load(results_dir + '/params_D.npy'))
            d_vx = scipy.io.loadmat(results_dir +"/MAS.mat")
            MAS_D  = unravel(d_vx["MAS"][0, :])
        else:
            model_D.train(ic_dataset, res_dataset, data_dataset, lam, MAS, params_prev, nIter=epochs)
            params_D = model_D.get_params(model_D.opt_state)
            save_data(model_D, params_D,  results_dir, 'D')  
    
            MAS_D =model_D.compute_MAS(params_D, coords, random.PRNGKey(12345), 
                                          num_samples=MAS_num_samples, plot_diffs=True, disp_freq=100, scale=scaled)
            flat_MAS, _  = ravel_pytree(MAS_D)
    
            scipy.io.savemat(results_dir+"/MAS.mat",  {'MAS':flat_MAS}, format='4')
    
    
    
        # ====================================
        # Train E
        # ====================================
    
        coords = [min_D, min_E]
        if replay:
            coords = [0, min_E]
            
        u_bc = np.asarray(coords[0]).reshape([1, -1])
        s_bc = model_D.predict_low(params_D, u_bc)
        ic_dataset = DataGenerator(u_bc, s_bc, 1)
    
        res_pts = coords[0] + (coords[1]-coords[0])*random.uniform(random.PRNGKey(1234), shape=[Npts, 1])
        res_val = model_D.predict_res(params_D, res_pts)
        err = res_val**k/np.mean( res_val**k) + c
        err_norm = err/np.sum(err)
        res_dataset = DataGenerator_RDPS(coords, res_pts, err_norm, batch_size_res, batch_size)
        
        t_data = jax.device_put(t_data_full[:, data_range_E].reshape([-1, 1]))
        s_data = jax.device_put(s_data_full[data_range_E, :].reshape([-1, 2]))
        data_dataset = DataGenerator(t_data, s_data, len(t_data))
        
        lam= [l, l, l]
        MAS= MAS_A + MAS_C + MAS_D
        params_prev = params_A + params_C + params_D
        results_dir = "../results_E/" + save_suff  + "_l_" +save_str+"/"
    
        model_E = DNN_class_MAS(layers_branch_low, ics_weight, res_weight, data_weight, params_D, lr)
        if reloadE:
            params_E = model_E.unravel_params(np.load(results_dir+ '/params_E.npy'))
            d_vx = scipy.io.loadmat(results_dir +"/MAS.mat")
            MAS_E  = unravel(d_vx["MAS"][0, :])
        else:
            model_E.train(ic_dataset, res_dataset, data_dataset, lam, MAS, params_prev, nIter=epochs)
            params_E = model_E.get_params(model_E.opt_state)
            save_data(model_E, params_E,  results_dir, 'E')  
    
            MAS_E =model_E.compute_MAS(params_E, coords, random.PRNGKey(12345), 
                                          num_samples=MAS_num_samples, plot_diffs=True, disp_freq=100, scale=scaled)
            flat_MAS, _  = ravel_pytree(MAS_E)
    
            scipy.io.savemat(results_dir+"/MAS.mat",  {'MAS':flat_MAS}, format='4')
    
    
        # ====================================
        # Train F
        # ====================================
        
        coords = [min_E, min_F]
        if replay:
            coords = [0, min_F]
            
        u_bc = np.asarray(coords[0]).reshape([1, -1])
        s_bc = model_E.predict_low(params_E, u_bc)
        ic_dataset = DataGenerator(u_bc, s_bc, 1)
        
        res_pts = coords[0] + (coords[1]-coords[0])*random.uniform(random.PRNGKey(1234), shape=[Npts, 1])
        res_val = model_E.predict_res(params_E, res_pts)
        err = res_val**k/np.mean( res_val**k) + c
        err_norm = err/np.sum(err)
        res_dataset = DataGenerator_RDPS(coords, res_pts, err_norm, batch_size_res, batch_size)
        
        t_data = jax.device_put(t_data_full[:, data_range_F].reshape([-1, 1]))
        s_data = jax.device_put(s_data_full[data_range_F, :].reshape([-1, 2]))
        data_dataset = DataGenerator(t_data, s_data, len(t_data))
        
    
        lam= [l, l, l, l]
        MAS= MAS_A + MAS_C + MAS_D + MAS_E
        params_prev = params_A + params_C + params_D + params_E
        results_dir = "../results_F/" + save_suff  + "_l_" +save_str+"/"
    
        model_F = DNN_class_MAS(layers_branch_low, ics_weight, res_weight, data_weight, params_E, lr)
        if reloadF:
            params_F = model_F.unravel_params(np.load(results_dir+'/params_F.npy'))
            d_vx = scipy.io.loadmat(results_dir+"/MAS.mat")
            MAS_F  = unravel(d_vx["MAS"][0, :])
        else:
            model_F.train(ic_dataset, res_dataset, data_dataset, lam, MAS, params_prev, nIter=epochs)
            params_F = model_F.get_params(model_F.opt_state)
            save_data(model_F, params_F, results_dir, 'F')  
    
            MAS_F =model_F.compute_MAS(params_F, coords, random.PRNGKey(12345), 
                                          num_samples=MAS_num_samples, plot_diffs=True, disp_freq=100, scale=scaled)
            flat_MAS, _  = ravel_pytree(MAS_F)
    
            scipy.io.savemat(results_dir+"/MAS.mat",  {'MAS':flat_MAS}, format='4')
            
        reloadA = True


# =============================================
# =============================================
if __name__ == "__main__":
    
    lvec =[0.001, 0.01, 0.1, 1, 10, 100]

    replay = False
    MAS = False
    RDPS = False 
    scaled = False
    run_SF(replay, MAS, RDPS, scaled, N_low=100) #SF
        
    replay = True
    MAS = False
    RDPS = False 
    scaled = False
    run_SF(replay, MAS, RDPS, scaled,MASl=0, N_low=25) #SF-replay
    run_SF(replay, MAS, RDPS, scaled,MASl=0, N_low=50) #SF-replay
    run_SF(replay, MAS, RDPS, scaled,MASl=0, N_low=100) #SF-replay
    run_SF(replay, MAS, RDPS, scaled,MASl=0, N_low=150) #SF-replay
    run_SF(replay, MAS, RDPS, scaled,MASl=0, N_low=200) #SF-replay
        
    replay = False
    MAS = True
    RDPS =False
    scaled = False
    run_SF(replay, MAS, RDPS, scaled, MASl=lvec, N_low=100) #SF-MAS


    


    
    
    
    
    
    
    
    
    
