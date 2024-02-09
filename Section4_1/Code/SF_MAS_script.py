"""
Created 2023
@author: Amanda Howard (amanda.howard@pnnl.gov)
"""

import os
import scipy.io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from DNN_MAS_class import DNN_MAS

import jax
import jax.numpy as np
from jax.experimental import optimizers
from jax.flatten_util import ravel_pytree
from jax import random, grad, vmap, jit, hessian
#import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
from torch.utils import data
from functools import partial


# Data generator
class DataGenerator(data.Dataset):
    def __init__(self, u,  s, 
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
        s = np.array(self.s)[idx]
        u = self.u[idx,:]
        # Construct batch
        inputs = u
        outputs = s
        return inputs, outputs
    
    
def save_data(model, params, u, s, save_results_to, save_prfx):
    S_full =  model.predict_full(params, u)
    fname= save_results_to +"output_" + save_prfx + ".mat"
    scipy.io.savemat(fname, { 'u': u,
                              'S_test_h':s, 
                              'S_full':S_full})
       

def get_data(ID, u, s, batch_size):
    if ID==0:
        beta_range = np.arange(0, 190, 2)
        beta_range_test = np.arange(1, 190, 2)
    if ID==1:
        beta_range = np.arange(0, 190, 2)
        beta_range_test = np.arange(1, 190, 2)
    if ID==2:
        beta_range = np.arange(190, 390, 2)
        beta_range_test = np.arange(191, 390, 2)
    if ID==3:
        beta_range = np.arange(390, 590, 2)
        beta_range_test = np.arange(391, 590, 2)
    if ID==4:
        beta_range = np.arange(590, 790, 2)
        beta_range_test = np.arange(591, 790, 2)
    if ID==5:
        beta_range = np.arange(790, 891, 2)
        beta_range_test = np.arange(791, 891, 2)

    
    u_A = u[beta_range, :, :]
    s_A = s[beta_range, :, :]
    ut_A = u[beta_range_test, :, :]
    st_A = s[beta_range_test, :, :]
    train_dataset = DataGenerator(u_A, s_A, batch_size)
    test_dataset = DataGenerator(ut_A, st_A, ut_A.shape[0])
    return train_dataset, test_dataset, u_A

# =============================================
# =============================================

def run_SF_MAS(layer_sizes_A, scaled, save_str_pre):
    batch_size = 10
    epochs = 100000

    MAS_num_samples = 400
    reloadA = False
    reloadC = False
    reloadD = False
    reloadE = False
    reloadF = False
    res = 1

    lvec = [.00001, .0001, .001, .01, .1, 1, 10, 100, 1000, 10000, 0]
    lvec = [.00001, 0]
    for i in np.arange(len(lvec)):
        l = lvec[i]
        save_str = save_str_pre + "_l" + str(lvec[i])
        
        df = pd.read_excel("../BatteryTest_Current_ChargeCurve.xlsx")
        s = df.iloc[:, 0].values
        s = s.reshape([s.shape[0], -1, 1])
        u = df.iloc[:, 1:].values
        u = u[:, 0:784]
        u = u.reshape([u.shape[0], 1, -1])
        u = jax.device_put(u)
        s = jax.device_put(s)
    
        # Create data sets
        train_dataset_A, test_dataset_A, u_A = get_data(0, u, s, batch_size)
        train_dataset_C, test_dataset_C, u_C = get_data(2, u, s, batch_size)
        train_dataset_D, test_dataset_D, u_D = get_data(3, u, s, batch_size)
        train_dataset_E, test_dataset_E, u_E = get_data(4, u, s, batch_size)
        train_dataset_F, test_dataset_F, u_F = get_data(5, u, s, batch_size)

        # ====================================
        # saving settings
        # ====================================
        results_dir_A = "../resultsA/"+save_str_pre+"/"
        if not os.path.exists(results_dir_A):
            os.makedirs(results_dir_A)
    
        results_dir_C = "../resultsC/"+save_str+"/"
        if not os.path.exists(results_dir_C):
            os.makedirs(results_dir_C)
            
        results_dir_D = "../resultsD/"+save_str+"/"
        if not os.path.exists(results_dir_D):
            os.makedirs(results_dir_D)
            
        results_dir_E = "../resultsE/"+save_str+"/"
        if not os.path.exists(results_dir_E):
            os.makedirs(results_dir_E)
            
        results_dir_F = "../resultsF/"+save_str+"/"
        if not os.path.exists(results_dir_F):
            os.makedirs(results_dir_F)
    
        
        # ====================================
        # DNN model A
        # ====================================
        lr = optimizers.exponential_decay(1e-3, decay_steps=2000, decay_rate=0.9)
        model_A = DNN_MAS(layer_sizes_A, lr, restart=0)
    
        if reloadA:
            params_A = model_A.unravel_params(np.load(results_dir_A + '/params_A.npy'))
            t = model_A.compute_MAS(params_A, u_A,  random.PRNGKey(12345), 
                                      num_samples=0, plot_diffs=False, disp_freq=40, scaled=True)
            _, unravel  = ravel_pytree(t)
            d_vx = scipy.io.loadmat(results_dir_A + "/MAS.mat")
            MAS_A  = unravel(d_vx["MAS"][0, :])
        else:     
            model_A.train(train_dataset_A, test_dataset_A, 0, nIter=epochs, lam=[])
            print('\n ... A Training done ...')
            scipy.io.savemat(results_dir_A +"losses.mat", 
                             {'training_error':model_A.error_training_log, 
                              'testing_error':model_A.error_test_log})
        
            params_A = model_A.get_params(model_A.opt_state)
            flat_params, _  = ravel_pytree(model_A.get_params(model_A.opt_state))
            np.save(results_dir_A + 'params_A.npy', flat_params)
        
            save_data(model_A, params_A, u, s, results_dir_A, "dataA")
    
    
            MAS_A =model_A.compute_MAS(params_A, u_A, random.PRNGKey(12345), 
                                      num_samples=MAS_num_samples, plot_diffs=True, disp_freq=100, scaled=True)
            flat_MAS, _  = ravel_pytree(MAS_A)
            scipy.io.savemat(results_dir_A +"/MAS.mat",  {'MAS':flat_MAS})
            reloadA = True

            

    
    
        # ====================================
        # DNN model C
        # ====================================     
        lr = optimizers.exponential_decay(1e-3, decay_steps=2000, decay_rate=0.95)
        model_C = DNN_MAS(layer_sizes_A, lr, restart = res, params_t=params_A, params_i=params_A)
    
        if reloadC:
            params_C = model_C.unravel_params(np.load(results_dir_C + '/params_C.npy'))
            t = model_C.compute_MAS(params_C, u_C,  random.PRNGKey(12345), 
                                      num_samples=0, plot_diffs=False, disp_freq=40, scaled=True)
            _, unravel  = ravel_pytree(t)
            d_vx = scipy.io.loadmat(results_dir_C + "/MAS.mat")
            MAS_C  = unravel(d_vx["MAS"][0, :])
        else:                    
            model_C.train(train_dataset_C, test_dataset_C, MAS_A, nIter=epochs, lam=[l])
            
            print('\n ... C Training done ...')
        
            scipy.io.savemat(results_dir_C +"losses.mat", 
                             {'training_error':model_C.error_training_log, 
                              'testing_error':model_C.error_test_log})
        
            params_C = model_C.get_params(model_C.opt_state)
            flat_params, _  = ravel_pytree(params_C)
            np.save(results_dir_C + 'params_C.npy', flat_params)
            
            save_data(model_C, params_C, u, s, results_dir_C, "dataC")
    
            MAS_C =model_C.compute_MAS(params_C, u_C,  random.PRNGKey(12345), 
                                      num_samples=MAS_num_samples, plot_diffs=True, disp_freq=100, scaled=True)
            flat_MAS, _  = ravel_pytree(MAS_C)
            scipy.io.savemat(results_dir_C +"/MAS.mat",  {'MAS':flat_MAS})
    
    
            
        
        # ====================================
        # DNN model D
        # ====================================     
        MAS = MAS_C + MAS_A
        params_in =  params_C + params_A
        
        model_D = DNN_MAS(layer_sizes_A, lr, restart = res, params_t=params_in, params_i=params_C)
    
        if reloadD:
            params_D = model_D.unravel_params(np.load(results_dir_D + '/params_D.npy'))
            t = model_D.compute_MAS(params_D, u_D,  random.PRNGKey(12345), 
                                      num_samples=0, plot_diffs=False, disp_freq=40, scaled=True)
            _, unravel  = ravel_pytree(t)
            d_vx = scipy.io.loadmat(results_dir_D + "/MAS.mat")
            MAS_D  = unravel(d_vx["MAS"][0, :])
        else:                     
            model_D.train(train_dataset_D, test_dataset_D, MAS, nIter=epochs, lam=[l, l])
            
            print('\n ... D Training done ...')
            
            scipy.io.savemat(results_dir_D +"losses.mat", 
                             {'training_error':model_D.error_training_log, 
                              'testing_error':model_D.error_test_log})
            
            params_D = model_D.get_params(model_D.opt_state)
            flat_params, _  = ravel_pytree(model_D.get_params(model_D.opt_state))
            np.save(results_dir_D + 'params_D.npy', flat_params)
    
            save_data(model_D, params_D, u, s, results_dir_D, "dataD")
    
            MAS_D =model_D.compute_MAS(params_D, u_D,  random.PRNGKey(145), 
                                      num_samples=MAS_num_samples, plot_diffs=True, disp_freq=100, scaled=True)
            
            
            flat_MAS, _  = ravel_pytree(MAS_D)
            scipy.io.savemat(results_dir_D +"/MAS.mat",  {'MAS':flat_MAS})
    
    
        # ====================================
        # DNN model E
        # ====================================     
        MAS = MAS_D + MAS_C + MAS_A
        params_in =  params_D + params_C + params_A
        
        model_E = DNN_MAS(layer_sizes_A, lr, restart = res, params_t=params_in, params_i=params_D)
    
        if reloadE:
            params_E = model_E.unravel_params(np.load(results_dir_E + '/params_E.npy'))
            t = model_E.compute_MAS(params_E, u_E,  random.PRNGKey(12345), 
                                      num_samples=0, plot_diffs=False, disp_freq=40, scaled=True)
            _, unravel  = ravel_pytree(t)
            d_vx = scipy.io.loadmat(results_dir_E + "/MAS.mat")
            MAS_E  = unravel(d_vx["MAS"][0, :])
        else:                     
            model_E.train(train_dataset_E, test_dataset_E, MAS,  nIter=epochs, lam=[l, l, l])
            
            print('\n ... E Training done ...')
            
            scipy.io.savemat(results_dir_E +"losses.mat", 
                             {'training_error':model_E.error_training_log, 
                              'testing_error':model_E.error_test_log})
            
            params_E = model_E.get_params(model_E.opt_state)
            flat_params, _  = ravel_pytree(params_E)
            np.save(results_dir_E + 'params_E.npy', flat_params)
    
            save_data(model_E, params_E, u, s, results_dir_E, "dataE")
    
            MAS_E =model_E.compute_MAS(params_E, u_E,  random.PRNGKey(145), 
                                      num_samples=MAS_num_samples, plot_diffs=True, disp_freq=100, scaled=True)
            
            
            flat_MAS, _  = ravel_pytree(MAS_E)
            scipy.io.savemat(results_dir_E +"/MAS.mat",  {'MAS':flat_MAS})
    
    
        # ====================================
        # DNN model F
        # ====================================     
        MAS = MAS_E + MAS_D + MAS_C + MAS_A
        params_in =  params_E + params_D + params_C + params_A
        
        model_F = DNN_MAS(layer_sizes_A, lr, restart = res, params_t=params_in, params_i=params_E)
    
        if reloadF:
            params_F = model_F.unravel_params(np.load(results_dir_E + '/params_E.npy'))
            t = model_F.compute_MAS(params_F, u_F,  random.PRNGKey(12345), 
                                      num_samples=0, plot_diffs=False, disp_freq=40, scaled=True)
            _, unravel  = ravel_pytree(t)
            d_vx = scipy.io.loadmat(results_dir_F + "/MAS.mat")
            MAS_F  = unravel(d_vx["MAS"][0, :])
        else:                     
            model_F.train(train_dataset_F, test_dataset_F, MAS,  nIter=epochs, lam=[l, l, l, l])
            
            print('\n ... F Training done ...')
            
            scipy.io.savemat(results_dir_F +"losses.mat", 
                             {'training_error':model_F.error_training_log, 
                              'testing_error':model_F.error_test_log})
            
            params_F = model_F.get_params(model_F.opt_state)
            flat_params, _  = ravel_pytree(params_F)
            np.save(results_dir_F + 'params_F.npy', flat_params)
    
            save_data(model_F, params_F, u, s, results_dir_F, "dataF")
    
            MAS_F =model_F.compute_MAS(params_F, u_F,  random.PRNGKey(145), 
                                      num_samples=MAS_num_samples, plot_diffs=True, disp_freq=100, scaled=True)
            
            flat_MAS, _  = ravel_pytree(MAS_F)
            scipy.io.savemat(results_dir_F +"/MAS.mat",  {'MAS':flat_MAS})
    
    
if __name__ == "__main__":    
    
        dim_x_branch_low = 784
        N_low = 80
        layer_sizes_A = [dim_x_branch_low, N_low, N_low, 1]
        scaled = False
        save_str_pre = "SF_wide"
        run_SF_MAS(layer_sizes_A, scaled, save_str_pre)
        

        N_low = 40
        layer_sizes_A = [dim_x_branch_low, N_low, N_low, N_low, N_low, 1]
        
        scaled = False
        save_str_pre = "SF_narrow"
        run_SF_MAS(layer_sizes_A, scaled, save_str_pre)

        
        