"""
Created 2023
@author: Amanda Howard (amanda.howard@pnnl.gov)
"""

import os
import scipy.io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from DNN_MAS_class import DNN_MAS
from DNN_MF_MAS import MF_DNN_MAS

import numpy as onp
import jax
import jax.numpy as np
from jax.example_libraries import optimizers
from jax.flatten_util import ravel_pytree
from jax import random
from copy import deepcopy
import pandas as pd
from torch.utils import data
from functools import partial
from jax import random, grad, vmap, jit, hessian


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
    
class DataGenerator_h(data.Dataset):
    def __init__(self, u, u_lin, s, 
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u
        self.u_lin = u_lin
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
        u_lin = self.u_lin[idx,:]
        # C onstruct batch
        inputs = u, u_lin
        outputs = s
        return inputs, outputs    


def save_data(model, params, u, s, save_results_to):
    S_full =  model.predict_full(params, u)
    fname= save_results_to +"output.mat"
    scipy.io.savemat(fname, { 'u': u,
                              'S_test_h':s, 
                              'S_full':S_full})

def save_data_MF(model, params, u, u_lin, s, save_results_to):
    S_full =  model.predict_full(params, u, u_lin)
    fname= save_results_to +"output.mat"
    scipy.io.savemat(fname, { 'u': u,
                              'S_test_h':s, 
                              'S_full':S_full})
    
    
def gen_prev_data_B(model_A, u, params_A):
    predA = model_A.predict_full(params_A, u)
    u = np.concatenate((u,predA), axis=2)
    return u, predA



def gen_prev_data_C(model_A, model_B, u, params_A, params_B):
    predA = model_A.predict_full(params_A, u)
    uB = np.concatenate((u,predA), axis=2)    
    predB = model_B.predict_full(params_B, uB, predA)
    u = np.concatenate((u,predB), axis=2)
    return u, predB


def gen_prev_data_D(model_A, model_B, model_C, u, params_A, params_B, params_C):
    predA = model_A.predict_full(params_A, u)
    uB = np.concatenate((u,predA), axis=2)
    predB = model_B.predict_full(params_B, uB, predA)
    uC = np.concatenate((u, predB), axis=2)
    predC = model_C.predict_full(params_C, uC, predB)
    u = np.concatenate((u,predC), axis=2)
    return u, predC

def gen_prev_data_E(model_A, model_B, model_C, model_D,
                    u, params_A, params_B, params_C, params_D):

    predA = model_A.predict_full(params_A, u)
    uB = np.concatenate((u,predA), axis=2)
    predB = model_B.predict_full(params_B, uB, predA)
    uC = np.concatenate((u, predB), axis=2)
    predC = model_C.predict_full(params_C, uC, predB)
    uD = np.concatenate((u,predC), axis=2)
    predD = model_D.predict_full(params_D, uD, predC)
    u = np.concatenate((u,predD), axis=2)    
    return u, predD

def gen_prev_data_F(model_A, model_B, model_C, model_D, model_E,
                    u, params_A, params_B, params_C, params_D, params_E):

    predA = model_A.predict_full(params_A, u)
    uB = np.concatenate((u,predA), axis=2)
    predB = model_B.predict_full(params_B, uB, predA)
    uC = np.concatenate((u, predB), axis=2)
    predC = model_C.predict_full(params_C, uC, predB)
    uD = np.concatenate((u,predC), axis=2)
    predD = model_D.predict_full(params_D, uD, predC)
    uE = np.concatenate((u,predD), axis=2)    
    predE = model_E.predict_full(params_E, uE, predD)
    u = np.concatenate((u,predE), axis=2)    
    return u, predE



    
def get_data(ID, u, s):
    if ID==0:
        beta_range = np.arange(0, 190, 2)
        beta_range_test = np.arange(1, 190, 10)
    if ID==1:
        beta_range = np.arange(0, 190, 2)
        beta_range_test = np.arange(1, 190, 10)
    if ID==2:
        beta_range = np.arange(190, 390, 2)
        beta_range_test = np.arange(1, 390, 20)
    if ID==3:
        beta_range = np.arange(390, 590, 2)
        beta_range_test = np.arange(1, 590, 20)
    if ID==4:
        beta_range = np.arange(590, 790, 2)
        beta_range_test = np.arange(1, 790, 20)
    if ID==5:
        beta_range = np.arange(790, 891, 2)
        beta_range_test = np.arange(1, 891, 20)

    u_A = u[beta_range, :, :]
    s_A = s[beta_range, :, :]
    ut_A = u[beta_range_test, :, :]
    st_A = s[beta_range_test, :, :]
    return u_A, s_A, ut_A, st_A
           
# =============================================
# =============================================
def run_MF_MAS(layer_sizes_A, layer_sizes_nl, layer_sizes_l, scaled, save_str_pre):

    batch_size = 10
    epochs = 100000
    lvec = [.00001, .0001, .001, .01, .1, 1, 10, 100, 0]
    reloadA = False
    reloadB = False
    reloadC = False
    reloadD = False
    reloadE = False
    reloadF = False
          
    for i in np.arange(len(lvec)):
          l = lvec[i]

          save_str = save_str_pre + "_l" + str(lvec[i])

          lr = optimizers.exponential_decay(1e-3, decay_steps=2000, decay_rate=0.9)
          MAS_num_samples = 400


          
          df = pd.read_excel("../BatteryTest_Current_ChargeCurve.xlsx")
          s = df.iloc[:, 0].values
          s = s.reshape([s.shape[0], -1, 1])
          u = df.iloc[:, 1:].values
          u = u[:, 0:784]
          u = u.reshape([u.shape[0], 1, -1])
          u = jax.device_put(u)
          s = jax.device_put(s)
         
          # Create data sets
          u_A, s_A, ut_A, st_A = get_data(0, u, s)
          u_B, s_B, ut_B, st_B = get_data(1, u, s)
          u_C, s_C, ut_C, st_C = get_data(2, u, s)
          u_D, s_D, ut_D, st_D = get_data(3, u, s)
          u_E, s_E, ut_E, st_E = get_data(4, u, s)
          u_F, s_F, ut_F, st_F = get_data(5, u, s)
             
          # ====================================
          # saving settings
          # ====================================
          results_dir_A = "../resultsA/"+save_str_pre +"/"
          if not os.path.exists(results_dir_A):
              os.makedirs(results_dir_A)
        
          results_dir_B = "../resultsB/"+save_str_pre +"/"
          if not os.path.exists(results_dir_B):
              os.makedirs(results_dir_B)
        
        
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
          model_A = DNN_MAS(layer_sizes_A, lr, restart=0)
          train_dataset_A = DataGenerator(u_A, s_A, batch_size)
          test_dataset_A = DataGenerator(ut_A, st_A, ut_A.shape[0])
          
          if reloadA:
              params_A = model_A.unravel_params(np.load(results_dir_A + '/params.npy'))

          else:     
              model_A.train(train_dataset_A, test_dataset_A, 0, nIter=epochs, lam=[])
              print('\n ... A Training done ...')
              scipy.io.savemat(results_dir_A +"losses.mat", 
                             {'training_error':model_A.error_training_log, 
                              'testing_error':model_A.error_test_log})
          
              params_A = model_A.get_params(model_A.opt_state)
              flat_params, _  = ravel_pytree(model_A.get_params(model_A.opt_state))
              np.save(results_dir_A + 'params.npy', flat_params)
          
              save_data(model_A, params_A, u, s, results_dir_A)
        
        
          # ====================================
          # DNN model B
          # ====================================
        
          model_B = MF_DNN_MAS(layer_sizes_l,layer_sizes_nl, lr, restart = 0)
          uB, predA = gen_prev_data_B(model_A, u_A, params_A)
          uB_t, predA_t = gen_prev_data_B(model_A, ut_A, params_A)
          uB_full, predA_full = gen_prev_data_B(model_A, u, params_A)
        
          if reloadB:
              params_B = model_B.unravel_params(np.load(results_dir_B + '/params.npy'))
              t = model_B.compute_MAS(params_B, uB, predA,  random.PRNGKey(12345), 
                                        num_samples=0, plot_diffs=False, disp_freq=40, scale=scaled)
              _, unravel  = ravel_pytree(t)
              d_vx = scipy.io.loadmat(results_dir_B + "/MAS.mat")
              MAS_B  = unravel(d_vx["MAS"][0, :])
          else: 
              train_datasetB = DataGenerator_h(uB, predA, s_B, batch_size)
              test_datasetB = DataGenerator_h(uB_t, predA_t, st_B, uB_t.shape[0])
        
              model_B.train(train_datasetB, test_datasetB,  0, [],  nIter=epochs)
              print('\n ... B Training done ...')
              
              scipy.io.savemat(results_dir_B +"losses.mat", 
                               {'training_error':model_B.error_training_log,
                                'testing_error':model_B.error_test_log,
                                'loss_total_log':model_B.loss_total_log}, format='4')
              
              params_B = model_B.get_params(model_B.opt_state)
              flat_params, _  = ravel_pytree(model_B.get_params(model_B.opt_state))
              np.save(results_dir_B + 'params.npy', flat_params)
              
              save_data_MF(model_B, params_B, uB_full, predA_full, s, results_dir_B)
              
              MAS_B =model_B.compute_MAS(params_B, uB, predA,  random.PRNGKey(12345), 
                                    num_samples=MAS_num_samples, plot_diffs=True, disp_freq=40, scale=scaled)
              flat_MAS, _  = ravel_pytree(MAS_B)
              scipy.io.savemat(results_dir_B +"/MAS.mat",  {'MAS':flat_MAS}, format='4')
        
          t = model_B.compute_MAS(params_B, uB, predA,  random.PRNGKey(12345), 
                                    num_samples=0, plot_diffs=False, disp_freq=40, scale=scaled)
          _, unravel  = ravel_pytree(t)
          
          # ====================================
          # DNN model C
          # ====================================
          lr = optimizers.exponential_decay(1e-3, decay_steps=2000, decay_rate=0.95)
        
          model_C = MF_DNN_MAS(layer_sizes_l,layer_sizes_nl, lr, restart = 1, params_t=params_B)
          
          uC, predB =           gen_prev_data_C(model_A, model_B, u_C,  params_A, params_B)
          uC_t, predB_t =       gen_prev_data_C(model_A, model_B, ut_C, params_A, params_B)
          uC_full, predB_full = gen_prev_data_C(model_A, model_B, u,    params_A, params_B)
        
          res_dir = results_dir_C
          if reloadC:
              params_C = model_C.unravel_params(np.load(res_dir + '/params.npy'))
              MAS_C  = unravel(scipy.io.loadmat(res_dir + "/MAS.mat")["MAS"][0, :])
          else: 
              train_datasetC = DataGenerator_h(uC, predB, s_C, batch_size)
              test_datasetC = DataGenerator_h(uC_t, predB_t, st_C, uC_t.shape[0])
        
              model_C.train(train_datasetC, test_datasetC,  MAS_B, [l],  nIter=epochs)
              print('\n ... C Training done ...')
              
              scipy.io.savemat(res_dir +"losses.mat", 
                               {'training_error':model_C.error_training_log,
                                'testing_error':model_C.error_test_log,
                                'loss_total_log':model_C.loss_total_log}, format='4')
              
              params_C = model_C.get_params(model_C.opt_state)
              flat_params, _  = ravel_pytree(params_C)
              np.save(res_dir + 'params.npy', flat_params)
              
              save_data_MF(model_C, params_C, uC_full, predB_full, s, res_dir)
              
              MAS_C =model_C.compute_MAS(params_C, uC, predB,  random.PRNGKey(12345), 
                                    num_samples=MAS_num_samples, plot_diffs=True, disp_freq=40, scale=scaled)
              flat_MAS, _  = ravel_pytree(MAS_C)
              scipy.io.savemat(res_dir +"/MAS.mat",  {'MAS':flat_MAS}, format='4')
        
        
          # ====================================
          # DNN model D
          # ====================================
          MAS = MAS_C + MAS_B
          params_in =  params_C + params_B
          
          model_D = MF_DNN_MAS(layer_sizes_l,layer_sizes_nl, lr, restart = 1, params_t=params_in)
          
          uD, predC =           gen_prev_data_D(model_A, model_B, model_C, u_D,  
                                                params_A, params_B, params_C)
          uD_t, predC_t =       gen_prev_data_D(model_A, model_B, model_C, ut_D, 
                                                params_A, params_B, params_C)
          uD_full, predC_full = gen_prev_data_D(model_A, model_B, model_C, u,
                                                params_A, params_B, params_C)
          res_dir = results_dir_D
        
          if reloadD:
              params_D = model_D.unravel_params(np.load(res_dir + '/params.npy'))
              d_vx = scipy.io.loadmat(res_dir + "/MAS.mat")
              MAS_D  = unravel(d_vx["MAS"][0, :])
          else: 
              train_datasetD = DataGenerator_h(uD, predC, s_D, batch_size)
              test_datasetD = DataGenerator_h(uD_t, predC_t, st_D, uD_t.shape[0])
        
              model_D.train(train_datasetD, test_datasetD,  MAS, [l, l], nIter=epochs)
              print('\n ... D Training done ...')
              
              scipy.io.savemat(res_dir +"losses.mat", 
                               {'training_error':model_D.error_training_log,
                                'testing_error':model_D.error_test_log,
                                'loss_total_log':model_D.loss_total_log}, format='4')
              
              params_D = model_D.get_params(model_D.opt_state)
              flat_params, _  = ravel_pytree(params_D)
              np.save(res_dir + 'params.npy', flat_params)
              
              save_data_MF(model_D, params_D, uD_full, predC_full, s, res_dir)
              
              MAS_D =model_D.compute_MAS(params_D, uD, predC,  random.PRNGKey(12345), 
                                    num_samples=MAS_num_samples, plot_diffs=True, disp_freq=40, scale=scaled)
              flat_MAS, _  = ravel_pytree(MAS_D)
              scipy.io.savemat(res_dir +"/MAS.mat",  {'MAS':flat_MAS}, format='4')
        
        
          # ====================================
          # DNN model E
          # ====================================
          MAS = MAS_D + MAS_C + MAS_B
          params_in =  params_D + params_C + params_B
          
          model_E = MF_DNN_MAS(layer_sizes_l,layer_sizes_nl, lr, restart = 1, params_t=params_in)
          
          uE, predD =           gen_prev_data_E(model_A, model_B, model_C, model_D, u_E,  
                                                params_A, params_B, params_C, params_D)
          uE_t, predD_t =       gen_prev_data_E(model_A, model_B, model_C, model_D, ut_E, 
                                                params_A, params_B, params_C, params_D)
          uE_full, predD_full = gen_prev_data_E(model_A, model_B, model_C, model_D, u,
                                                params_A, params_B, params_C, params_D)
          res_dir = results_dir_E
        
          if reloadE:
              params_E = model_E.unravel_params(np.load(res_dir + '/params.npy'))
              d_vx = scipy.io.loadmat(res_dir + "/MAS.mat")
              MAS_E  = unravel(d_vx["MAS"][0, :])
          else: 
              train_datasetE = DataGenerator_h(uE, predD, s_E, batch_size)
              test_datasetE = DataGenerator_h(uE_t, predD_t, st_E, uE_t.shape[0])
        
              model_E.train(train_datasetE, test_datasetE,  MAS, [l, l, l], nIter=epochs)
              print('\n ... E Training done ...')
              
              scipy.io.savemat(res_dir +"losses.mat", 
                               {'training_error':model_E.error_training_log,
                                'testing_error':model_E.error_test_log,
                                'loss_total_log':model_E.loss_total_log}, format='4')
              
              params_E = model_E.get_params(model_E.opt_state)
              flat_params, _  = ravel_pytree(params_E)
              np.save(res_dir + 'params.npy', flat_params)
              
              save_data_MF(model_E, params_E, uE_full, predD_full, s, res_dir)
              
              MAS_E =model_E.compute_MAS(params_E, uE, predD,  random.PRNGKey(12345), 
                                    num_samples=MAS_num_samples, plot_diffs=True, disp_freq=40, scale=scaled)
              flat_MAS, _  = ravel_pytree(MAS_E)
              scipy.io.savemat(res_dir +"/MAS.mat",  {'MAS':flat_MAS}, format='4')
        
        
          # ====================================
          # DNN model F
          # ====================================
          MAS = MAS_E + MAS_D + MAS_C + MAS_B
          params_in =  params_E + params_D + params_C + params_B
          
          model_F = MF_DNN_MAS(layer_sizes_l,layer_sizes_nl, lr, restart = 1, params_t=params_in)
          
          uF, predE =           gen_prev_data_F(model_A, model_B, model_C, model_D, model_E, u_F,  
                                                params_A, params_B, params_C, params_D, params_E)
          uF_t, predE_t =       gen_prev_data_F(model_A, model_B, model_C, model_D, model_E, ut_F, 
                                                params_A, params_B, params_C, params_D, params_E)
          uF_full, predE_full = gen_prev_data_F(model_A, model_B, model_C, model_D, model_E, u,
                                                params_A, params_B, params_C, params_D, params_E)
          res_dir = results_dir_F
        
          if reloadF:
              params_F = model_F.unravel_params(np.load(res_dir + '/params.npy'))
              d_vx = scipy.io.loadmat(res_dir + "/MAS.mat")
              MAS_F  = unravel(d_vx["MAS"][0, :])
          else: 
              train_datasetF = DataGenerator_h(uF, predE, s_F, batch_size)
              test_datasetF = DataGenerator_h(uF_t, predE_t, st_F, uF_t.shape[0])
        
              model_F.train(train_datasetF, test_datasetF,  MAS, [l, l, l, l],  nIter=epochs)
              print('\n ... F Training done ...')
              
              scipy.io.savemat(res_dir +"losses.mat", 
                               {'training_error':model_F.error_training_log,
                                'testing_error':model_F.error_test_log,
                                'loss_total_log':model_F.loss_total_log}, format='4')
              
              
              params_F = model_F.get_params(model_F.opt_state)
              flat_params, _  = ravel_pytree(params_F)
              np.save(res_dir + 'params.npy', flat_params)
              
              save_data_MF(model_F, params_F, uF_full, predE_full, s, res_dir)
              
              MAS_F =model_F.compute_MAS(params_F, uF, predE,  random.PRNGKey(12345), 
                                    num_samples=MAS_num_samples, plot_diffs=True, disp_freq=40, scale=scaled)
              flat_MAS, _  = ravel_pytree(MAS_F)
              scipy.io.savemat(res_dir +"/MAS.mat",  {'MAS':flat_MAS}, format='4')
        
          reloadA = True
          reloadB = True

        

if __name__ == "__main__":
    # Modified
    dim_in = 784
    N_low = 40
    layer_sizes_A = [dim_in, N_low, N_low, N_low, 1]
    N_low=40

    layer_sizes_nl = [dim_in+1, N_low, N_low, N_low, 1]
    layer_sizes_l = [1, 1]
    scaled = False
    save_str_pre = "MF_narrow"
    run_MF_MAS(layer_sizes_A, layer_sizes_nl, layer_sizes_l, scaled, save_str_pre)


    N_low=80
    layer_sizes_nl = [dim_in+1, N_low, N_low, 1]
    scaled = False
    save_str_pre = "MF_wide"
    run_MF_MAS(layer_sizes_A, layer_sizes_nl, layer_sizes_l, scaled, save_str_pre)

            
    
    
