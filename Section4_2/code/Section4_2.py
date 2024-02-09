"""
Created October 2023
@author: Amanda Howard (amanda.howard@pnnl.gov)
"""

import os
import scipy.io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from MF_funcs import DNN_class, MF_DNN_class

import jax
import jax.numpy as np
from jax.example_libraries import optimizers
from jax.flatten_util import ravel_pytree
from jax import random, grad, vmap, jit
import pandas as pd
from torch.utils import data
import itertools
from functools import partial
from jax.nn import relu

import numpy as onp
import matplotlib.pyplot as plt


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


def save_data(model, params, u, s, X_all, y_all,  save_results_to):
    S_full =  model.predict_u(params, u)
    S_all =  model.predict_u(params, X_all)


    fname= save_results_to +"output.mat"
    scipy.io.savemat(fname, { 'u_test': u,
                              'S_test':s, 
                              'S_pred_test':S_full,
                              'u_all': X_all, 
                              'S_all':y_all, 
                              'S_pred_all':S_all}, format='4')

    scipy.io.savemat(save_results_to +"losses.mat", 
                 {'training_loss':model.loss_training_log,
                  'testing_loss':model.loss_testing_log}, format='4')

    flat_params, _  = ravel_pytree(params)
    np.save(save_results_to + 'params.npy', flat_params)


def train_test_split(data_array, train_ratio=0.75):
    n_train = int(len(data_array)*train_ratio)
    data_train = data_array[:n_train,:]
    data_test = data_array[n_train:,:]
    return data_train, data_test


def train_test_split_months(data_array, months=[0, 1, 2], train_ratio=0.75):
    n_train = int(len(data_array)*train_ratio)
    data_train = data_array[:n_train,:]
        
    ind= np.asarray(np.where(data_array[:n_train, 5] == months[0]))

    for m in months[1:]:
        ind= np.hstack([ind, np.asarray(np.where(data_array[:n_train, 5] == m))])
    
    ind = ind[0, :]
    data_train = data_array[ind,:]
    return data_train



def generate_input_data_path(data_name):
    '''
    Generate the path to input data
    '''
    return '../processed_data/data/{}.csv'.format(data_name)


           
# =============================================
# =============================================
def MF_CL_script(region, months, reloadA, cont_learn_str, cont_learn_reload, lambda_vec, N_low, N_nl, save_str_pre, activation_func=np.tanh):
    layer_sizes_A = [5, N_low, N_low, N_low, 1]
    layer_sizes_nl = [6, N_nl, N_nl, N_nl, 1]
    layer_sizes_l = [1,5, 1]
    
    batch_size = 100
    epochs = 100000
    MAS_num_samples = 1201
    

    data = pd.read_csv(generate_input_data_path(f'{region}_daily'), index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.truncate(after='2019-07-01')
    data = data[['Electricity demand, daily sum, (GWh)',
                 'Temperature, daily mean (degC)',
                 'Temperature, daily peak (degC)',
                 'Holiday']].dropna()
    data_ml = np.concatenate((data[['Electricity demand, daily sum, (GWh)',
                                    'Temperature, daily mean (degC)',
                                    'Temperature, daily peak (degC)']].values,
                              data[['Holiday']].astype('int').values,
                              (data.index.weekday).values.reshape(-1, 1), 
                              (data.index.month).values.reshape(-1, 1)), axis=1)
    
    data_train, data_test = train_test_split(data_ml, train_ratio=0.75)
    X_train = data_train[:,1:]
    y_train = data_train[:,:1]
    X_test = data_test[:,1:]
    y_test = data_test[:,:1]
    X_all = np.concatenate((X_train, X_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    
    data_train_A = train_test_split_months(data_ml, months=[0, 1, 2], train_ratio=0.75)
    X_train_A = data_train_A[:,1:]
    y_train_A = data_train_A[:,:1]

    for i in np.arange(len(lambda_vec)):
          l = lambda_vec[i]

          save_str = save_str_pre + "_l" + str(lambda_vec[i])

          lr = optimizers.exponential_decay(1e-3, decay_steps=2000, decay_rate=0.99)

          # ====================================
          # saving settings
          # ====================================
          results_dir_A = "../resultsA/"+save_str_pre + "/"
          if not os.path.exists(results_dir_A):
              os.makedirs(results_dir_A)

          # ====================================
          # DNN model A
          # ====================================
          model_A = DNN_class(layer_sizes_A, lr, restart=0, activation_func=activation_func)
          train_dataset_A = DataGenerator(X_train_A, y_train_A, batch_size)
          test_dataset_A = DataGenerator(X_test, y_test, y_test.shape[0])
          
          if reloadA:
              params_A = model_A.unravel_params(np.load(results_dir_A + '/params.npy'))
          else:                       
              model_A.train(train_dataset_A, test_dataset_A, nIter=epochs, F = 0, lam=[])
              print('\n ... A Training done ...')

              params_A = model_A.get_params(model_A.opt_state)
              save_data(model_A, params_A, X_test, y_test, X_all, y_all, results_dir_A)
        
          # ====================================
          # MF-DNN training
          # ====================================
    
          lam = []
          params_prev = []
          MAS_in = []

          for cl_step in np.arange(len(cont_learn_str)):
              
              if cl_step == 0:
                  results_dir = "../results" + cont_learn_str[cl_step] +  "/"+save_str_pre + "/"
              else:
                  results_dir = "../results" + cont_learn_str[cl_step] +  "/"+save_str + "/"

        
              if not os.path.exists(results_dir):
                  os.makedirs(results_dir)
                  
              if cl_step == 0:
                 res = 0
              else: 
                 res = 1
                 
              data_train_A = train_test_split_months(data_ml, months=months[cl_step], train_ratio=0.75)
              X_train_A = data_train_A[:,1:]
              y_train_A = data_train_A[:,:1]
            
              train_dataset = DataGenerator(X_train_A, y_train_A, batch_size) #Change these!!!
              test_dataset = DataGenerator(X_test, y_test, y_test.shape[0])

              model = MF_DNN_class(layer_sizes_nl, layer_sizes_l, layer_sizes_A, lr, params_A, 
                                   restart = res, params_t =params_prev)

              if cont_learn_reload[cl_step]:
                   params = model.unravel_params(np.load(results_dir + '/params.npy'))
                   if cl_step == 0:
                       t = model.compute_MAS(params, X_train_A, random.PRNGKey(12345), 
                                                 num_samples=0, plot_diffs=False, disp_freq=40)
                       _, unravel  = ravel_pytree(t)
                   d_vx = scipy.io.loadmat(results_dir + "/MAS.mat")
                   MAS  = unravel(d_vx["MAS"][0, :])
              else:     
                model.train(train_dataset, test_dataset, nIter=epochs, F = MAS_in, lam = lam)
            
                print('\n ... ' +  cont_learn_str[cl_step] + ' Training done ...')

                params = model.get_params(model.opt_state)
                MAS =model.compute_MAS(params, X_train_A,   random.PRNGKey(12345), 
                                      num_samples=MAS_num_samples, plot_diffs=False, disp_freq=40)
                flat_MAS, _  = ravel_pytree(MAS)
                scipy.io.savemat(results_dir +"/MAS.mat",  {'MAS':flat_MAS}, format='4')
                save_data(model, params, X_test, y_test, X_all, y_all, results_dir)
                
              params_prev.append(params)
              lam.append(l)
        
              MAS_in.append(MAS)
              
              if cl_step == 0 and not cont_learn_reload[cl_step]:
                  t = model.compute_MAS(params, X_train_A, random.PRNGKey(12345), 
                                            num_samples=0, plot_diffs=False, disp_freq=40)
                  _, unravel  = ravel_pytree(t)
              cont_learn_reload[0] = True
              reloadA = True

def SF_CL_script(region, months, cont_learn_str, cont_learn_reload, lambda_vec,  N_low, save_str_pre, activation_func=np.tanh):
      layer_sizes_A = [5, N_low, N_low, N_low, 1]
      MAS_num_samples = 1201
      
      batch_size = 100
      epochs = 100000
      
      data = pd.read_csv(generate_input_data_path(f'{region}_daily'), index_col=0)
      data.index = pd.to_datetime(data.index)
      data = data.truncate(after='2019-07-01')  # do not use the last year to avoid the influence of COVID
      data = data[['Electricity demand, daily sum, (GWh)',
                   'Temperature, daily mean (degC)',
                   'Temperature, daily peak (degC)',
                   'Holiday']].dropna()
      data_ml = np.concatenate((data[['Electricity demand, daily sum, (GWh)',
                                      'Temperature, daily mean (degC)',
                                      'Temperature, daily peak (degC)']].values,
                                data[['Holiday']].astype('int').values,
                                (data.index.weekday).values.reshape(-1, 1), 
                                (data.index.month).values.reshape(-1, 1)), axis=1)
            
      data_train, data_test = train_test_split(data_ml, train_ratio=0.75)
      X_train = data_train[:,1:]
      y_train = data_train[:,:1]
      X_test = data_test[:,1:]
      y_test = data_test[:,:1]
      X_all = np.concatenate((X_train, X_test), axis=0)
      y_all = np.concatenate((y_train, y_test), axis=0)
      
      data_train_A = train_test_split_months(data_ml, months=[0, 1, 2], train_ratio=0.75)
      X_train_A = data_train_A[:,1:]
      y_train_A = data_train_A[:,:1]

      for i in np.arange(len(lambda_vec)):
            l = lambda_vec[i]
            
            save_str = save_str_pre + "_l" + str(lambda_vec[0])
      
            lr = optimizers.exponential_decay(1e-3, decay_steps=2000, decay_rate=0.99)

            # ====================================
            # DNN models
            # ====================================
            lam = []
            params_prev = []
            MAS_in = []
            
            for cl_step in np.arange(len(cont_learn_str)):
                
                if cl_step == 0:
                  results_dir = "../results" + cont_learn_str[cl_step] +  "/"+save_str_pre + "/"
                else:
                  results_dir = "../results" + cont_learn_str[cl_step] +  "/"+save_str + "/"
                  
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                    
                if cl_step == 0:
                   res = 0
                else: 
                   res = 1
                   
                data_train_A = train_test_split_months(data_ml, months=months[cl_step], train_ratio=0.75)
                X_train_A = data_train_A[:,1:]
                y_train_A = data_train_A[:,:1]
                   
              
                train_dataset = DataGenerator(X_train_A, y_train_A, batch_size) #Change these!!!
                test_dataset = DataGenerator(X_test, y_test, y_test.shape[0])
      
      
      
                model = DNN_class(layer_sizes_A, lr, restart=res, params_t = params_prev, activation_func=activation_func)
                
      
                if cont_learn_reload[cl_step]:
                     params = model.unravel_params(np.load(results_dir + '/params.npy'))
                     if cl_step == 0:
                         t = model.compute_MAS(params, X_train_A, random.PRNGKey(12345), 
                                                   num_samples=0, plot_diffs=False, disp_freq=40)
                         _, unravel  = ravel_pytree(t)
                     d_vx = scipy.io.loadmat(results_dir + "/MAS.mat")
                     MAS  = unravel(d_vx["MAS"][0, :])
                else:     
                  model.train(train_dataset, test_dataset, nIter=epochs, F = MAS_in, lam = lam)
              
                  print('\n ... ' +  cont_learn_str[cl_step] + ' Training done ...')
      
                  params = model.get_params(model.opt_state)
                  MAS =model.compute_MAS(params, X_train_A,   random.PRNGKey(12345), 
                                        num_samples=MAS_num_samples, plot_diffs=False, disp_freq=600)
                  flat_MAS, _  = ravel_pytree(MAS)
                  scipy.io.savemat(results_dir +"/MAS.mat",  {'MAS':flat_MAS}, format='4')
                  save_data(model, params, X_test, y_test, X_all, y_all, results_dir)
                  
                params_prev.append(params)
                lam.append(l)
          
                MAS_in.append(MAS)
                
                if cl_step == 0 and not cont_learn_reload[cl_step]:
                    t = model.compute_MAS(params, X_train_A, random.PRNGKey(12345), 
                                              num_samples=0, plot_diffs=False, disp_freq=40)
                    _, unravel  = ravel_pytree(t)
      
                save_str = save_str_pre + "_l" + str(lambda_vec[i])
      
                cont_learn_reload[0] = True
      
                  

                
if __name__ == "__main__":
    months = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]] #Dividing up the dataset for continual learning

    cont_learn_str = ['B', 'C', 'D', 'E']
    
    reloadA = False #allows to reload from a previously trained model. 

    lambda_vec = [.0001, .001, .01, .1, 1, 10, 100, 1000, 0] #lambda values for MAS. \lambda = 0 corresponds to 
                                                #learning without MAS

    MF_name = "_MF_100_100"
    SF_name = "_SF_100"    
    noCL_name = "_noCL"
    
 #   MF_CL_script('la', months, reloadA, cont_learn_str, [False, False, False, False], lambda_vec, 
 #                100, 100, 'la_' + MF_name, activation_func=np.tanh)
 #   MF_CL_script('ny', months, reloadA, cont_learn_str, [False, False, False, False], lambda_vec, 
 #                100, 100, 'ny_' + MF_name, activation_func=np.tanh)
 #   MF_CL_script('sac', months, reloadA, cont_learn_str, [False, False, False, False], lambda_vec, 
 #                100, 100, 'sac_' + MF_name, activation_func=np.tanh)

  #  SF_CL_script('la', months, cont_learn_str, [False, False, False, False], lambda_vec, 100, 
  #               'la' + SF_name, activation_func=np.tanh)
  #  SF_CL_script('ny', months, cont_learn_str, [False, False, False, False], lambda_vec, 100, 
  #               'ny'+ SF_name, activation_func=np.tanh)
  #  SF_CL_script('sac', months, cont_learn_str, [False, False, False, False], lambda_vec,100, 
  #              'sac'+ SF_name, activation_func=np.tanh)

    months = [np.arange(12)]
   # SF_CL_script('la', months, [''], [False], [0], 40, 'la' + noCL_name, activation_func=relu)
   # SF_CL_script('ny', months, [''], [False], [0], 40, 'ny' + noCL_name, activation_func=relu)
   # SF_CL_script('sac', months, [''], [False], [0], 40, 'sac' + noCL_name, activation_func=relu)
    

    linestyles =  ['-', '-', '-']
    colors = ['#59a14f', '#4e79a7', '#e15759']
    markers =  ['', 's', '^']
    labels = ['NY', 'Sac.', 'LA']
    regions = ['ny', 'sac', 'la']
    
    fig1, ax = plt.subplots(figsize=(7, 4))
    fig2, bx = plt.subplots(nrows=3, ncols=2, figsize=(12, 7))
    fig2.subplots_adjust(left=.5, wspace=2)


    for ip in onp.arange(len(regions)):
        region = regions[ip]
        data = pd.read_csv(generate_input_data_path(f'{region}_daily'), index_col=0)
        data.index = pd.to_datetime(data.index)
        data = data.truncate(after='2019-07-01')  # do not use the last year to avoid the influence of COVID
        data = data[['Electricity demand, daily sum, (GWh)',
                     'Temperature, daily mean (degC)',
                     'Temperature, daily peak (degC)',
                     'Holiday']].dropna()
        train_ratio=0.75
        n_train = int(len(data)*train_ratio)
    
        data_dir = "../results/" + region  + noCL_name
        d_vx = scipy.io.loadmat(data_dir + "/output.mat")
        uHF, s_xHF, predHF= (d_vx["u_test"].astype(onp.float32), 
                    d_vx["S_test"].astype(onp.float32),
                    d_vx["S_pred_test"].astype(onp.float32))
        
        errorsSF = onp.sqrt(onp.mean((s_xHF[:, 0]-predHF[0, :])**2))

        errors = onp.zeros(len(lambda_vec))
        for i in onp.arange(len(lambda_vec)): 
            data_dir = "../resultsE/" + region  + SF_name + "_l" +  str(lambda_vec[i])
            d_vx = scipy.io.loadmat(data_dir + "/output.mat")
            predHF= d_vx["S_pred_test"].astype(onp.float32)
            errors[i] = onp.sqrt(onp.mean((s_xHF[:, 0]-predHF[0, :])**2))

        errorsMF = onp.zeros(len(lambda_vec))
        for i in onp.arange(len(lambda_vec)): 
            data_dir = "../resultsE/" + region  +"_" + MF_name + "_l" + str(lambda_vec[i])
            d_vx = scipy.io.loadmat(data_dir + "/output.mat")
            predHF= d_vx["S_pred_test"].astype(onp.float32)
            errorsMF[i] = onp.sqrt(onp.mean((s_xHF[:, 0]-predHF[0, :])**2))

                
    
        plt.figure(fig1.number)
        plt.loglog(lambda_vec[:-1], errors[:-1], colors[ip], marker=markers[ip],linestyle=linestyles[ip], label=labels[ip] + ' SF')
        plt.loglog(lambda_vec[:-1], errorsMF[:-1], colors[ip], marker=markers[ip], linestyle='--', label=labels[ip] + ' MF')
        plt.legend()
        out_str =str("{:10.4f}".format(errorsSF)) + ' & ' +str("{:10.4f}".format(errors[-1])) + ' & ' +  str("{:10.4f}".format(errorsMF[-1]))  + ' & ' + \
            str("{:10.4f}".format(min(errors[:-1])))+ ' & ' + str("{:10.4f}".format(min(errorsMF[:-1]))) + ' \\\\'
        print(out_str)
    
        
        plt.figure(fig2.number)
        data_dir = "../resultsE/" + region  +"_" + MF_name + "_l" +"0"
    
        d_vx = scipy.io.loadmat(data_dir + "/output.mat")
        pred= (d_vx["S_pred_test"].astype(np.float32))
        bx[ip, 0].plot(data[n_train:].index, pred[0, :], color=colors[1], linewidth=2, label='Results MF')

        data_dir = "../resultsE/" + region  + SF_name + "_l" +"0"
        d_vx = scipy.io.loadmat(data_dir + "/output.mat")
        pred= (d_vx["S_pred_test"].astype(np.float32))
        bx[ip, 0].plot(data[n_train:].index, pred[0, :], color=colors[2], linewidth=2, label='Results SF')

        bx[ip, 0].plot(data[n_train:].index, s_xHF[:, 0], 'k-',label='Test data')
        if ip == 0:
            bx[ip, 0].legend(fontsize=12, ncol=1)

        bx[ip, 0].set(xticks=['2019-01-01'])
    
        bx[ip, 0].set_xlim([pd.to_datetime('2018-07-01', format = '%Y-%m-%d'),
                 pd.to_datetime('2019-07-01', format = '%Y-%m-%d')])



        data_dir = "../resultsE/" + region  +"_" + MF_name + "_l" +str(lambda_vec[np.argmin(errorsMF[:-1])])
    
        d_vx = scipy.io.loadmat(data_dir + "/output.mat")
        pred= (d_vx["S_pred_test"].astype(np.float32))
        bx[ip, 1].plot(data[n_train:].index, pred[0, :], color=colors[1], linewidth=2, label='Results MF')

        data_dir = "../resultsE/" + region  + SF_name + "_l" + str(lambda_vec[np.argmin(errors[:-1])])
        d_vx = scipy.io.loadmat(data_dir + "/output.mat")
        pred= (d_vx["S_pred_test"].astype(np.float32))
        bx[ip, 1].plot(data[n_train:].index, pred[0, :], color=colors[2], linewidth=2, label='Results SF')

        bx[ip, 1].plot(data[n_train:].index, s_xHF[:, 0], 'k-',label='Test data')
      #  if ip == 0:
       #     bx[ip, 1].legend(fontsize=12, ncol=2)

        bx[ip, 1].set(xticks=['2019-01-01'])
        bx[ip, 1].tick_params(labelsize=14)
        bx[ip, 0].tick_params(labelsize=14)

        bx[ip, 1].set_xlim([pd.to_datetime('2018-07-01', format = '%Y-%m-%d'),
                 pd.to_datetime('2019-07-01', format = '%Y-%m-%d')])
        bx[ip, 1].set_ylabel('Daily Energy Use (GWh)', fontsize=12)
        bx[ip, 0].set_ylabel('Daily Energy Use (GWh)', fontsize=12)



    for j in range(3):
        bx[j, 1].yaxis.set_label_coords(-.1, 0.5)
        bx[j, 0].yaxis.set_label_coords(-.12, 0.5)



    plt.figure(fig1.number)
    plt.xlabel('$\lambda$', fontsize=20)
    plt.ylabel('RMSE (GWh)', fontsize=20)
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1.05))
    
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig('Figure17.png', format='png')


    plt.figure(fig2.number)
    plt.tight_layout()
    plt.savefig('Figure16.png', format='png')
    
    
    
    
    fig4, gx = plt.subplots(figsize=(10, 4))
    plt.figure(fig4.number)
    for  i in onp.arange(3):
        region = regions[i]
    
        data = pd.read_csv(generate_input_data_path(f'{region}_daily'), index_col=0)
        data.index = pd.to_datetime(data.index)
        data = data.truncate(after='2019-07-01')  # do not use the last year to avoid the influence of COVID
        data = data[['Electricity demand, daily sum, (GWh)',
                     'Temperature, daily mean (degC)',
                     'Temperature, daily peak (degC)',
                     'Holiday']].dropna()

        data_dir = "../results/" + region  + noCL_name
    
        
        d_vx = scipy.io.loadmat(data_dir + "/output.mat")
        uHF, s_xHF, predHF= (d_vx["u_all"].astype(onp.float32), 
                    d_vx["S_all"].astype(onp.float32),
                    d_vx["S_pred_all"].astype(onp.float32))


        plt.plot(data[:].index, s_xHF[:, 0], color=colors[i], label=labels[i])
    
    plt.ylabel('Daily Energy Use (GWh)', fontsize=20)
    #plt.xlabel('', fontsize=20)
    gx.fill_between(['2015-12-31', '2016-03-31'], -100, 700, alpha=.6, facecolor='#4e79a7', label='Dataset 1')
    gx.fill_between(['2016-12-31', '2017-03-31'], -100, 700, alpha=.6, facecolor='#4e79a7')
    gx.fill_between(['2017-12-31', '2018-03-31'], -100, 700, alpha=.6, facecolor='#4e79a7')
    
    gx.fill_between(['2016-03-31', '2016-06-30'], -100, 700, alpha=.6, facecolor='#F28E2B', label='Dataset 2')
    gx.fill_between(['2017-03-31', '2017-06-30'], -100, 700, alpha=.6, facecolor='#F28E2B')
    gx.fill_between(['2018-03-31', '2018-06-30'], -100, 700, alpha=.6, facecolor='#F28E2B')

    gx.fill_between(['2016-06-30', '2016-09-30'], -100, 700, alpha=.6, facecolor='#BAB0AC', label='Dataset 3')
    gx.fill_between(['2017-06-30', '2017-09-30'], -100, 700, alpha=.6, facecolor='#BAB0AC')
    gx.fill_between(['2015-06-30', '2015-09-30'], -100, 700, alpha=.6, facecolor='#BAB0AC')
    
    gx.fill_between(['2016-09-30', '2016-12-31'], -100, 700, alpha=.6, facecolor='#59a14f', label='Dataset 4')
    gx.fill_between(['2017-09-30', '2017-12-31'], -100, 700, alpha=.6, facecolor='#59a14f')
    gx.fill_between(['2015-09-30', '2015-12-31'], -100, 700, alpha=.6, facecolor='#59a14f')
    
    gx.fill_between(['2018-06-30', '2019-07-01'], -100, 700, alpha=.6, facecolor='#e15759', label='Test data')

    plt.legend(fontsize=12, bbox_to_anchor=(1.02, 1.05))

    gx.set_xticks(['2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01'])
    plt.ylim([0, 700])

    gx.set_xlim([pd.to_datetime('2015-07-01', format = '%Y-%m-%d'),
             pd.to_datetime('2019-07-01', format = '%Y-%m-%d')])
    gx.set_xticklabels(['2016', '2017', '2018', '2019'])
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig('Figure15.png', format='png')
    
    
    
            