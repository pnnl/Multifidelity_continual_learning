"""
Created 2023
@author: Amanda Howard (amanda.howard@pnnl.gov)

"""
import os
import scipy.io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from MF_funcs import MF_DNN_class
from SF_funcs import DNN_class, DataGenerator_bcs, DataGenerator_res, DataGenerator_ICS_A

import jax
import jax.numpy as np
from jax import random, jit
from jax.example_libraries import optimizers
from jax.flatten_util import ravel_pytree
from functools import partial
from torch.utils import data
from scipy.interpolate import griddata


def get_pts(model, params, dom_coords, key=random.PRNGKey(1234), N = 10000):
    key, subkey = random.split(key)
    k = 2
    c = 0
    
    X_star = dom_coords[0:1,:] + (dom_coords[1:2,:]-dom_coords[0:1,:])*\
        random.uniform(key, shape=[N, 2])

    # Predictions
    u_pred_res = model.predict_res(params, X_star)
    err = u_pred_res**k/np.mean( u_pred_res**k) + c
    err_norm = err/np.sum(err)

    return X_star, err_norm

def save_data(model, params, save_results_to):
        d_vx = scipy.io.loadmat("AC.mat")
        t, x, U_star = (d_vx["tt"].astype(np.float32), 
                   d_vx["x"].astype(np.float32), 
                   d_vx["uu"].astype(np.float32))
        t, x = np.meshgrid(t[0, :], x[0, :])
        X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))
        
        # Predictions
        u_pred = model.predict_u(params, X_star)
        u_pred_res = model.predict_res(params, X_star)

        U_pred = griddata(X_star, u_pred.flatten(), (t, x), method='cubic')
        U_pred_res = griddata(X_star, u_pred_res.flatten(), (t, x), method='cubic')

        fname= save_results_to +"beta_test.mat"
        scipy.io.savemat(fname, {'t':t,
                                  'x':x, 
                                  'U_star':U_star, 
                                  'U_pred_res':U_pred_res, 
                                  'U_pred':U_pred}, format='4')

def gen_prev_data(u, model_A, params_A):
    predA = model_A.predict_u(params_A, u)
    return predA


class DataGenerator_res_RDPS(data.Dataset):
    def __init__(self, dim, res_pts, coords,err_norm,
                 batch_size=64, batch_size_res=32, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.dim = dim
        self.res_pts = res_pts
        self.N = res_pts.shape[0]
        self.err_norm = err_norm
        self.coords = coords

        self.batch_size = batch_size

        self.batch_size_res = batch_size_res
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs = self.__data_generation(subkey)
        return inputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        batch_1 = self.batch_size_res
        batch_2 = self.batch_size-batch_1
        idx = random.choice(key, self.N, (batch_1,), replace=False, p=self.err_norm)
        key, subkey = random.split(key)
        ax1 = self.res_pts[idx,:]
        
        ax2 = self.coords[0:1,:] + (self.coords[1:2,:]-
                                    self.coords[0:1,:])*\
            random.uniform(key, shape=[batch_2, self.dim])
     
        x = np.concatenate([ax1, ax2])
        return x

# Define the exact solution and its derivatives
def u0(x):
    """
    :param x: x = (t, x)
    """
    t = x[:,0:1]
    x = x[:,1:2]
    return x*x*np.cos(np.pi * x) 
                
# =============================================
# =============================================
    
def run_MF(replay, MAS,  MASl=[0], N=200):
    
    save_suff = "MF_" + str(N)
    Npts = 1
    if replay:
        save_suff = save_suff + "_replay"
    if MAS:
        save_suff = save_suff + "_MAS"

        Npts = 50000
    
    batch_size = 1000 #500
    batch_size_res = 0

    
    MAS_num_samples = 10
    if MAS:
        MAS_num_samples = 3000
    scaled = False
        
    N_low =200
    layers = [2, N_low, N_low, N_low, N_low, N_low, 1]
    N_low=N
    layer_sizes_nl = [3, N_low, N_low, N_low, N_low, N_low, 1]
    layer_sizes_l = [1, 20, 1]

    batch_size_s = 100
    epochs = 100000
    epochsA= 50000


    lr = optimizers.exponential_decay(5e-4, decay_steps=2000, decay_rate=0.99)
    lrA = optimizers.exponential_decay(1e-4, decay_steps=2000, decay_rate=0.99)
    ics_weight = 100.0 
    res_weight = 1.0
    ut_weight = 1


    ymin_A = 0.0
    ymin_B = 0.25
    ymin_C = 0.5
    ymin_D = 0.75
    ymin_E = 1.0
    
    reloadA = False
    reloadA2 = False
    reloadB = False
    reloadC = False
    reloadD = False
    

    for l in MASl:
        save_str = save_suff + "_l_" + str(l)
        # ====================================
        # saving settings
        # ====================================
        results_dir_A = "../results_A/MF/"
        if not os.path.exists(results_dir_A):
            os.makedirs(results_dir_A)
        results_dir_A2 = "../results_B/" + save_suff + "/"
        if not os.path.exists(results_dir_A2):
            os.makedirs(results_dir_A2)
                
        results_dir_B = "../results_C/"+save_str+"/"
        if not os.path.exists(results_dir_B):
            os.makedirs(results_dir_B)
        results_dir_C= "../results_D/"+save_str+"/"
        if not os.path.exists(results_dir_C):
            os.makedirs(results_dir_C)
        results_dir_D= "../results_E/"+save_str+"/"
        if not os.path.exists(results_dir_D):
            os.makedirs(results_dir_D)            
    
    
        # ====================================
        # DNN model A
        # ====================================
        model_A = DNN_class(layers, ics_weight, res_weight, ut_weight, lrA, restart =0)
    
        ics_coords = np.array([[ymin_A, -1.0],[ymin_A, 1.0]])
        bc1_coords = np.array([[ymin_A, -1.0],[ymin_B, -1.0]])
        bc2_coords = np.array([[ymin_A, 1.0],[ymin_B, 1.0]])
        dom_coords = np.array([[ymin_A, -1.0],[ymin_B, 1.0]])
    
        ics_sampler = DataGenerator_ICS_A(2, ics_coords, lambda x: u0(x), batch_size)
        bc1 = DataGenerator_bcs(2, bc1_coords, bc2_coords, batch_size_s)
        bc2 = DataGenerator_bcs(2, bc1_coords, bc2_coords, batch_size_s)
        res_sampler = DataGenerator_res(2, dom_coords, batch_size)
    
        if reloadA:
            params_A = model_A.unravel_params(np.load(results_dir_A + '/params.npy'))
        else:     
            model_A.train(ics_sampler, bc1, bc2, res_sampler, nIter=epochsA, F = 0, lam = [])
    
            print('\n ... A Training done ...')
            scipy.io.savemat(results_dir_A +"losses.mat", 
                         {'training_loss':model_A.loss_training_log,
                          'res_loss':model_A.loss_res_log,
                          'ics_loss':model_A.loss_ics_log,
                          'ut_loss':model_A.loss_ut_log}, format='4')
        
            params_A = model_A.get_params(model_A.opt_state)
            flat_params, _  = ravel_pytree(model_A.get_params(model_A.opt_state))
            np.save(results_dir_A + 'params.npy', flat_params)
        save_data(model_A, params_A, results_dir_A)
    
    
        # ====================================
        # DNN model A2
        # ====================================
        
        model_A2 = MF_DNN_class(layer_sizes_nl, layer_sizes_l, ics_weight, res_weight, ut_weight, 
                                lr, lambda x: gen_prev_data(x, model_A, params_A), restart =0)
        model = model_A2
        results_dir = results_dir_A2
        ics_coords = np.array([[ymin_A, -1.0],[ymin_A, 1.0]])
        bc1_coords = np.array([[ymin_A, -1.0],[ymin_B, -1.0]])
        bc2_coords = np.array([[ymin_A, 1.0],[ymin_B, 1.0]])
        dom_coords = np.array([[ymin_A, -1.0],[ymin_B, 1.0]])
    
        rng_key = random.PRNGKey(1234)
        key, subkey = random.split(rng_key)
    
        ics_sampler = DataGenerator_ICS_A(2, ics_coords,  lambda x: u0(x), batch_size)
        bc1 = DataGenerator_bcs(2, bc1_coords, bc2_coords, batch_size_s)
        bc2 = DataGenerator_bcs(2, bc1_coords, bc2_coords, batch_size_s)
        res_sampler = DataGenerator_res(2, dom_coords,  batch_size)
    
        if reloadA2:
            params_A2 = model.unravel_params(np.load(results_dir + '/params.npy'))
            t = model.compute_MAS(params_A2, dom_coords, random.PRNGKey(12345), 
                                      num_samples=0, plot_diffs=True, disp_freq=100, scale=scaled)
            _, unravel  = ravel_pytree(t)
            d_vx = scipy.io.loadmat(results_dir + "/MAS.mat")
            MAS_A  = unravel(d_vx["MAS"][0, :])
        else:     
            model.train(ics_sampler, bc1, bc2, res_sampler, nIter=epochsA, F = 0, lam = [])
    
            print('\n ... A2 Training done ...')
            scipy.io.savemat(results_dir +"losses.mat", 
                         {'training_loss':model.loss_training_log,
                          'res_loss':model.loss_res_log,
                          'ics_loss':model.loss_ics_log,
                          'ut_loss':model.loss_ut_log}, format='4')
        
            params_A2 = model.get_params(model.opt_state)
            flat_params, _  = ravel_pytree(params_A2)
            np.save(results_dir + 'params.npy', flat_params)
        
            MAS_A =model.compute_MAS(params_A2, dom_coords, random.PRNGKey(12345), 
                                      num_samples=MAS_num_samples, plot_diffs=True, disp_freq=100, scale=scaled)
            flat_MAS, _  = ravel_pytree(MAS_A)
            scipy.io.savemat(results_dir +"/MAS.mat",  {'MAS':flat_MAS}, format='4')
            save_data(model, params_A2, results_dir)
     
        model_A2 = model
    
    
        # ====================================
        # DNN model B
        # ====================================
        model_B = MF_DNN_class(layer_sizes_nl, layer_sizes_l, ics_weight, res_weight, ut_weight, lr,
                               lambda x: gen_prev_data(x, model_A2, params_A2), restart =1, params_t = params_A2)
        model = model_B
        results_dir = results_dir_B
        ymin = ymin_B
        if replay:
            ymin = 0.0
        ymax = ymin_C
        ics_coords = np.array([[ymin, -1.0],[ymin, 1.0]])
        bc1_coords = np.array([[ymin, -1.0],[ymax,-1.0]])
        bc2_coords = np.array([[ymin, 1.0], [ymax, 1.0]])
        dom_coords = np.array([[ymin, -1.0],[ymax, 1.0]])
    
        if reloadB:
            params_B = model.unravel_params(np.load(results_dir + '/params.npy'))
            t = model.compute_MAS(params_B, dom_coords,  random.PRNGKey(12345), 
                                      num_samples=0, plot_diffs=False, disp_freq=40, scale=scaled)
            _, unravel  = ravel_pytree(t)
            d_vx = scipy.io.loadmat(results_dir + "/MAS.mat")
            MAS_B  = unravel(d_vx["MAS"][0, :])
        else:     
            x, err_norm = get_pts(model_A2, params_A2, dom_coords,  N = Npts)
    
            ics_sampler = DataGenerator_ICS_A(2, ics_coords,  lambda x: gen_prev_data(x, model_A2, params_A2), batch_size)
            bc1 = DataGenerator_bcs(2, bc1_coords, bc2_coords, batch_size_s)
            bc2 = DataGenerator_bcs(2, bc1_coords, bc2_coords, batch_size_s)
        
            res_sampler = DataGenerator_res_RDPS(2, x, dom_coords, err_norm, batch_size, batch_size_res)
        
        
            model.train(ics_sampler, bc1, bc2, res_sampler, nIter=epochs, F = MAS_A, lam = [l])
    
            print('\n ... B Training done ...')
            scipy.io.savemat(results_dir +"losses.mat", 
                         {'training_loss':model.loss_training_log,
                          'res_loss':model.loss_res_log,
                          'ics_loss':model.loss_ics_log,
                          'ut_loss':model.loss_ut_log}, format='4')
        
            params_B = model.get_params(model.opt_state)
            flat_params, _  = ravel_pytree(params_B)
            np.save(results_dir + 'params.npy', flat_params)
            save_data(model, params_B, results_dir)
    
            MAS_B =model.compute_MAS(params_B, dom_coords, random.PRNGKey(12345), 
                                      num_samples=MAS_num_samples, plot_diffs=True, disp_freq=100, scale=scaled)
            flat_MAS, _  = ravel_pytree(MAS_B)
            scipy.io.savemat(results_dir +"/MAS.mat",  {'MAS':flat_MAS}, format='4')
            
            
        model_B = model
            
        # ====================================
        # DNN model C
        # ====================================
        MAS = MAS_B + MAS_A
        params_in =  params_B + params_A2
    
        model_C = MF_DNN_class(layer_sizes_nl, layer_sizes_l, ics_weight, res_weight, ut_weight, lr, 
                               lambda x: gen_prev_data(x, model_B, params_B), restart =1, params_t = params_in)
        model = model_C
        results_dir = results_dir_C
        ymin = ymin_C
        if replay:
            ymin = 0.0
        ymax = ymin_D
        ics_coords = np.array([[ymin, -1.0],[ymin, 1.0]])
        bc1_coords = np.array([[ymin, -1.0],[ymax,-1.0]])
        bc2_coords = np.array([[ymin, 1.0],[ymax, 1.0]])
        dom_coords = np.array([[ymin, -1.0],[ymax, 1.0]])
    
        if reloadC:
             params_C = model.unravel_params(np.load(results_dir + '/params.npy'))
             t = model.compute_MAS(params_C, dom_coords, random.PRNGKey(12345), 
                                      num_samples=0, plot_diffs=False, disp_freq=40, scale=scaled)
             _, unravel  = ravel_pytree(t)
             d_vx = scipy.io.loadmat(results_dir + "/MAS.mat")
             MAS_C  = unravel(d_vx["MAS"][0, :])
        else:     
             x, err_norm  = get_pts(model_B, params_B, dom_coords, N = Npts)
    
    
             ics_sampler = DataGenerator_ICS_A(2, ics_coords,  lambda x: gen_prev_data(x, model_B, params_B), batch_size)
             bc1 = DataGenerator_bcs(2, bc1_coords, bc2_coords, batch_size_s)
             bc2 = DataGenerator_bcs(2, bc1_coords, bc2_coords, batch_size_s)
             res_sampler = DataGenerator_res_RDPS(2, x, dom_coords, err_norm, batch_size, batch_size_res)
    
            
             model.train(ics_sampler, bc1, bc2, res_sampler, nIter=epochs, F = MAS, lam = [l, l])
        
             print('\n ... C Training done ...')
             scipy.io.savemat(results_dir +"losses.mat", 
                          {'training_loss':model.loss_training_log,
                           'res_loss':model.loss_res_log,
                           'ics_loss':model.loss_ics_log,
                           'ut_loss':model.loss_ut_log}, format='4')
         
             params_C = model.get_params(model.opt_state)
             flat_params, _  = ravel_pytree(params_C)
             np.save(results_dir + 'params.npy', flat_params)
             save_data(model, params_C, results_dir)
        
             MAS_C =model.compute_MAS(params_C, dom_coords, random.PRNGKey(12345), 
                                      num_samples=MAS_num_samples, plot_diffs=True, disp_freq=100, scale=scaled)
             flat_MAS, _  = ravel_pytree(MAS_C)
             scipy.io.savemat(results_dir +"/MAS.mat",  {'MAS':flat_MAS}, format='4')
        model_C = model 
        
        # ====================================
        # DNN model D
        # ====================================
        MAS = MAS_C + MAS_B + MAS_A
        params_in = params_C + params_B + params_A2
        
        model_D = MF_DNN_class(layer_sizes_nl, layer_sizes_l, ics_weight, res_weight, ut_weight, lr, 
                               lambda x: gen_prev_data(x, model_C, params_C),
                               restart =1, params_t = params_in)
        model = model_D
        results_dir = results_dir_D
        
        ymin = ymin_D
        if replay:
            ymin = 0.0
        ymax = ymin_E
        ics_coords = np.array([[ymin, -1.0],[ymin, 1.0]])
        bc1_coords = np.array([[ymin, -1.0],[ymax,-1.0]])
        bc2_coords = np.array([[ymin, 1.0],[ymax, 1.0]])
        dom_coords = np.array([[ymin, -1.0],[ymax, 1.0]])
    
        
        if reloadD:
             params_D = model.unravel_params(np.load(results_dir + '/params.npy'))
             t = model.compute_MAS(params_D, dom_coords, random.PRNGKey(12345), 
                                      num_samples=0, plot_diffs=False, disp_freq=40, scale=scaled)
             _, unravel  = ravel_pytree(t)
             d_vx = scipy.io.loadmat(results_dir + "/MAS.mat")
             MAS_D  = unravel(d_vx["MAS"][0, :])
        else:     
             x, err_norm = get_pts(model_C, params_C, dom_coords, N = Npts)
             ics_sampler = DataGenerator_ICS_A(2, ics_coords,  lambda x: gen_prev_data(x, model_C, params_C), batch_size)
             bc1 = DataGenerator_bcs(2, bc1_coords, bc2_coords, batch_size_s)
             bc2 = DataGenerator_bcs(2, bc1_coords, bc2_coords, batch_size_s)
             res_sampler = DataGenerator_res_RDPS(2, x, dom_coords, err_norm, batch_size, batch_size_res)
    
             model.train(ics_sampler, bc1, bc2, res_sampler, nIter=epochs, F = MAS, lam = [l, l, l])
        
             print('\n ... D Training done ...')
             scipy.io.savemat(results_dir +"losses.mat", 
                          {'training_loss':model.loss_training_log,
                           'res_loss':model.loss_res_log,
                           'ics_loss':model.loss_ics_log,
                           'ut_loss':model.loss_ut_log}, format='4')
         
             params_D = model.get_params(model.opt_state)
             flat_params, _  = ravel_pytree(params_D)
             np.save(results_dir + 'params.npy', flat_params)
             save_data(model, params_D, results_dir)
        
             MAS_D =model.compute_MAS(params_D, dom_coords, random.PRNGKey(12345), 
                                      num_samples=MAS_num_samples, plot_diffs=True, disp_freq=100, scale=scaled)
             flat_MAS, _  = ravel_pytree(MAS_D)
             scipy.io.savemat(results_dir +"/MAS.mat",  {'MAS':flat_MAS}, format='4')
        
        reloadA = True
        reloadA2 = True
        
            
# =============================================
# =============================================
if __name__ == "__main__":
    MAS_vec = [.001, .01, .1, 1, 10, 100]

    replay = False
    MAS = False
    run_MF(replay, MAS, N=300) #MF
        
    replay = True
    MAS = False
    run_MF(replay, MAS, N=100) #MF-replay
    run_MF(replay, MAS, N=200) #MF-replay
    run_MF(replay, MAS, N=300) #MF-replay
    run_MF(replay, MAS, N=400) #MF-replay
        
    replay = False
    MAS = True
    run_MF(replay, MAS, MASl=MAS_vec) #MF-MAS




    
    
    
                        
