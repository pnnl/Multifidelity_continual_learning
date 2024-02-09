"""
Created 2023
@author: Amanda Howard (amanda.howard@pnnl.gov)

"""
import os
import scipy.io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from SF_funcs import DataGenerator_bcs, DataGenerator_res, DataGenerator_ICS_A, DNN_class
import jax
import jax.numpy as np
from jax.example_libraries import optimizers
from jax.flatten_util import ravel_pytree
from scipy.interpolate import griddata

def save_data(model, params, save_results_to):
    
        d_vx = scipy.io.loadmat("AC.mat")
        t, x, U_star = (d_vx["tt"].astype(np.float32), 
                   d_vx["x"].astype(np.float32), 
                   d_vx["uu"].astype(np.float32))

        t, x = np.meshgrid(t[0, :], x[0, :])
        X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))
        
        # Predictions
        u_pred = model.predict_u(params, X_star)
        U_pred = griddata(X_star, u_pred.flatten(), (t, x), method='cubic')

        u_pred_res = model.predict_res(params, X_star)
        U_pred_res = griddata(X_star, u_pred_res.flatten(), (t, x), method='cubic')

        fname= save_results_to +"beta_test.mat"
        scipy.io.savemat(fname, {'t':t,
                                  'x':x, 
                                  'U_star':U_star, 
                                  'U_pred_res':U_pred_res, 
                                  'U_pred':U_pred}, format='4')


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
    
def run_SF(N=200):
    
    save_suff = "SF_" + str(N)

    batch_size = 500
        
    N_low =N
    layers = [2, N_low, N_low, N_low, N_low, N_low, 1]
    
    batch_size_s = 100
    epochs = 100000
    lr = optimizers.exponential_decay(1e-4, decay_steps=2000, decay_rate=0.99)
    ics_weight = 100.0 #was 10
    res_weight = 1.0
    ut_weight = 1

    ymin_A = 0.0
    ymin_B = 1.0
    
    reloadA = False

    # ====================================
    # saving settings
    # ====================================
    save_str = save_suff
    results_dir_A = "../results_single/" + save_str + "/"
    if not os.path.exists(results_dir_A):
        os.makedirs(results_dir_A)
    

    # ====================================
    # DNN model A
    # ====================================
    model_A = DNN_class(layers, ics_weight, res_weight, ut_weight, lr, restart =0)
    ics_coords = np.array([[ymin_A, -1.0],[ymin_A, 1.0]])
    bc1_coords = np.array([[ymin_A, -1.0],[ymin_B, -1.0]])
    bc2_coords = np.array([[ymin_A, 1.0],[ymin_B, 1.0]])
    dom_coords = np.array([[ymin_A, -1.0],[ymin_B, 1.0]])

    ics_sampler = DataGenerator_ICS_A(2, ics_coords, lambda x: u0(x), batch_size)
    bc1 = DataGenerator_bcs(2, bc1_coords, bc2_coords, batch_size_s)
    bc2 = DataGenerator_bcs(2, bc2_coords, bc2_coords, batch_size_s)
    res_sampler = DataGenerator_res(2, dom_coords, batch_size)

    if reloadA:
        params_A = model_A.unravel_params(np.load(results_dir_A + '/params.npy'))
    else:     
        model_A.train(ics_sampler, bc1, bc2, res_sampler, nIter=epochs, F = 0, lam = [])

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

            
 # =============================================
 # =============================================
if __name__ == "__main__":

     run_SF() 
