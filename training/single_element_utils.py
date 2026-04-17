#vb, 16.04.2026


import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sampling.sampler_utils_RC3D import *
from sampling.simulating_sig_vec_RC3D import *
from test_utils import predict_D, predict_sig, get_inp_from_folder, test_model_instance

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def single_element_test(idx_eps, geom, model_path, save_path = None, 
                        multirow = False, NN_comp = None, all_cols = False, train_points = False):
    
    """
    Conduct single-element-test for idx_eps (e.g. 0 = pure tension in x-direction)

    Creates a new set of data for epsilon in just one direction (idx_eps) and calculates from it: 
        - sig_NN, sig_NLFEA
        - D_NN, D_NLFEA    
    Plots these values against each other for selected idx_sig and idx_D. This is like one single iteration in the FEA loop.
    Requires generation of new "predictions" with NN / calculations with NLFEA



    Args:
        idx_eps         (list)      desired dimension for input epsilon - if len(idx_eps) >1: the first eps_idx will be varied, the second will be in three steps.
        geom            (list)      geometrical input parameters (t, rho, CC)
        idx_sig         (list)      desired dimension for output sigma (if multirow = False)
        idx_D           (list)      desired dimension for output D (if multirow = False)
        model_path      (str)       path to trained model, inp and stats
        save_path       (str)       location to save the figure
        multirow        (bool)      if True: plots 6 rows with all sig_i and D_i corresponding to the selected eps_i
        NN_comp         (str-list)  if not None: Contains the path and ep number to a second NN which shall be 
                                    compared to the first NN in model_path (e.g. ['04_Training\\new_data\\_simple_logs\\v_3xx', '_xx', 'TWODIM'])
        allcols         (bool)      if True, prints all predictions of all stiffness matrix entries, not just the ones related to the varied epsilon
        train_points    (bool)      if True, will plot training points of the corresponding geometry and stress / stiffness 
                                    in addition to the predicted and NLFEA curves
    
    Returns: 
        figure of stress path for single element test
    
    """

    if all_cols or train_points: 
        raise UserWarning('The functionality for "all_cols" or "train_points" has not yet been implemented.')
    

    # Step 1: Sample a meaningful vector for idx_eps
    min_, max_ = -3e-3, 5e-3
    inp_vector = sample_idx_eps(idx_eps, min_, max_, geom)
    print(f'Sampled eps values.')

    # Step 2a: Calculate all NLFEA values for given eps input
    constants, mat_dict = get_constant_sampling_params(sample_2d = False)
    sig_D_NLFEA = calculate_sig_D_NLFEA(inp_vector['X_test'], constants, mat_dict, cm = 3)
    print('Calculated NLFEA values')

    # Step 2b: Calculate all LFEA values for given eps input
    sig_D_linel = calculate_sig_D_NLFEA(inp_vector['X_test'], constants, mat_dict, cm = 1)
    print('Calculated LFEA values')


    # Step 2c: Calculate all NN values for given eps input
    sig_D_NN = predict_sig_D_NN(inp_vector, model_path, NN_comp)
    print('Calculated NN values')


    # Step 2d: (Optional) Fetch training points to plot based on given geometry
    # TODO!


    # Step 3: Plot the figures
    plot_single_element_test(idx_eps, inp_vector['X_test'], sig_D_NLFEA, sig_D_linel, sig_D_NN, 
                             multirow, all_cols, save_path)



    return


################################ auxiliary functions for single element test ################################



def sample_idx_eps(idx_eps, min_idx_eps, max_idx_eps, geom, num_samples = 100, small_value = 1e-20):
    '''
    samples eps_inp only for the dimension given in idx_eps  
    
    idx_eps      (list)       desired dimension for input epsilon
    geom         (list)       geometrical input parameters (t, rho, CC)
    model_path   (str)        path to sampled data
    range_factor (float)      to reduce the max. range of epsilons in the input vector
    num_samples  (int)        amount of values to be sampled in eps

    '''


    if len(idx_eps) < 2: 
        eps_vec = small_value*np.ones((num_samples, 6))                  # other values are set to "zero", i.e. small_value here
        idx_eps_vec = np.linspace(min_idx_eps, max_idx_eps, num_samples)
        if idx_eps[0] > 2 and idx_eps[0] < 6:
            idx_eps_vec = idx_eps_vec/10                           # convert to 1/mm
        eps_vec[:,idx_eps[0]] = idx_eps_vec
        
        if len(geom) > 0:
            t_vec = np.tile(np.array(geom), (num_samples, 1))
            X_test = np.concatenate((eps_vec, t_vec), axis = 1)
        else: 
            X_test = eps_vec

        test_data = {
             'X_test': X_test,
             'y_test': np.zeros_like(X_test)        # dummy variable (empty).
        }

    else:
        raise UserWarning('This has not yet been double-checked for multirow = True.')
        eps_vec = small_value*np.ones((num_samples, 8))                  # other values are set to "zero", i.e. small_value here
        idx_eps_vec = np.linspace(min_idx_eps, max_idx_eps, num_samples)
        if idx_eps[0] > 2 and idx_eps[0] < 6:
            idx_eps_vec = idx_eps_vec/10                           # convert to 1/mm
        eps_vec[:,idx_eps[0]] = idx_eps_vec

        max_idx_eps1 = np.max(mat_data_np['X_train'][:,idx_eps[1]])
        min_idx_eps1 = np.min(mat_data_np['X_train'][:,idx_eps[1]])
        eps_vec_min, eps_vec_max = small_value*np.ones((num_samples,8)), small_value*np.ones((num_samples,8))
        eps_vec_min[:,idx_eps[0]] = idx_eps_vec                             # assign same varying values of eps_0
        eps_vec_max[:,idx_eps[0]] = idx_eps_vec                             # assign same varying values of eps_0
        eps_vec_min[:,idx_eps[1]] = np.tile(min_idx_eps1, (num_samples,))  # add constant min or max values of eps_1
        eps_vec_max[:,idx_eps[1]] = np.tile(max_idx_eps1, (num_samples,))

        t_vec = np.tile(np.array(geom), (num_samples, 1))

        inp_vec = {
             'min': np.concatenate((eps_vec_min, t_vec), axis = 1),
             '0': np.concatenate((eps_vec, t_vec), axis = 1),
             'max': np.concatenate((eps_vec_max, t_vec), axis = 1),
        }


    return test_data

def calculate_sig_D_NLFEA(eps_g: np.array, constants: dict, mat_dict: dict, cm: int):
    """
    calculates sig and D based on sig-simulator (same as used for sampling the data)

    Args: 
        test_data     (np.arr): contains the relevant eps-values.
        constants       (dict): constant values for sampling (contains values like t, rho_x, rho_y, CC, ...)
        mat_dict        (dict): additional constant values for sampling (material parameters that can be derived from "constants")    
        cm               (int): cm = 1: linear elastic, cm = 3: nonlinear
    
    Returns: 
        sig_D_NLFEA     (dict): containing two arrays: sig_g and dh
    """
    with HiddenPrints():
        simulatesig = SigSimulator(constants)

        # 2.1 Find layer strains
        e = simulatesig.find_e_vec(eps_g)

        # 2.2 Find layer stresses
        s = simulatesig.find_s_vec(e, mat_dict, cm_klij = cm)

        # 2.3 Find generalised stresses
        sig_g = simulatesig.find_sh_vec(s, cm_klij = cm)

        # 2.4 Find stiffnesses
        dh = simulatesig.find_dh_vec(s, mat_dict, cm_klij = cm)
        
        sig_D_NLFEA = {
                        'sig': sig_g,
                        'D': dh,
                }
    
    
    return sig_D_NLFEA

def predict_sig_D_NN(inp_vector:dict, model_path:str, NN_comp:bool):
    """
    Predicts sig_g and D based on trained model. If NN_comp is True: calculates with two different versions of NN to enable comparison.
    
    """
    with HiddenPrints():
        inp = get_inp_from_folder(model_path[0], model_path[1])
        test_model = test_model_instance(inp, model_path[0], model_path[1])
        stats = get_stats_from_folder(model_path[0], model_path[1])

        sig_NN = predict_sig(test_model, inp, inp_vector, stats, sobolev = inp['Sobolev'])
        De_NN = predict_D(test_model, inp, inp_vector, stats, sobolev = inp['Sobolev'])

        sig_D_NN = {
                    'sig': sig_NN,
                    'D': De_NN,
                }
        
        if NN_comp: 
            raise UserWarning('This has not yet been implemented.')
    

    return sig_D_NN
    
def get_stats_from_folder(model_path: str, model_version: int):
    """
    Extract "stats" dict from given trained model version

    Args: 
        model_path      (str):  path to train logs
        model_version   (int):  version of interest

    Returns:
        stats           (dict): containing all statistically relevant data from training data of trained model
    """

    full_path = os.path.join(model_path, 'v_'+str(model_version))
    with open(os.path.join(full_path, 'stats.pkl'),'rb') as handle:
        stats = pickle.load(handle)

    return stats



################################ auxiliary functions for plotting ################################ 

def plot_single_element_test(idx_eps:list, eps_data:np.array, sig_D_NLFEA:dict, sig_D_linel:dict, sig_D_NN:dict,
                             multirow: bool, all_cols: bool, save_path: str):
    """
    Plots the results of the single-element-test

    Args:
        idx_eps     (list)


    Returns: 
        fig     saves figure at indicated location
    
    """


    fig, axs = plt.subplots(1,2, figsize = [21, 7], squeeze = False)

    # Plot predictions NN vs NLFEA vs LFEA for sigma and D
    plot_comparison(axs[0,0], idx_eps, eps_data, sig_D_NLFEA['sig'], sig_D_linel['sig'], sig_D_NN['sig']['pred'], multirow)
    plot_comparison(axs[0,1], idx_eps, eps_data, sig_D_NLFEA['D'], sig_D_linel['D'], sig_D_NN['D']['pred'], multirow)

    # Add title
    figure_formatting_single_el(fig, axs, idx_eps, multirow)
    
    # Save figure
    save_single_el_plot(fig, save_path, multirow, idx_eps)

    return



def plot_comparison(axs, idx_eps, eps_data, NLFEA_data, linel_data, NN_data, multirow):
    """
    actual plotting function
    """

    if not multirow: 
        if len(NLFEA_data.shape) > 2:
            axs.plot(eps_data[:,idx_eps[0]], NLFEA_data[:,idx_eps[0],idx_eps[0]], color = 'black', label = 'NLFEA')
            axs.plot(eps_data[:,idx_eps[0]], linel_data[:, idx_eps[0], idx_eps[0]], color = 'lightgrey', linestyle = ':', label = 'LFEA')
            axs.plot(eps_data[:,idx_eps[0]], NN_data[:, idx_eps[0],idx_eps[0]], color = 'lightblue', linestyle = '--', label = 'NN')
        
        else: 
            axs.plot(eps_data[:,idx_eps[0]], NLFEA_data[:,idx_eps[0]], color = 'black', label = 'NLFEA')
            axs.plot(eps_data[:,idx_eps[0]], linel_data[:, idx_eps[0]], color = 'lightgrey', linestyle = ':', label = 'LFEA')
            axs.plot(eps_data[:,idx_eps[0]], NN_data[:, idx_eps[0]], color = 'lightblue', linestyle = '--', label = 'NN')


    else: 
        print('Not yet tested...')
        for i in range(axs.shape[0]):
            # if multirow
            if len(NLFEA_data.shape) > 2:
                axs[i].plot(eps_data[:,idx_eps[0]], NLFEA_data[:,i,idx_eps[0]], color = 'black', label = 'NLFEA')
                axs[i].plot(eps_data[:,idx_eps[0]], linel_data[:, i, idx_eps[0]], color = 'lightgrey', linestyle = ':', label = 'LFEA')
                axs[i].plot(eps_data[:,idx_eps[0]], NN_data[:, i,idx_eps[0]], color = 'lightblue', linestyle = '--', label = 'NN')
            
            else: 
                axs[i].plot(eps_data[:,idx_eps[0]], NLFEA_data[:,i], color = 'black', label = 'NLFEA')
                axs[i].plot(eps_data[:,idx_eps[0]], linel_data[:, i], color = 'lightgrey', linestyle = ':', label = 'LFEA')
                axs[i].plot(eps_data[:,idx_eps[0]], NN_data[:, i], color = 'lightblue', linestyle = '--', label = 'NN')

    return


def figure_formatting_single_el(fig, axs, idx_eps, multirow):
    """
    adding labels and title
    
    """
    
    sig_labels = np.array(['n_x', 'n_y', 'n_xy', 'm_x', 'm_y', 'm_xy'])
    eps_labels = np.array(['eps_x', 'eps_y', 'eps_xy', 'chi_x', 'chi_y', 'chi_xy'])
    D_labels = np.array([[f'D_{i}{j}' for j in range(6)] for i in range(6)])


    # set axis labels
    for i in range(axs.shape[0]):
        if not multirow:
            axs[0,0].set_ylabel(sig_labels[idx_eps[0]])
            axs[0,1].set_ylabel(D_labels[idx_eps[0], idx_eps[0]])
        else: 
            raise UserWarning('not yet implemented.')
        for j in range(axs.shape[1]):
            axs[0,j].set_xlabel(eps_labels[idx_eps[0]])

    

    # set title "Variation of idx_eps"
    fig.suptitle(f'Variation of {eps_labels[idx_eps[0]]}')
    
    
    return


def save_single_el_plot(fig, save_path, multirow, idx_eps):
    """
    saves plot to specified folder

    Args: 
        fig             (fig):  figure
        save_path       (str):  str, location where figure shall be saved
        multirow        (bool): True if multiple rows for every different stress shall be shown.
        idx_eps         (list): Varied epsilon value.


    Returns: 
        saved figure

    """
    
    if save_path is not None and not multirow: 
        filename = 'stress_path_'+str(idx_eps[0])
        save_folder = os.path.join(os.getcwd(), save_path)
        fig.savefig(os.path.join(save_folder, filename))
        print('Saved stress path figure ', filename, ' at ', save_path)
    elif multirow: 
        filename = 'stress_path_'+str(idx_eps[0])+'_multirow.png'
        save_folder = os.path.join(os.getcwd(), save_path)
        fig.savefig(os.path.join(save_folder, filename))
        print('Saved stress path multirow figure at ', save_path)


    return


