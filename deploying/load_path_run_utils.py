## run_load_path_utils
## vb, 07.05.2026

import os, sys
import shutil
import time
import pickle

import numpy as np
import pandas as pd
import wandb

from deploy_utils import HiddenPrints, run_deployment, extend_material_parameters, get_paths
from dict_CC import dict_CC


def run_deployment_loadpath(inp_run, force, new_folder_path = None):
    """
    Run one load-step of the deployment
    inp_run         (dict)      contains all information for the deplyoment run (e.g. mat_tot_dict, numit, ...)
    force           (float)     force for current load step.
    new_folder_path (str)   folder where to save figures in the case that load steps are calculated based on previous load step
    """

    wandb.login()

    # 1 - get model version and epoch number
    v_model = inp_run['model_no'][0]
    ep_no = inp_run['model_no'][1]

    # 2 - define location of trained model and input data to be used 
    path_collection = get_paths(vnum=v_model, epnum=ep_no, vnumD=v_model, epnumD=ep_no)

    # 3 - update force to [N, kN], 
    mat_tot_dict_ = inp_run['mat_tot_dict']
    mat_tot_dict_.update({'F': mat_tot_dict_["L"]*force, 
                          'F_N': np.array([0])})

    if mat_tot_dict_['mat'] == 3:
        mat_tot_dict = extend_material_parameters(mat_tot_dict_)
    else: 
        mat_tot_dict = mat_tot_dict_

    mat_tot = mat_tot_dict
    print(mat_tot)

    ###########################################
    # Deployment with NN
    ###########################################

    if inp_run['predict'][0] or inp_run['predict'][1]:
        conv_plt = {'conv': True,
                    'else': False}
        simple = True                                           # Leave this on "True" for deployment.
        n_simple = 1                                            # Leave this on "1" for deployment
        NN_hybrid = {
            'predict_D': inp_run['predict'][1],                 # If true, solves the system with NN_hybrid solver (prediction of D); if False: "normal" solver
            'predict_sig': inp_run['predict'][0],               # If true, solves the system with NN_hybrid solver (prediction of sig); if False: "normal" solver
            'PERM': None,                                       # if true: permutates the values of the real stiffness matrix to simulate NN predictions
            'model_dim': 'TWODIM',                              # can be ONEDIM_y, ONEDIM_x, TWODIM or ALLDIM, only needs to be specified if NN is used.
            'numit': inp_run['numit'],
            }
        ####
        # Note: predict_D should not be used in lin.el. case, as there is only one initialisation and this happens with lin.el. model. 
        # => for lin.el. always set predict_D = False (glass / steel / RC)
        # => for nonlin: can choose what should be predicted.
        ####

        mat_res = run_deployment(mat_tot, conv_plt, simple, n_simple, NN_hybrid, path_collection, new_folder_path)


    ###########################################
    # Deployment without NN (= ground truth)
    ###########################################

    conv_plt = {'conv': True,
                'else': False}
    simple = True
    n_simple = 1
    NN_hybrid_2 = {'predict_D': False,
                'predict_sig': False,
                'PERM': None,
                'numit': inp_run['numit'],
                }

    mat_res = run_deployment(mat_tot, conv_plt, simple, n_simple, NN_hybrid_2, path_collection, new_folder_path)

    wandb.log(path_collection)
    wandb.log(NN_hybrid)

    return NN_hybrid, conv_plt


def save_deployment_loadpath(new_folder_path, force_i, NN_hybrid, conv_plt):
    '''
    plotting and saving to folders. Data will always be saved to new folder with timestamp
    
    Args: 
        new_folder_path     (str): path to folder where to save data
        force_i             (float): the force value for the current iteration
        NN_hybrid           (dict): which values are predicted (sig / D)
        conv_plt            (dict): the convergence plot parameters

    '''


    ####### which data files to save: #######
    if NN_hybrid['predict_sig'] and NN_hybrid['predict_D']:
            relative_path = ['data_out\\mat_res_norm.pkl', 'data_out\\mat_res_NN.pkl']
    elif NN_hybrid['predict_sig']:
            relative_path = ['data_out\\mat_res_norm.pkl', 'data_out\\mat_res_NN_sig.pkl']
    elif NN_hybrid['predict_D']:
            relative_path = ['data_out\\mat_res_norm.pkl', 'data_out\\mat_res_NN_D.pkl']
    else:
            relative_path = ['data_out\\mat_res_norm.pkl']


    ######## which plots to save: #######
    if conv_plt['conv'] and conv_plt['else']:
        if NN_hybrid['predict_sig'] and NN_hybrid['predict_D']:
            folder_paths = ['plots\\mike_plots', 'plots\\diagonal_plots', 'plots\\conv_plots', 
                        'plots\\eps_sig_plots_it', 'plots\\De_plots_it', 'plots\\mike_plots_D']
        elif NN_hybrid['predict_sig']:
            folder_paths = ['plots\\mike_plots', 'plots\\diagonal_plots', 'plots\\conv_plots', 
                        'plots\\eps_sig_plots_it']
        elif NN_hybrid['predict_D']:
            folder_paths = ['plots\\mike_plots', 'plots\\diagonal_plots_D', 'plots\\diagonal_plots_D_true', 'plots\\conv_plots', 
                        'plots\\eps_sig_plots_it', 'plots\\De_plots_it', 'plots\\mike_plots_D', 'plots\\mike_plots_D_true']
    elif conv_plt['conv']:
        folder_paths = ['plots\\conv_plots'] 
    elif not conv_plt['conv'] and not conv_plt['else']:
        pass 
        # do not save any plots to new folder


    ######## Copy files to new subfolder ########
    subfolder_path = os.path.join(new_folder_path, str(force_i))
    os.makedirs(subfolder_path, exist_ok = True)

    for i, file_path in enumerate(relative_path):
        file_path_n = os.path.join('05_Deploying', file_path)
        destination_path = os.path.join(subfolder_path, os.path.basename(file_path_n))
        shutil.copy(file_path_n, destination_path)
        print(f'File {i + 1} copied to {destination_path}')

    ######## Copy folders ########
    for folder in folder_paths:
        folder_path = os.path.join('05_Deploying', folder)
        source_folder = os.path.abspath(folder_path)
        destination_folder = os.path.join(subfolder_path, os.path.basename(folder_path))
        shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)
        print(f'Folder "{folder}" copied to {destination_folder}')
        for root, dirs, files in os.walk(source_folder, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                os.remove(file_path)
    


class deploy_utils():
    def __init__(self):
        #TODO!
        pass

    def run_deployment():
        #TODO!
        return


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout



def extend_material_parameters(mat_tot_dict_, mat_dict=dict_CC):
    # Add constant steel parameters
    fsy = 435               # [MPa]
    fsu = 470               # [MPa]
    Es = 205e3              # [MPa]
    Esh = 8e3               # [MPa]
    D = 16                  # [mm]

    mat_tot_dict_.update({'fsy': fsy, 'fsu': fsu, 'Es': Es, 'Esh': Esh, 'D': D})
    mat_tot_dict = mat_tot_dict_

    # Add additional concrete parameters
    index = int(np.where(mat_dict['CC'] == mat_tot_dict['CC'])[0])
    mat_tot_dict['E_1'] = mat_dict['Ec'][index]
    mat_tot_dict['tb0'] = mat_dict['tb0'][index]
    mat_tot_dict['tb1'] = mat_dict['tb1'][index]
    mat_tot_dict['ect'] = mat_dict['ect'][index]
    mat_tot_dict['ec0'] = mat_dict['ec0'][index]
    mat_tot_dict['fcp'] = mat_dict['fcp'][index]
    mat_tot_dict['fct'] = mat_dict['fct'][index]

    return mat_tot_dict





