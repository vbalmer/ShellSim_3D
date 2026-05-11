import os, sys
import numpy as np
import time
import pickle
import pandas as pd
import os
from deploy_utils import HiddenPrints, run_deployment, extend_material_parameters, get_paths
import wandb

def run_deployment_loadpath(inp_run, force, new_folder_path = None):
    """
    Run one load-step of the deployment
    inp_run         (dict)      contains all information for the deplyoment run (e.g. mat_tot_dict, numit, ...)
    force           (float)     force for current load step.
    new_folder_path (str)   folder where to save figures in the case that load steps are calculated based on previous load step
    """

    wandb.login()

    numpy_sampler = False
    single_sample = False
    load_steps = True
    v_model = inp_run['model_no'][0]
    ep_no = inp_run['model_no'][1]

    # define location of trained model and input data to be used
    # path_collection = get_paths(vnum='182', epnum='19926')                                                # for lin.el. RC case
    # path_collection = get_paths(vnum='198', epnum='9997')                                                 # for glass case   
    # path_collection = get_paths(vnum='431', epnum='9272', vnumD='431', epnumD='9272')                     # for nonlin. RC case  
    path_collection = get_paths(vnum=v_model, epnum=ep_no, vnumD=v_model, epnumD=ep_no)


    # Inputs for iteration

    # If the data comes from the sampler:
    if numpy_sampler:
        raise Warning('Outdated')
        path = '..\\01_SamplingFeatures'
        # name = 'data_240724_1752_case4\outfile.npy'
        name = 'data_20241104_1545_case8\outfile.npy'
        features = np.load(os.path.join(path, name))
        mat_tot_dict_ = {
            'L': features[:,0],         # length
            'B': features[:,1],         # width
            'E_1': features[:,2],       # Young's modulus steel / glass / concrete
            'E_2': features[:,3],       # Young's modulus - / interlayer / reinforcing steel
            'ms': features[:,4],        # mesh_size
            'F': features[:,5],         # force_magnitude
            's': features[:,6],         # scenario 0...6
            't_1': features[:,7],       # thickness of the plate
            't_2': features[:,8],       # thickness of plate
            'nl': features[:,9],        # amount of layers
            'nu_1': features[:,10],     # Poisson's ratio
            'nu_2': features[:,11],     # Poisson's ratio
            'mat': features[:,12]       # Material type     (1 = lin.el., 3 = CMM, 10 = glass)
        }

    elif numpy_sampler == False and not single_sample and not load_steps: 
        raise Warning('Outdated')
        path = '..\\01_SamplingFeatures'
        name = 'output\\data_20241111_0904_case10\\outfile.pkl'
        with open(os.path.join(path, name),'rb') as handle:
                in_dict = pickle.load(handle)
        mat_tot_dict_ = in_dict

    if single_sample:
        raise UserWarning('This was the input for individual runs, before the function definition')
        # pure shear force (for RC linear, mat = 1, scenario 8)
        # mat_tot_dict_ = {
        #     'L': np.array([6000]),
        #     'B': np.array([6000]),
        #     'CC': np.array([1]),
        #     'E_1': np.array([0]),
        #     'E_2': np.array([0]),
        #     'ms': np.array([600]),
        #     'F': np.array([3.3e6]),       #=10*L
        #     'F_N': np.array([0]),         # not required in this case
        #     's': np.array([8]),
        #     't_1': np.array([300]),
        #     't_2': np.array([0]),
        #     'nl': np.array([20]),
        #     'nu_1': np.array([0]),
        #     'nu_2': np.array([0]),
        #     'mat': np.array([3]),
        #     'rho': np.array([0.025]),
        # }

        # pure tension (for RC linear, mat = 1, scenario 8)
        # mat_tot_dict_ = {
        #     'L': np.array([6000]),
        #     'B': np.array([6000]),
        #     'CC': np.array([1]),
        #     'E_1': np.array([0]),
        #     'E_2': np.array([0]),
        #     'ms': np.array([600]),
        #     'F': np.array([-3.3e6]),       #=[n_x]*L
        #     'F_N': np.array([0]),         # not required in this case
        #     's': np.array([9]),
        #     't_1': np.array([300]),
        #     't_2': np.array([0]),
        #     'nl': np.array([20]),
        #     'nu_1': np.array([0]),
        #     'nu_2': np.array([0]),
        #     'mat': np.array([3]),
        #     'rho': np.array([0.025]),
        # }

        # combined in-plane action (for RC linear, mat = 1, scenario 8)
        # mat_tot_dict_ = {
        #     'L': np.array([6000]),
        #     'B': np.array([6000]),
        #     'CC': np.array([1]),
        #     'E_1': np.array([0]),
        #     'E_2': np.array([0]),
        #     'ms': np.array([600]),
        #     'F': np.array([-3.3e6]),       #=10*L
        #     'F_N': np.array([0]),         # not required in this case
        #     's': np.array([112]),
        #     't_1': np.array([300]),
        #     't_2': np.array([0]),
        #     'nl': np.array([20]),
        #     'nu_1': np.array([0]),
        #     'nu_2': np.array([0]),
        #     'mat': np.array([3]),
        #     'rho': np.array([0.025]),
        # }



        # moment + normal force (for RC linear, mat = 1, scenario 11)
        # mat_tot_dict_ = {
        #     'L': np.array([6000]),
        #     'B': np.array([6000]),
        #     'CC': np.array([2]),
        #     'E_1': np.array([0]),
        #     'E_2': np.array([0]),
        #     'ms': np.array([600]),
        #     'F': np.array([0.005]),           # Uniformly distributed load in z-direction
        #     'F_N': np.array([600000]),       # Normal force (n[N/m]*L); e.g. n=50kN/m --> F_N = 100*7500 = 750000
        #     's': np.array([11]),
        #     't_1': np.array([250]),
        #     't_2': np.array([0]),
        #     'nl': np.array([20]),
        #     'nu_1': np.array([0]),
        #     'nu_2': np.array([0]),
        #     'mat': np.array([3]),
        #     'rho': np.array([0.015])
        # }

        # bending 2D (for glass, mat = 10, scenario 20)
        # mat_tot_dict_ = {
        #     'L': np.array([1500]),
        #     'B': np.array([1500]),
        #     'E_1': np.array([70000]),
        #     'E_2': np.array([300]),
        #     'ms': np.array([150]),
        #     'F': np.array([0.005]),
        #     'F_N': np.array([0]),       # not required in this case.
        #     's': np.array([20]),
        #     't_1': np.array([5]),
        #     't_2': np.array([0.4]),
        #     'nl': np.array([5]),
        #     'nu_1': np.array([0.23]),
        #     'nu_2': np.array([0.5]),
        #     'mat': np.array([10])
        # }

        # bending 2D (for RC, mat = 3, scenario 20, 22, 23)
        # mat_tot_dict_ = {
        #     'L': np.array([6000]),
        #     'B': np.array([6000]),
        #     'CC': np.array([2]),
        #     'E_1': np.array([0]),
        #     'E_2': np.array([0]),
        #     'ms': np.array([600]),
        #     'F': np.array([0.03]),
        #     'F_N': np.array([0]),       # not required in this case.
        #     's': np.array([22]),
        #     't_1': np.array([250]),
        #     't_2': np.array([0]),
        #     'nl': np.array([20]),
        #     'nu_1': np.array([0]),
        #     'nu_2': np.array([0]),
        #     'mat': np.array([3]),
        #     'rho': np.array([0.015])
        # }


        # bending 1D (for RC nonlinear, mat = 3, scenario 10)
        # mat_tot_dict_ = {
        #     'L': np.array([6000]),
        #     'B': np.array([6000]),
        #     'CC': np.array([2]),
        #     'E_1': np.array([0]),
        #     'E_2': np.array([0]),
        #     'ms': np.array([600]),
        #     'F': np.array([0.005]),
        #     'F_N': np.array([0]),       # not required in this case.
        #     's': np.array([22]),
        #     't_1': np.array([250]),
        #     't_2': np.array([0]),
        #     'nl': np.array([13]),
        #     'nu_1': np.array([0]),
        #     'nu_2': np.array([0]),
        #     'mat': np.array([3]),
        #     'rho': np.array([0.01])
        # }

    if load_steps: 
        mat_tot_dict_ = inp_run['mat_tot_dict']
        mat_tot_dict_.update({'F': mat_tot_dict_["L"]*force, 
                              'F_N': np.array([0])})


    if mat_tot_dict_['mat'] == 3:
        mat_tot_dict = extend_material_parameters(mat_tot_dict_)
    else: 
        mat_tot_dict = mat_tot_dict_

    # mat_tot_raw = pd.DataFrame.from_dict(mat_tot_dict)

    if not single_sample and not load_steps:
        raise Warning('Outdated')
        # import mat_res.pkl from data that was used for training of algorithm (if it comes from global sampling..)
        path_mat_res = '..\\02_Simulator'
        # name = 'Simulator\\results\saved_runs\data_240805_1134_case4\mat_res.pkl'        # take 240805 here, as there the SN are included
        # name = 'Simulator\\results\saved_runs\data_20241104_1854_case8\mat_res.pkl'
        name = 'Simulator\\results\saved_runs\data_20241111_1101_case10\mat_res.pkl'
        with open(os.path.join(path_mat_res, name),'rb') as handle:
            mat_res = pickle.load(handle)

        # Choose the simulation number(s) that shall be tested
        desired_SN = [21, 32, 45, 74, 76, 82]
        mat_tot = mat_tot_raw.iloc[[np.where(mat_res['SN'] == i)[0][0] for i in desired_SN]]
        mat_tot.reset_index(drop=True, inplace=True)
        print('As a check, in mat_tot (from sampling) (t1_tot = ', round(mat_tot['t_1'][0], 1), 'mm) should be equal to t1 in mat_res (directly read) (t1_res = ', round(mat_res['t_1'][desired_SN[0]],1), 'mm).')
    else: 
        mat_tot = mat_tot_dict

    print(mat_tot)


    ###########################################
    # Deployment with NN
    ###########################################

    conv_plt = {'conv': True,
                'else': False}
    simple = True                                           # Leave this on "True" for deployment.
    # samples = int(mat_tot.shape[0])
    n_simple = 1                                            # can also be len(desired_SN) if imported data from simulations (not single_sample)
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
    mat_res_pd_NN = pd.DataFrame.from_dict(mat_res)


    ###########################################
    # Deployment without NN (= ground truth)
    ###########################################

    conv_plt = {'conv': True,
                'else': False}
    simple = True  
    # samples = int(mat_tot.shape[0])
    n_simple = 1
    NN_hybrid_2 = {'predict_D': False,
                'predict_sig': False,
                'PERM': None,
                'numit': inp_run['numit'],
                }

    mat_res = run_deployment(mat_tot, conv_plt, simple, n_simple, NN_hybrid_2, path_collection, new_folder_path)
    mat_res_pd = pd.DataFrame.from_dict(mat_res)


    ###########################################################################
    # Deployment without NN but with permutation of stiffness matrix
    ###########################################################################

    # conv_plt = {'conv': True,
    #             'else': False}
    # simple = True  
    # # samples = int(mat_tot.shape[0])
    # n_simple = 1
    # NN_hybrid_3 = {'predict_D': False,
    #              'predict_sig': False,
    #              'PERM': None,
    #              'PERM1': [0.8, 1.2]}

    # mat_res = run_deployment(mat_tot, conv_plt, simple, n_simple, NN_hybrid_3, path_collection)
    # mat_res_pd = pd.DataFrame.from_dict(mat_res)

    wandb.log(path_collection)
    wandb.log(NN_hybrid)

    return NN_hybrid, conv_plt


def save_deployment_loadpath(new_folder_path, force_i, NN_hybrid, conv_plt):
    #########################################
    # Plotting and saving to folders
    #########################################


    # if data should be saved to folder instead of being overwritten with the next simulation, use save_folder = True

    import shutil
    import os

    save_folder = True

    if save_folder:
        ####### which data files to save: #######
        if NN_hybrid['predict_sig'] and NN_hybrid['predict_D']:
                relative_path = ['data_out\\mat_res_norm.pkl', 'data_out\\mat_res_NN.pkl']
        elif NN_hybrid['predict_sig']:
                relative_path = ['data_out\\mat_res_norm.pkl', 'data_out\\mat_res_NN_sig.pkl']
        elif NN_hybrid['predict_D']:
                relative_path = ['data_out\\mat_res_norm.pkl', 'data_out\\mat_res_NN_D.pkl']
        else:
                relative_path = ['data_out\\mat_res_norm.pkl']

        # if NN_hybrid_3['PERM'] is not None: 
        #         relative_path.append('data_out\\mat_res_norm_perm.pkl')

        # if NN_hybrid_3['PERM1'] is not None:
        #         relative_path.append('data_out\\mat_res_norm_perm1.pkl')

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


        # Copy files to new subfolder
        subfolder_path = os.path.join(new_folder_path, str(force_i))
        os.makedirs(subfolder_path, exist_ok = True)

        for i, file_path in enumerate(relative_path):
            file_path_n = os.path.join('05_Deploying', file_path)
            destination_path = os.path.join(subfolder_path, os.path.basename(file_path_n))
            shutil.copy(file_path_n, destination_path)
            print(f'File {i + 1} copied to {destination_path}')

        # Copy folders
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
    
    