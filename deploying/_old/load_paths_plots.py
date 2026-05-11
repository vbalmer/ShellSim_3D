# Plots and predictions for stress paths
# bav, 10.11.2025


import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Plot_NoDash import read
from stress_paths_plots import get_colors_from_map


def plot_load_path_wrapper(path_depl, case_study, until_load_level, save_path, type, thresh):
    '''
    Wrapper function around plot_load_paths. Allows for plotting one of the two scenarios: 
    (1)         if path_depl is a single path: plots different values of eps_x, eps_y, eps_xy in one plot against load.
    (2)         if path_depl is a dict:        plots contents of dict (with different case studies and different rho_y)
    
    path_depl           (str / dict)        path to deployment runs
    case_study          (str)               only needed for case (1) - see function definition of plot_load_paths
    until_load_level    (list)              only needed for case (1) - see function definition of plot_load_paths
    save_path           (str)               location where to save the plot
    thresh              (float)             threshold at which the displacement and load level should be cut off to plot nicely.

    '''


    if isinstance(path_depl, dict):
        # plotting multiple load-deformation paths
        load_steps = {}
        mat_displ = {}
        for key0 in path_depl.keys(): # case study
            load_steps[key0] = {}
            mat_displ[key0] = {}
            for key1 in path_depl[key0].keys(): # rho_i
                load_steps[key0][key1], mat_displ[key0][key1] = plot_load_paths('05_Deploying\\data_out\\'+path_depl[key0][key1], case_study = key0, 
                                                                                until_load_level = None, save_path = None, type = type)
        multi_plot_load_deform(load_steps, mat_displ, save_path, type, thresh)

    else: 
        # plotting one single load-deformation path
        _, _ = plot_load_paths(path_depl, case_study, until_load_level, save_path, type)
    
    return


def plot_load_paths(path_depl, case_study, until_load_level, save_path, type):
    """
    plots load-deformation path of load sequence given in path_depl folder for specified case study.
    Assuming deployment with predictions of sigma and D at the same time.
 
    path_depl       (str)       path to deployment folder (EXcluding subfolder name like "100")
    case_study      (str)       Case study (e.g. 2D-1 = pure shear, 2D-2 = pure tension in y-direction, etc.)
    until_load_level(list)      list of [lowest load level, highest load level] as index. Can be used for debugging when don't want to plot all levels.
                                If "None": plots all load levels available in the folder.
    save_path       (str)       Path to save figure at. If path = None: Will not save the figure and instead return the variables to plot multi-plot.
    type            (str)       Either 'u' or 'eps'. To either plot n against u or epsilon.

    """

    # 1 - determine load steps
    load_steps = get_loadsteps(path_depl)

    # 2 - determine relevant graph dimensions (x, y or xy)
    relevant_dim = get_relevant_dim(case_study)

    # 3 - determine displacements or strains (for NN and NLFEA)
    if type == 'u':
        displ_NN    = get_max_displ(relevant_dim, load_steps, path_depl, tag = 'NN')
        displ_NLFEA = get_max_displ(relevant_dim, load_steps, path_depl, tag = 'norm')
        mat_displ = {
            'NN': displ_NN,
            'NLFEA': displ_NLFEA
        }
    elif type == 'eps':
        eps_NN    = get_max_eps(relevant_dim, load_steps, path_depl, tag = 'NN')
        eps_NLFEA = get_max_eps(relevant_dim, load_steps, path_depl, tag = 'norm')
        mat_displ = {
            'NN': eps_NN,
            'NLFEA': eps_NLFEA
        }

    if save_path is not None: 
        # plot the data and save the graph
        plot_load_deform(load_steps, mat_displ, case_study, until_load_level, save_path, type)

    return load_steps, mat_displ


##### subfunctions #####

def get_loadsteps(path_depl): 

    subfolders = [name for name in os.listdir(path_depl)
              if os.path.isdir(os.path.join(path_depl, name))]

    load_steps = np.array(subfolders, dtype = np.float32)

    sorted_load_steps = load_steps[np.argsort(np.abs(load_steps))]

    return sorted_load_steps


def get_relevant_dim(case_study):

    if case_study not in ['2D-1', '2D-2', '2D-3', '2D-4', '2D-5', '2D-1C', '2D-8C']:
        raise UserWarning('TODO: Combination cases not yet implemented.')

    # mapping = {
    #     "2D-1": [0,1,2],                # indices for displacements, in the order [ux, uy, uz, thx, thy, thz] or [epsx, epsy, epsxy]
    #     "2D-2": [0,1], 
    #     "2D-3": [0,1],
    #     "2D-4": [0,1],
    #     "2D-5": [0,1],
    #     "2D-1C": [0,1,2],
    # }

    mapping = {                     # for just plotting one strain / displacement per load-deformation path.
        "2D-1": [2],                # indices for displacements, in the order [ux, uy, uz, thx, thy, thz] or [epsx, epsy, epsxy]
        "2D-2": [1], 
        "2D-3": [1],
        "2D-4": [0],
        "2D-5": [0],
        "2D-1C": [2],
        "2D-8C": [0,1,2],
    }

    relevant_dim = mapping.get(case_study)

    return relevant_dim


def get_max_displ(relevant_dim, load_steps, path_depl, tag):
    """
    get maximum displacements from the results of the calculation

    tag = 'NN' or 'norm' (for NLFEA)
    """

    displ = np.zeros((len(load_steps), max(len(relevant_dim),1)))

    for i, force_i in enumerate(load_steps):
        with open(os.path.join(path_depl+'\\'+str(int(force_i)), 'mat_res_'+tag+'.pkl'),'rb') as handle:
                    mat_res = pickle.load(handle)
        
        ux, uy, uz, thx, thy, thz = mat_res['ux'], mat_res['uy'], mat_res['uz'], mat_res['thx'], mat_res['thy'], mat_res['thz']

        u_all_max = [np.max(np.abs(ux)), np.max(np.abs(uy)), np.max(np.abs(uz)),
                      np.max(np.abs(thx)), np.max(np.abs(thy)), np.max(np.abs(thz))]

        for j,k in enumerate(relevant_dim):
            displ[i, j] = u_all_max[k]

    return displ

def get_max_eps(relevant_dim, load_steps, path_depl, tag):
    '''
    get maximum strain from the results of the calculation

    tag = 'NN' or 'norm' (for NLFEA)
    ''' 
    strains = np.zeros((len(load_steps), max(len(relevant_dim),1)))

    for i, force_i in enumerate(load_steps):
        with open(os.path.join(path_depl+'\\'+str(int(force_i)), 'mat_res_'+tag+'.pkl'),'rb') as handle:
                    mat_res = pickle.load(handle)
        

        eps_g = mat_res['eps_g'][0]
        if len(eps_g.shape) == 3:
            all_strains = [np.max(np.abs(eps_g[:,0,0])), np.max(np.abs(eps_g[:,0,1])), np.max(np.abs(eps_g[:,0,2])),
                        np.max(np.abs(eps_g[:,0,3])), np.max(np.abs(eps_g[:,0,4])), np.max(np.abs(eps_g[:,0,5]))] 
        elif len(eps_g.shape) == 4:
            all_strains = [np.max(np.abs(eps_g[:,0,0,0])), np.max(np.abs(eps_g[:,0,0,1])), np.max(np.abs(eps_g[:,0,0,2])),
                        np.max(np.abs(eps_g[:,0,0,3])), np.max(np.abs(eps_g[:,0,0,4])), np.max(np.abs(eps_g[:,0,0,5]))] 

        for j,k in enumerate(relevant_dim):
            strains[i, j] = all_strains[k]*1e3      #strains in permille 

    return strains

def plot_load_deform(load_steps, mat_displ, case_study, until_load_level, save_path, type): 
    if until_load_level == None: 
        step = len(load_steps)
        step_l = 1
    else:
        step = until_load_level[1]
        step_l = until_load_level[0]
    print(f'First and last load level are {load_steps[step_l-1]} kN/m and {load_steps[step-1]} kN/m')
    num_cols = 1
    _, colors2, _ = get_colors_from_map({'0': 0, '1': 0, '2': 0})


    fig, ax = plt.subplots(1, 1, figsize = [num_cols*8, 5])
    if type == 'u':
        labels = ['u_x', 'u_y', 'u_z']
    elif type == 'eps':
        labels = ['eps_x', 'eps_y', 'gamma_xy']
    if num_cols == 1:
        ax = np.array([ax])

    for i in range(num_cols):
        for j in range(mat_displ['NN'].shape[1]):
            ax[i].plot(mat_displ['NLFEA'][step_l:step,j], np.abs(load_steps[step_l:step]), color = colors2[str(j)], marker = 'o', label = 'NLFEA, ' +labels[j])
            ax[i].plot(mat_displ['NN'][step_l:step,j], np.abs(load_steps[step_l:step]), color = colors2[str(j)], linestyle = '--', marker = 'x', label = 'NN, '+labels[j])
            if type == 'u':
                ax[i].set_xlabel('Displacements [mm]')
            elif type == 'eps':
                ax[i].set_xlabel('Strains [‰]')
    
    ax[num_cols-1].legend()
    ax[0].set_ylabel('Force [N/mm]')
    
    if save_path is not None: 
        filename = 'Load-Deformation_'+type+'_'+case_study+'.png'
        plt.savefig(os.path.join(save_path, filename))
        print(f'Saved figure {filename} at {save_path}')

    return 



def multi_plot_load_deform(load_steps, mat_displ, save_path, type, thresh):
    """
    function for plotting multiple case studies and multiple reinforcement ratios.
    """
    
    # 1 - Determine the last suitable load level to plot
    for key0 in mat_displ.keys():
        for key1 in mat_displ[key0].keys():
            idx_NLFEA = trim_vectors(mat_displ[key0][key1]['NLFEA'][:,0], threshold = thresh)
            idx_NN = trim_vectors(mat_displ[key0][key1]['NN'][:,0], threshold = thresh)
            idx = min(idx_NLFEA, idx_NN)
            mat_displ[key0][key1]['NN'] = mat_displ[key0][key1]['NN'][:idx,0]
            mat_displ[key0][key1]['NLFEA'] = mat_displ[key0][key1]['NLFEA'][:idx,0]
            load_steps[key0][key1] = load_steps[key0][key1][:idx]
            print(f'First and last load level for case {key0} and {key1} are {load_steps[key0][key1][0]} kN/m and {load_steps[key0][key1][-1]} kN/m')
    num_rows = 1
    num_cols = len(load_steps.keys())
    _, colors2, _ = get_colors_from_map({'0': 0, '1': 0, '2': 0})

    # 2 - set up the figure
    fig, ax = plt.subplots(num_rows, num_cols, figsize = [num_cols*8, 5])
    ax = np.atleast_1d(ax)
    if type == 'u':
        x_labels = {'2D-1': '$u_{z}$ [mm]', 
                    '2D-2': '$u_{y}$ [mm]', 
                    '2D-3': '$u_{y}$ [mm]', 
                    '2D-4': '$u_{x}$ [mm]', 
                    '2D-5': '$u_{x}$ [mm]'}
    elif type == 'eps':
        x_labels = {'2D-1': '$\gamma_{xy}$ [‰]',
                    '2D-2': '$\epsilon_{y}$ [‰]',
                    '2D-3': '$\epsilon_{y}$ [‰]',
                    '2D-4': '$\epsilon_{x}$ [‰]', 
                    '2D-5': '$\epsilon_{x}$ [‰]', 
                    '2D-8C': ['$\epsilon_{x}$ [‰]', '$\epsilon_{y}$ [‰]', '$\epsilon_{xy}$ [‰]']}
    y_labels = {'2D-1': '$n_{xy}$ [N/mm]', 
                '2D-2': '$n_{y}$ [N/mm]', 
                '2D-3': '-$n_{y}$ [N/mm]', 
                '2D-4': '$n_{x}$ [N/mm]', 
                '2D-5': '$-n_{x}$ [N/mm]',
                '2D-8C': ['$n_{x}$ [N/mm]', '$n_{y}$ [N/mm]', '$n_{xy}$ [N/mm]']}

    # 3 - plot parameters
    for i, key0 in enumerate(load_steps.keys()):
         for j, key1 in enumerate(load_steps[key0].keys()):
            ax[i].plot(mat_displ[key0][key1]['NLFEA'], np.abs(load_steps[key0][key1]), color = colors2[str(j)], marker = 'o', label = 'NLFEA, ' +key1)
            ax[i].plot(mat_displ[key0][key1]['NN'], np.abs(load_steps[key0][key1]), color = colors2[str(j)], marker = 'x', linestyle = '--', label = 'NN, ' +key1)
            ax[i].set_xlabel(x_labels[key0])
            ax[i].set_ylabel(y_labels[key0])
            ax[i].set_title(key0)
    
    ax[num_cols-1].legend()
    
    

    if save_path is not None: 
        filename = 'Load-Deformation-orth'+type+'_'+str(load_steps.keys())+'.png'
        plt.savefig(os.path.join(save_path, filename))
        print(f'Saved figure {filename} at {save_path}')

    return


def trim_vectors(disp, threshold):
    '''
    trims vectors when a jump occurs (to not plot absurdly large values)
    '''
    idx = 0

    for i in range(1, len(disp)):
        prev_val = disp[i-1]
        curr_val = disp[i]

        # avoid zero-division
        if prev_val == 0:
            continue

        if abs(curr_val - prev_val) > threshold:
                # print(f'Threshold exceeded at {i}')
                return i
    
    return len(disp)


def scatter_diagonal_plot(load_steps, mat_displ, save_path, type, thresh):
    # 1 - Determine the last suitable load level to plot
    for key0 in mat_displ.keys():
        for key1 in mat_displ[key0].keys():
            idx_NLFEA = trim_vectors(mat_displ[key0][key1]['NLFEA'][:,0], threshold = thresh)
            idx_NN = trim_vectors(mat_displ[key0][key1]['NN'][:,0], threshold = thresh)
            idx = min(idx_NLFEA, idx_NN)
            mat_displ[key0][key1]['NN'] = mat_displ[key0][key1]['NN'][:idx,0]
            mat_displ[key0][key1]['NLFEA'] = mat_displ[key0][key1]['NLFEA'][:idx,0]
            load_steps[key0][key1] = load_steps[key0][key1][:idx]
            print(f'First and last load level for case {key0} and {key1} are {load_steps[key0][key1][0]} kN/m and {load_steps[key0][key1][-1]} kN/m')
    num_rows = 1
    num_cols = len(load_steps.keys())
    _, colors2, _ = get_colors_from_map({'0': 0, '1': 0, '2': 0})

   # 2 - set up the figure
    fig, ax = plt.subplots(num_rows, num_cols, figsize = [num_cols*8, 5])
    ax = np.atleast_1d(ax)
    if type == 'u':
        x_labels = {'2D-1': '$u_{z}$ [mm]', 
                    '2D-2': '$u_{y}$ [mm]', 
                    '2D-3': '$u_{y}$ [mm]', 
                    '2D-4': '$u_{x}$ [mm]', 
                    '2D-5': '$u_{x}$ [mm]'}
    elif type == 'eps':
        x_labels = {'2D-1': '$\gamma_{xy,NLFEA}$ [‰]',
                    '2D-2': r'$\varepsilon_{y,NLFEA}$ [‰]',
                    '2D-3': r'$\varepsilon_{y,NLFEA}$ [‰]',
                    '2D-4': r'$\varepsilon_{x,NLFEA}$ [‰]', 
                    '2D-5': r'$\varepsilon_{x, NLFEA}$ [‰]'}
    
    y_labels = {'2D-1': '$\gamma_{xy,NN}$ [‰]',
                    '2D-2': r'$\varepsilon_{y,NN}$ [‰]',
                    '2D-3': r'$\varepsilon_{y,NN}$ [‰]',
                    '2D-4': r'$\varepsilon_{x,NN}$ [‰]', 
                    '2D-5': r'$\varepsilon_{x,NN}$ [‰]'}
    
    

    # 3 - plot parameters
    errors = {}
    for i, key0 in enumerate(load_steps.keys()):
        errors[key0] = {}
        for j, key1 in enumerate(load_steps[key0].keys()):
            ax[i].scatter(mat_displ[key0][key1]['NLFEA'], mat_displ[key0][key1]['NN'], color = colors2[str(j)], marker = 'o', label = key1)
            errors[key0][key1] = calculate_errors_diagonal(mat_displ, key0, key1)
            ax[i].set_xlabel(x_labels[key0])
            ax[i].set_ylabel(y_labels[key0])
            ax[i].set_title(key0)
        keys1 = list(mat_displ[key0].keys())
        ax[i].text(
                0.05, 0.95,                 # x,y position in axes coordinates (0–1)
                f"RMSE {keys1[0]}: {np.round(errors[key0][keys1[0]]['rmse'][0][0], 2)}\n"
                f"RMSE {keys1[1]}: {np.round(errors[key0][keys1[1]]['rmse'][0][0], 2)}\n"
                f"RMSE {keys1[2]}: {np.round(errors[key0][keys1[2]]['rmse'][0][0], 2)}\n"
                f"$R^2$ {keys1[0]}: {np.round(errors[key0][keys1[0]]['r_squared2'][0][0], 2)}\n"
                f"$R^2$ {keys1[1]}: {np.round(errors[key0][keys1[1]]['r_squared2'][0][0], 2)}\n"
                f"$R^2$ {keys1[2]}: {np.round(errors[key0][keys1[2]]['r_squared2'][0][0], 2)}",
                transform=ax[i].transAxes,     # makes coordinates relative to the axes
                fontsize=9,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="white",
                    ec="black",
                    alpha=0.8
                )
            )
        diagonal = np.arange(np.min(mat_displ[key0][key1]['NLFEA']), np.max(mat_displ[key0][keys1[0]]['NLFEA']))
        ax[i].plot(diagonal, diagonal, color = 'grey', linestyle = '--')

    ax[num_cols-1].legend()
    
    

    if save_path is not None: 
        filename = 'Load-deformation-diagonal'+type+'_'+str(load_steps.keys())+'.png'
        plt.savefig(os.path.join(save_path, filename))
        print(f'Saved figure {filename} at {save_path}')

    return


def diagonal_loadpath_plot(path_depl, save_path, type = 'eps', thresh = 10):
    '''
    creates diagonal plots for epsilon from given deployments
    '''

    if isinstance(path_depl, dict):
        # plotting multiple load-deformation paths
        load_steps = {}
        mat_displ = {}
        for key0 in path_depl.keys(): # case study
            load_steps[key0] = {}
            mat_displ[key0] = {}
            for key1 in path_depl[key0].keys(): # rho_i
                load_steps[key0][key1], mat_displ[key0][key1] = plot_load_paths('05_Deploying\\data_out\\'+path_depl[key0][key1], case_study = key0, 
                                                                                until_load_level = None, save_path = None, type = type)
        scatter_diagonal_plot(load_steps, mat_displ, save_path, type, thresh)

    else: 
        UserWarning('This function is not layed out to be used for single paths.')
    

def calculate_errors_diagonal(mat_displ, key0, key1):
    num_cols_plt = len(mat_displ[key0][key1]['NN'])
    predictions = mat_displ[key0][key1]['NN']
    Y = mat_displ[key0][key1]['NLFEA']


    ### Calculate errors
    r_squared2 = np.zeros((1,num_cols_plt))
    rse_max=np.zeros((1,num_cols_plt))
    n_5p, n_10p, rmse, aux_, aux__, rrmse = np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt))
    rrse_max, nrse_max, nrmse, log_max, mean_log_err = np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt))
    rse, nrse, log_err = np.zeros((Y.shape[0], num_cols_plt)), np.zeros((Y.shape[0], num_cols_plt)), np.zeros((Y.shape[0], num_cols_plt))


    i = 0
    Y_col = Y.reshape((1, -1))
    pred_col = predictions.reshape((1,-1))
    r_squared2[:,i] = np.corrcoef(Y_col, pred_col)[0, 1]**2
    rse[:,i] = np.sqrt((pred_col-Y_col)**2)
    rse_max[:,i] = np.max(rse[:,i])
    rmse[:,i] = np.sqrt(np.mean((pred_col - Y_col) ** 2))
    mean = np.mean(Y)
    aux_[:,i] = np.sqrt(np.mean((mean*np.ones(Y_col.shape) - Y_col) ** 2))
    if aux_[:,i].any() == 0:
        aux_[aux_[:,i] == 0] = 1
    rrmse[:,i] = np.divide(rmse[:,i],aux_[:,i])
    rrse_max[0,i] = np.max(np.divide(np.sqrt((pred_col-Y_col)**2), aux_[:,i]))


    # Calculate normalised RMSE
    q_95 = np.quantile(Y,0.95)
    q_5 = np.quantile(Y,0.05)
    aux__ = q_95-q_5
    if aux__ == 0:
        aux__ = 1
    nrse[:,i] = np.divide(np.sqrt((pred_col-Y_col)**2), aux__)*100
    nrmse[:,i] = np.divide(rmse[:,i], aux__)
    nrse_max[:,i] = np.max(nrse[:,i]/100)



    errors = {
        'rse': rse,
        'rse_max': rse_max,
        'rmse': rmse,
        'nrmse': nrmse,
        'nrse': nrse,
        'nrse_max': nrse_max,
        'rrmse': rrmse,
        'rrse_max': rrse_max,
        'r_squared2': r_squared2,
    }

    return errors


