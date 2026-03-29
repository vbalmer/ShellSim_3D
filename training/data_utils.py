# vb, 25.03.2026

import h5py
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import torch


SEED = 42

################################## Reading data ##################################


def read_h5_file(save_data_path, filename, n_every:int) -> tuple:
    name = filename
    path = save_data_path + '\\output_' + name + '.h5'
    with h5py.File(path, 'r') as f:
        data = f[name][::n_every,:]

    return data

def get_data(path_data:str, sobolev:bool) -> tuple:
    """
    Fetches data given in path
    
    Args:
        path_data       (str):  Location where data is stored
        sobolev         (bool): If True: include stiffness data in labels, otherwise leave out.

    Returns:
        features     (np.arr):  Epsilon values, shape: (ntot, 6)
        labels_concat(np.arr):  Stress and Stiffness values, shape: (ntot, 6+36) or (ntot, 6) depending on SOBOLEV   

    """

    features = read_h5_file(path_data, 'eps_g', n_every = 1)
    labels = read_h5_file(path_data, 'sig_g', n_every = 1)
    labels_diff = read_h5_file(path_data, 'D', n_every = 1)
    if sobolev: 
        labels_concat = np.concatenate((labels, labels_diff.reshape((-1, labels.shape[1]**2))), axis = 1)
    else: 
        labels_concat = labels

    print(f'Total amount of datapoints: {features.shape[0]/1e6:.2f}*1e6.')

    return features, labels_concat

def add_geom_data(path_data:str, features: np.array, GEOM_SIZE: int) -> np.array:
    """
    Adds variable geometrical data given in data_path to features 

    Args:
        path_data   (str)   : Location where geom data (and features) are stored
        features    (np.arr): Features without geom (just strains), shape: (ntot, 6)
        GEOM_SIZE   (int)   : Size of to be added variable geometrical input parameters

    Returns: 
        features    (np.arr): Extended by the geom values (ntot, 6+GEOM_SIZE)
    """
    if GEOM_SIZE == 0:
        return features
    
    else: 
        geom = read_h5_file(path_data, 't', n_every = 1)
        features = np.concatenate((features, geom), axis = 1)
        raise UserWarning('This has not yet been tested.')

        return features
    

################################## Split data ##################################

def split_data(features, labels, test_size = 0.1, eval_size = 0.2):
    """
    Splits data into train, eval and test set

    Args: 
        features (np.arr)   : All features, shape (ntot, 6+GEOM_SIZE)
        labels   (np.arr)   : All labels, shape (ntot, 6+36)

    Returns: 
        train_eval_test_data (mat): Dict containing np.arrays of train, eval and test data for X and y.

    """
    
    test_size_2 = eval_size/(1-test_size)

    X_aux, X_test, y_aux, y_test = train_test_split(features, labels, test_size = test_size, random_state = SEED)
    X_train, X_eval, y_train, y_eval = train_test_split(X_aux, y_aux, test_size= test_size_2, random_state = SEED)

    train_eval_test_data = {
        'X_train': X_train,
        'X_eval': X_eval,
        'X_test': X_test,
        'y_train': y_train,
        'y_eval': y_eval,
        'y_test': y_test,
    }

    return train_eval_test_data

def plot_split_data(data:dict, plot:bool):
    """
    Creates 6 plots for 6 subdatasets: 
    - train, test, eval
    - eps and sig
    """
    if plot:
        plot_3D_data(data['X_train'][:,:6], 'train_scatter_eps_g')
        plot_3D_data(data['X_eval'][:,:6], 'eval_scatter_eps_g')
        plot_3D_data(data['X_test'][:,:6], 'test_scatter_eps_g')
        plot_3D_data(data['y_train'][:,:6], 'train_scatter_sig_g')
        plot_3D_data(data['y_eval'][:,:6], 'eval_scatter_sig_g')
        plot_3D_data(data['y_test'][:,:6], 'test_scatter_sig_g')
  
    return


################################## Normalisation ##################################

def get_stats(data):
    """
    Get statistical parameters (like mu, sig, ...) for all partial datasets (train, eval, test)

    Args:
        data        (dict): datset split in train, eval and test (features and labels), i.e. 6 arrays in mat.
    
    Returns: 
        stats       (dict): statistical parameters of each sub-dataset
    """

    stats = {}
    for key in data.keys():
        stats['stats_'+key] = statistics(data[key])

    return stats

def get_normalised_data(data: dict, stats: dict, sobolev: bool) -> dict:
    """
    Get normalised dataset for all 6 subsets in data dict

    Args: 
        data        (dict): datset split in train, eval and test (features and labels), i.e. 6 arrays in mat.
        stats       (dict): statistics for normalisation. only train statistics used.
        sobolev     (bool): True if stiffness data is also included in labels.
    
    Returns: 
        normalised_data (dict): normalised dataset 
    """

    xdim = data['X_test'].shape[1]         # 6+GEOM_SIZE
    ydim = data['y_test'].shape[1]         # 6 or 6+36 depending on SOBOLEV
    if sobolev:
        norm_type_x = ['x-std']*xdim
        norm_type_y = ['y-std']*ydim
    else: 
        norm_type_x = ['x-std']*xdim
        norm_type_y = ['y-std']*6+['y_st-stitched']*36

    normalised_data = {}
    for key in data.keys():
        if 'X' in key:
            normalised_data[key+'_t'] = transform_data(data[key], stats, forward = True, type = norm_type_x)
        elif 'y' in key: 
            normalised_data[key+'_t'] = transform_data(data[key], stats, forward = True, type = norm_type_y)

    return normalised_data


def statistics(data:np.array):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    max = np.max(data, axis=0)
    min = np.min(data, axis=0)
    q_95 = np.percentile(data, 95, axis=0)
    q_5 = np.percentile(data, 5, axis=0)

    delta = 1e-8
    log_mean = np.mean(np.log(data+np.abs(min)+delta), axis = 0)
    log_std = np.std(np.log(data+np.abs(min)+delta), axis = 0)

    stats = {
        'mean': mean,
        'std': std,
        'log_mean': log_mean,
        'log_std': log_std,
        'max': max,
        'min': min,
        'q_5': q_5,
        'q_95': q_95
    }
    return stats

def transform_data(data:np.array, stats_:dict, forward: bool, type: list):
    """
    Returns standardised data set with transformed data for the forward case.
    Or returns original-scale data set transformed back from given transformed data.

    Args: 
        data            (dict)    :     still the individual data sets for sig, eps
        stats_          (dict)    :     contains statistical values that are constant for the transformation
        forward         (bool)    :     True: forward transform (to standardnormaldistr.), False: backward transform to original values
        type            (str-list):     if 'std': standard normal distr.
                                        if 'st-stitched': standard normal distr.; but for D entries: transformed according to sigma and eps - std
                                        
    Returns: 
        new_data       (np.array)  :        Uniformly distributed "new" data set, normalised
    """

    # Calculate new data

    data_stdnorm = np.zeros((data.shape))
    data_nonorm = np.zeros((data.shape))
    new_data = np.zeros((data.shape))
    new_data_ = np.zeros((data.shape))
    np_data = data.copy()
    np_data_ = data.copy()
    
    # for the stitched transformation of the D-matrix, need transformation of D based on D_coeff:
    if 'y-st-stitched' in type:
        D_coeff_sz = int(np.sqrt(type.count('y-st-stitched')))
        # print('D_coeff_sz:', D_coeff_sz)
        D_coeff = np.zeros((D_coeff_sz,D_coeff_sz))
        for j in range(D_coeff_sz):
            for k in range(D_coeff_sz):
                D_coeff[j,k] = stats_['stats_y_train']['std'][j]/stats_['stats_X_train']['std'][k]
        D_coeff_ = D_coeff.reshape((1, D_coeff_sz*D_coeff_sz))


    # Carry out the transformation
    if forward:
        for i in range(data.shape[1]):
            # Defining the correct statistical values (x or y depending on input data)
            if 'x' in type[i]: 
                stats = stats_['stats_X_train']
            elif 'y' in type[i]: 
                stats = stats_['stats_y_train']
            else: 
                raise RuntimeError('Please define an appropriate transformation type including the variable x or y to be transformed.')
            
            # Transforming the data
            if 'std' in type[i]:
                # 1 - Transformation to Standard Normal distribution
                if stats['std'][i] == 0:
                    stats['std'][i] = 1.0
                data_stdnorm[:,i] = (np_data[:,i]-stats['mean'][i]*np.ones(np_data.shape[0]))/(stats['std'][i]*np.ones(np_data.shape[0]))
                new_data[:,i] = data_stdnorm[:,i]

            elif 'st-stitched' in type[i]: 
                # 3 - Transformation of D, based on physically meaningful transformation of statistical variables of sigma and eps
                # assumes shape of y: sig+D (nxD_coeff_sz)+(nxD_coeff_sz*D_coeff_sz)
                np_data_[:,i] = np_data[:,i]
                data_stdnorm[:,i] = np.divide(1,D_coeff_[0,i-D_coeff_sz])*np_data_[:,i]
                new_data[:,i] = data_stdnorm[:,i]


    elif not forward:
        for i in range(data.shape[1]):
            # Defining the correct statistical values (x or y depending on input data)
            if 'x' in type[i]: 
                stats = stats_['stats_X_train']
            elif 'y' in type[i]: 
                stats = stats_['stats_y_train']
            else: 
                raise RuntimeError('Please define an appropriate transformation type including the variable x or y to be transformed.')
            
            # Transforming the data
            if 'std' in type[i]:
                # 1 - Redo Standard normal distribution
                if stats['std'][i] == 0:
                    stats['std'][i] = 1.0
                data_nonorm[:,i] = np_data[:,i]*stats['std'][i]*np.ones(np_data.shape[0])+stats['mean'][i]*np.ones(np_data.shape[0])
                new_data[:,i] = data_nonorm[:,i]
                
            elif 'st-stitched' in type[i]: 
                # 3 - Transformation of D, based on physically meaningful transformation of statistical variables of sigma and eps
                # assumes shape of y: sig+D (nxD_coeff_sz)+(nxD_coeff_sz*D_coeff_sz)
                data_nonorm[:,i] = D_coeff_[0,i-D_coeff_sz]*np_data[:,i]
                new_data_[:,i] = data_nonorm[:,i]
                new_data[:,i] = new_data_[:,i]

    return new_data

def plot_norm_data(data: dict, plot: bool):
    """
    Plot normalised training dataset for sigma.
    """
    if plot:
        plot_3D_data(data['X_train_t'][:,:6], 'train_norm_scatter_eps_g')
        plot_3D_data(data['y_train_t'][:,:6], 'train_norm_scatter_sig_g')
    return

def data_to_torch(data: dict) -> dict:
    """
    Convert numpy data to torch data
    """
    
    data_torch = {}

    for key in data.keys():
        data_torch[key+'t'] = torch.from_numpy(data[key])
        data_torch[key+'t'] = data_torch[key+'t'].type(torch.float32)


    return data_torch

################################## Visualisation & Plotting ##################################



def plot_3D_data(data, filename, n_every: int = 1):
    """
    visualise 3d scatters of different partial datasets

    Args:
        data    (np.arr): dataset, shape: (ntot, 6) (can be strains or stresses)
        filename   (str): Either "scatter_eps_g" or "scatter_sig_g"     
        n_every    (int): Plot only every "n_every"th value.
    Returns: 
        figure     (fig): Saved as png in specified folder (training\\plots)

    """
    fig = plt.figure(figsize=(14, 7))
    
    for i in range(2):
        x = data[:,i*3]
        y = data[:,i*3+1]
        z = data[:,i*3+2]
        # print(f'Plotting {len(x)/1e6}*1e6/{len(x)/(1e6)*n_every}*1e6 points')
        
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        ax.scatter(x, y, z, s=2, alpha=0.1)
        ax.set_title(f'$N$ = {len(x)/1e6:.2f}*1e6', y = 0.95)
        figure_formatting(ax, i, filename)

    t2 = time.perf_counter()
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "training\\plots\\" + filename + ".png"))
    print(f'Saved {filename} to training\\plots\\{filename}.png')
    # print(f'time saving figure: {(time.perf_counter()-t2)/60:.2f}min')

    return

def figure_formatting(ax, i, filename):
    if 'eps' in filename:
        if i == 0:
            ax.set_xlabel('eps_x')
            ax.set_ylabel('eps_y')
            ax.set_zlabel('gamma_xy')
        elif i == 1:
            ax.set_xlabel('chi_x')
            ax.set_ylabel('chi_y')
            ax.set_zlabel('chi_xy')

    elif 'sig' in filename: 
        if i == 0:
            ax.set_xlabel('n_x')
            ax.set_ylabel('n_y')
            ax.set_zlabel('n_xy')
        elif i == 1:
            ax.set_xlabel('m_x')
            ax.set_ylabel('m_y')
            ax.set_zlabel('m_xy')