import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from matplotlib.offsetbox import AnchoredText
import os
import wandb
import pickle
import torch
import glob
import math
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.transforms
from matplotlib.font_manager import FontProperties
import shutil
import re
from matplotlib.lines import Line2D


'''--------------------------------------LOADING DATA--------------------------------------------------'''

# Define torch dataset
class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
'''--------------------------------------LOADING DATA--------------------------------------------------'''


def read_data(path: str, id: str):
    '''
    Reads in the data from given path
    id      (str)           Identifyer of data to be read: either 'sig', 'eps' or 't'
    '''
    if 'add' in id:
        with open(os.path.join(path, 'newadd_data_'+id.replace("_add", "") +'.pkl'),'rb') as handle:
            new_data = pickle.load(handle)
    else: 
        with open(os.path.join(path, 'new_data_'+id +'.pkl'),'rb') as handle:
            new_data = pickle.load(handle)
    
    # transform data to numpy
    new_data = pd.DataFrame.from_dict(new_data)
    new_data_np = new_data.to_numpy()

    return new_data_np

def save_data(X_train:np.array, X_eval:np.array, X_test:np.array, y_train:np.array, y_eval:np.array, y_test:np.array, path:str):
    mat = {}
    mat['X_train'] = X_train
    mat['y_train'] = y_train
    mat['X_eval'] = X_eval
    mat['y_eval'] = y_eval
    mat['X_test'] = X_test
    mat['y_test'] = y_test

    with open(os.path.join(path, 'mat_data_np_TrainEvalTest.pkl'), 'wb') as fp:
        pickle.dump(mat, fp)
    return


def data_to_torch(X_train_t:np.array, y_train_t:np.array, X_eval_t:np.array, y_eval_t:np.array, 
                  X_test_t:np.array, y_test_t:np.array, 
                  path: str, sobolev: bool, batch_size: int, tag = False):
    '''
    Takes numpy arrays and creates torch dataloaders
    Saves relevant information to mat_data_TrainEvalTest

    '''

    # transform data to torch
    X_train_tt = torch.from_numpy(X_train_t)
    X_train_tt = X_train_tt.type(torch.float32)
    X_eval_tt = torch.from_numpy(X_eval_t)
    X_eval_tt = X_eval_tt.type(torch.float32)
    X_test_tt = torch.from_numpy(X_test_t)
    X_test_tt = X_test_tt.type(torch.float32)

    if sobolev:
        y_train_tt = torch.from_numpy(y_train_t)
        y_train_tt = y_train_tt.type(torch.float32)
        y_eval_tt = torch.from_numpy(y_eval_t)
        y_eval_tt = y_eval_tt.type(torch.float32)
        y_test_tt = torch.from_numpy(y_test_t)
        y_test_tt = y_test_tt.type(torch.float32)
    elif not sobolev:
        if not tag:
            y_train_tt = torch.from_numpy(y_train_t[:,0:8])
            y_train_tt = y_train_tt.type(torch.float32)
            y_eval_tt = torch.from_numpy(y_eval_t[:,0:8])
            y_eval_tt = y_eval_tt.type(torch.float32)
            y_test_tt = torch.from_numpy(y_test_t[:,0:8])
            y_test_tt = y_test_tt.type(torch.float32)
        else: 
            y_train_tt = torch.from_numpy(y_train_t)
            y_train_tt = y_train_tt.type(torch.float32)
            y_eval_tt = torch.from_numpy(y_eval_t)
            y_eval_tt = y_eval_tt.type(torch.float32)
            y_test_tt = torch.from_numpy(y_test_t)
            y_test_tt = y_test_tt.type(torch.float32)


    # Creating Datasets
    data_train_t = MyDataset(X_train_tt, y_train_tt)
    data_eval_t = MyDataset(X_eval_tt, y_eval_tt)
    data_test_t = MyDataset(X_test_tt, y_test_tt)

    # Transform to DataLoaders
    if batch_size is not None:
        raise Warning('note: this code is deprecated. batching takes place in simple_train function')
        train_loader = DataLoader(data_train_t, batch_size, shuffle=True) #, num_workers=16) 
        val_loader = DataLoader(data_eval_t, batch_size, shuffle=False) #, num_workers=16)
        test_loader = DataLoader(data_test_t, batch_size, shuffle=False) #, num_workers=16)
        print('Training set size:', len(list(train_loader)))
        print('Validation set size:', len(list(val_loader)))
        print('Test set size:', len(list(test_loader)))

    # Save data
    mat = {}
    mat['X_train_tt'] = X_train_tt
    mat['y_train_tt'] = y_train_tt
    mat['X_eval_tt'] = X_eval_tt
    mat['y_eval_tt'] = y_eval_tt
    mat['X_test_tt'] = X_test_tt
    mat['y_test_tt'] = y_test_tt
    if batch_size is not None:
        mat['test_loader'] = test_loader

    with open(os.path.join(path, 'mat_data_TrainEvalTest.pkl'), 'wb') as fp:
        pickle.dump(mat, fp)

    if batch_size is not None: 
        raise Warning('Note: this code is deprecated. batching takes place in simple_train function')
        loaders = {
            "train": train_loader, 
            "val": val_loader, 
            "test": test_loader
        }
    else: 
        loaders = None

    return loaders, mat

'''--------------------------------------HISTOGRAMS,STATISTICS--------------------------------------------------'''

def histogram(X:np.array, y:np.array, amt_data_points:int, nbins: int, name: str, path: str):
    '''
    Plots histograms of training and evaluation data set to check for consistency with previous steps of process.
    '''
    
    keys1 = ['eps_x', 'eps_y', 'eps_xy', 'chi_x', 'chi_y', 'chi_xy', 'gamma_xz', 'gamma_yz']
    keys2 = ['n_x', 'n_y', 'n_xy', 'm_x', 'm_y', 'm_xy', 'v_xz', 'v_yz']
    keys3 = np.array([['D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18'],
            ['D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28'],
            ['D31', 'D32', 'D33', 'D34', 'D35', 'D36', 'D37', 'D38'],
            ['D41', 'D42', 'D43', 'D44', 'D45', 'D46', 'D47', 'D48'],
            ['D51', 'D52', 'D53', 'D54', 'D55', 'D56', 'D57', 'D58'],
            ['D61', 'D62', 'D63', 'D64', 'D65', 'D66', 'D67', 'D68'],
            ['D71', 'D72', 'D73', 'D74', 'D75', 'D76', 'D77', 'D78'],
            ['D81', 'D82', 'D83', 'D84', 'D85', 'D86', 'D87', 'D88']])

    if name == 'sig': 
        keys = keys2
        plot_data = y
    elif name == 'eps':
        keys = keys1
        plot_data = X
    elif name == 't':
        keys = 't'
        plot_data = X
    elif name == 't2':
        keys = ['t1', 't2', 'nl', '']
        plot_data = X
    elif name == 'De':
        keys = keys3
        plot_data = y
    else: 
        raise "Error: No real name defined"

    if name == 'sig' or name == 'eps':
        fig2, axs2 = plt.subplots(2, 4, sharey=True, tight_layout=True)
        for i in range(3):
            axs2[0, i].set_xlabel(keys[i])
            axs2[1, i].set_xlabel(keys[i+3])
            axs2[0, i].hist(plot_data[:,i], bins=nbins)
            if plot_data.shape[1]>3:
                axs2[1, i].hist(plot_data[:,i+3], bins=nbins)        
        if plot_data.shape[1]>3:
            axs2[0, 3].set_xlabel(keys[6])
            axs2[1, 3].set_xlabel(keys[7])    
            axs2[0, 3].hist(plot_data[:,6], bins=nbins)
            axs2[1, 3].hist(plot_data[:,7], bins=nbins)
    elif name == 't' or name == 't2': 
        fig2, axs2 = plt.subplots(1, X.shape[1], sharey = True, tight_layout = True)
        for i in range(X.shape[1]):
            if X.shape[1] == 1:
                axs2.hist(plot_data[:,i], bins = nbins)
                axs2.set_xlabel(keys[i])
            else: 
                axs2[i].hist(plot_data[:,i], bins = nbins)
                axs2[i].set_xlabel(keys[i])
    elif name == 'De':
        fig2, axs2 = plt.subplots(plot_data.shape[1], plot_data.shape[2], figsize = (16, 16), sharey = True)
        for i in range(plot_data.shape[1]):
            for j in range(plot_data.shape[2]):
                axs2[i, j].hist(plot_data[:, i, j], bins = nbins)
                axs2[i, j].set_xlabel(keys3[i,j])


    if len(plot_data) == amt_data_points:
        fig2.suptitle('All Data, n = ' + str(len(plot_data)))
    elif len(plot_data) > 0.5*amt_data_points:
        fig2.suptitle('Training Data, n = ' + str(len(plot_data)))
    elif len(plot_data) < 0.2*amt_data_points:
        fig2.suptitle('Test Data, n = ' + str(len(plot_data)))
    else:
        fig2.suptitle('Validation Data, n = ' + str(len(plot_data)))


    if path is not None: 
        plt.savefig(os.path.join(path, 'hist_'+name))
        print('saved histogram ', name)

    return



def histogram_torch(data: DataLoader, amt_data_points:int, nbins: int, name: str):       
    '''
    Plots histograms of training and evaluation data set to check for consistency with previous steps of process.
    '''
    all_features = np.zeros((len(list(data)), 8))
    all_labels = np.zeros((len(list(data)), 8))

    for i, (features, labels) in enumerate(data):
        feat_eps = features[:,0:8]
        all_features[i,:] = feat_eps.numpy()
        all_labels[i,:] = labels.numpy()
    # print(all_features.shape)
    
    fig2, axs2 = plt.subplots(2, 4, sharey=True, tight_layout=True)
    keys1 = ['eps_x', 'eps_y', 'eps_xy', 'chi_x', 'chi_y', 'chi_xy', 'gamma_xz', 'gamma_yz']
    keys2 = ['n_x', 'n_y', 'n_xy', 'm_x', 'm_y', 'm_xy', 'v_xz', 'v_yz']

    if name == 'sig': 
        keys = keys2
        plot_data = all_labels
    elif name == 'eps':
        keys = keys1
        plot_data = all_features
    else: 
        raise "Error: No real name defined"

    for i in range(3):
        axs2[0, i].set_xlabel(keys[i])
        axs2[1, i].set_xlabel(keys[i+3])
        axs2[0, i].hist(plot_data[:,i], bins=nbins)
        axs2[1, i].hist(plot_data[:,i+3], bins=nbins)        
    axs2[0, 3].set_xlabel(keys[6])
    axs2[1, 3].set_xlabel(keys[7])    
    axs2[0, 3].hist(plot_data[:,6], bins=nbins)
    axs2[1, 3].hist(plot_data[:,7], bins=nbins)
    
    if len(data) == amt_data_points:
        fig2.suptitle('All Data, n = ' + str(len(data)))
    elif len(data) > 0.5*amt_data_points:
        fig2.suptitle('Training Data, n = ' + str(len(data)))
    elif len(data) < 0.2*amt_data_points:
        fig2.suptitle('Test Data, n = ' + str(len(data)))
    else:
        fig2.suptitle('Validation Data, n = ' + str(len(data)))

    plt.show()

    return plt


def statistics_pd(data:pd.DataFrame):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    max = data.max(axis=0)
    min = data.min(axis=0)
    q_5 = data.quantile(0.05, axis=0)
    q_95 = data.quantile(0.95, axis=0)
    stats = {
        'mean': mean,
        'std': std,
        'max': max,
        'min': min,
        'q_5': q_5,
        'q_95': q_95,
    }
    return stats

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

def find_D_linel(t, E = 33600, magic_factor = 1):
    '''
    determines the linear elastic stiffness matrix for a given thickness 
    nu and E are fixed to 0.2 and 33600 MPa respectively.
    used only for double-normalisation in transformation of stiffness matrix.
    yields lin.el. stiffness matrix in [MN, cm]

    t               (int)           thickness [mm]
    magic_factor    (int)           weighting factor, used to multiply times D_linel to improve ratio between normalised sigma and D
    '''
    nu = 0
    D_p = (E/(1-nu**2))*np.array([[1,  nu,  0],
                                [nu,  1,  0],
                                [0,   0,  (1-nu)/2]])
    D_s = 5/6*t*E/(2*(1+nu))*np.array([[1, 0],
                                       [0, 1]])
    D_const_linel = np.vstack((np.hstack((t*D_p, np.zeros((3,3)), np.zeros((3,2)))), 
                                np.hstack((np.zeros((3,3)), (t**3/12)*D_p, np.zeros((3,2)))), 
                                np.hstack((np.zeros((2,3)), np.zeros((2,3)), D_s))
                                ))
    D_const_linel_ = transf_units(D_const_linel.reshape((1,8,8)), 'D', forward=True,linel=True)

    D_const_linel_star = np.divide(D_const_linel_, magic_factor)

    return D_const_linel_star

def find_mask_D(magic_factor = 1):
    '''
    finds the weighting mask to improve normalisation of D based on magic factor.
    '''
    mask = (magic_factor-1)*np.array([[0, 1, 1, 0, 1, 1, 0, 0],
                                    [1, 0, 1, 1, 0, 1, 0, 0],
                                    [1, 1, 0, 1, 1, 0, 0, 0],
                                    [0, 1, 1, 0, 1, 1, 0, 0],
                                    [1, 0, 1, 1, 0, 1, 0, 0],
                                    [1, 1, 0, 1, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0]])+1

    return mask

def transform_data(data:np.array, stats_:dict, forward: bool, type: list, sc = False, dn = False, log_add = None):
    """
    Returns standardised data set with transformed data for the forward case.
    Or returns original-scale data set transformed back from given transformed data.

    Input: 
    data            (pd.DataFrame)      still the individual data sets for sig, eps
    stats_          (dict)              contains statistical values that are constant for the transformation
    forward         (bool)              True: forward transform (to standardnormaldistr.), 
                                        False: backward transform to original values
    type            (str-list)          if 'std': standard normal distr.
                                        if 'range': max-min
                                        if 'st-stitched': standard normal distr. 
                                        if 'log' / 'lg-stitched': log normalised
                                        but for D entries: transformed according to sigma and eps - std
    sc              (bool)              True: scaling of data (+10) for potentially better training
    dn              (bool)              True: stiffness data is normalised w.r.t. linear elastic constant of D for bringing the values closer to each other
    log_add         (dict)              containing values 'add_data_eps' (n x 8), 'add_data_sig' (n x 8)
                                        add_data_eps and add_data_sig are non-normalised (i.e. MN, cm) values corresponding to the D being transformed.
                                        this dict is ONLY used for the forward and inverse transform of D in log-case
                                        
    Output: 
    new_data       (np.array)           Uniformly distributed "new" data set
    """

    # Calculate new data

    data_stdnorm = np.zeros((data.shape))
    data_nonorm = np.zeros((data.shape))
    data_range = np.zeros((data.shape))
    data_star = np.zeros((data.shape))
    data_lognorm = np.zeros((data.shape))
    data_norange = np.zeros((data.shape))
    new_data = np.zeros((data.shape))
    new_data_ = np.zeros((data.shape))
    np_data = data.copy()
    np_data_ = data.copy()
    delta = 1e-8
    
    # for the stitched transformation of the D-matrix, need transformation of D based on D_coeff:
    if 'y-st-stitched' or 'lg-stitched' in type:
        D_coeff_sz = int(np.sqrt(type.count('y-st-stitched')))
        # print('D_coeff_sz:', D_coeff_sz)
        D_coeff = np.zeros((D_coeff_sz,D_coeff_sz))
        for j in range(D_coeff_sz):
            for k in range(D_coeff_sz):
                D_coeff[j,k] = stats_['stats_y_train']['std'][j]/stats_['stats_X_train']['std'][k]
        D_coeff_ = D_coeff.reshape((1, D_coeff_sz*D_coeff_sz))
        # for double normalisation
        # D_linel = find_D_linel(t=200, magic_factor=1, only_mask = True)
        D_weights = find_mask_D(magic_factor=0.2)
        D_weights_ = D_weights.reshape((1,64))

    if log_add is not None:
        eps_current = np.repeat(log_add['add_data_eps'][:, np.newaxis, :], 8, axis=1).reshape((-1, 64))             # expected shape: n, 64
        sig_current = np.repeat(log_add['add_data_sig'][:, :, np.newaxis], 8, axis=2).reshape((-1,64))              # expected shape: n, 64
        eps_min = np.abs(np.repeat(stats_['stats_X_train']['min'][np.newaxis, :8], 8, axis=0)).reshape((-1,64))      # expected shape: , 64
        sig_min = np.abs(np.repeat(stats_['stats_y_train']['min'][:8, np.newaxis], 8, axis=1)).reshape((-1,64))      # expected shape: , 64
        eps_log_std_ = np.repeat(stats_['stats_X_train']['log_std'][np.newaxis, :8], 8, axis=0).reshape((-1,64))    # expected shape: , 64
        sig_log_std_ = np.repeat(stats_['stats_y_train']['log_std'][:8, np.newaxis], 8, axis=1).reshape((-1,64))    # expected shape: , 64
        eps_log_std, sig_log_std = np.zeros_like(eps_log_std_), np.zeros_like(sig_log_std_)
        eps_log_std[eps_log_std == 0] = 1
        sig_log_std[sig_log_std == 0] = 1

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
                if sc: 
                    new_data[:,i] = data_stdnorm[:,i]+10
            elif 'range' in type[i]:
                # 2 - Transformation maximin / range
                if stats['max'][i]-stats['min'][i] == 0:
                    stats['max'][i] = 1
                data_range[:,i] = (np_data[:,i]-stats['min'][i]*np.ones(np_data.shape[0]))/((stats['max'][i]-stats['min'][i])*np.ones(np_data.shape[0]))
                new_data[:,i] = data_range[:,i]*1
            elif 'st-stitched' in type[i]: 
                # 3 - Transformation of D, based on physically meaningful transformation of statistical variables of sigma and eps
                # assumes shape of y: sig+D (nxD_coeff_sz)+(nxD_coeff_sz*D_coeff_sz)
                if dn and i >= D_coeff_sz:
                    #for double-normalisation, only in D, only cond
                    np_data_[:,i] = np.divide(np_data[:,i], D_weights_[0,i-D_coeff_sz])
                else:
                    np_data_[:,i] = np_data[:,i]
                data_stdnorm[:,i] = np.divide(1,D_coeff_[0,i-D_coeff_sz])*np_data_[:,i]
                new_data[:,i] = data_stdnorm[:,i]
            elif 'log' in type[i]:
                # 4 Log-transformation for sig, eps (incl. standardisation)
                if stats['log_std'][i] == 0:
                    stats['log_std'][i] = 1.0
                data_star[:,i] = np.log(np_data[:,i]+np.abs(stats['min'][i])+delta)
                data_lognorm[:,i] = (data_star[:,i]-stats['log_mean'][i])/(stats['log_std'][i])
                new_data[:,i] = data_lognorm[:,i]
            elif 'lg-stitched' in type[i]:
                # Log-transformation for D, based on physically meaningful transformation.
                # assumes shape of y: sig+D (nxD_coeff_sz)+(nxD_coeff_sz*D_coeff_sz)
                data_lognorm[:,i] = ((eps_current[:,i-D_coeff_sz] + eps_min[0,i-D_coeff_sz]+delta)* eps_log_std[0,i-D_coeff_sz])  /  ((sig_current[:,i-D_coeff_sz] + sig_min[0,i-D_coeff_sz]+delta)* sig_log_std[0,i-D_coeff_sz]) *np_data[:,i] 
                new_data[:,i] = data_lognorm[:,i]


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
                if sc: 
                    np_data[:,i] = np_data[:,i]-10
                if stats['std'][i] == 0:
                    stats['std'][i] = 1.0
                data_nonorm[:,i] = np_data[:,i]*stats['std'][i]*np.ones(np_data.shape[0])+stats['mean'][i]*np.ones(np_data.shape[0])
                new_data[:,i] = data_nonorm[:,i]
                
            elif 'range' in type[i]:
                # 2 - Redo maximin / range
                if stats['max'][i]-stats['min'][i] == 0:
                    stats['max'][i] = 1
                data_norange[:,i] = (np_data[:,i]/1)*(stats['max'][i]-stats['min'][i]) + stats['min'][i]*np.ones(np_data.shape[0])
                new_data[:,i] = data_norange[:,i]
            elif 'st-stitched' in type[i]: 
                # 3 - Transformation of D, based on physically meaningful transformation of statistical variables of sigma and eps
                # assumes shape of y: sig+D (nxD_coeff_sz)+(nxD_coeff_sz*D_coeff_sz)
                data_nonorm[:,i] = D_coeff_[0,i-D_coeff_sz]*np_data[:,i]
                new_data_[:,i] = data_nonorm[:,i]
                if dn and i >= D_coeff_sz: 
                    new_data[:,i] = D_weights_[0,i-D_coeff_sz]*new_data_[:,i]
                else: 
                    new_data[:,i] = new_data_[:,i]
            elif 'log' in type[i]:
                # 4 Log-transformation for sig, eps (incl. standardisation)
                if stats['log_std'][i] == 0:
                    stats['log_std'][i] = 1.0
                data_star[:,i] = np_data[:,i]*stats['log_std'][i]+stats['log_mean'][i]
                data_nonorm[:,i] = np.exp(data_star[:,i])-np.abs(stats['min'][i])-delta
            elif 'lg-stitched' in type[i]:
                # Log-transformation for D, based on physically meaningful transformation.
                # assumes shape of y: sig+D (nxD_coeff_sz)+(nxD_coeff_sz*D_coeff_sz)
                data_lognorm[:,i] = ((sig_current[:,i-D_coeff_sz] + sig_min[i-D_coeff_sz]+delta)* sig_log_std[i-D_coeff_sz])  /  ((eps_current[:,i-D_coeff_sz] + eps_min[i-D_coeff_sz]+delta)* eps_log_std[i-D_coeff_sz]) *np_data[:,i]  
                new_data[:,i] = data_lognorm[:,i]


    return new_data


'''-------------------------------------- INITIALISE WANDB --------------------------------------------------'''

def init_wandb(inp, inp1, inp2, project_name):
    if inp1 == None and inp2 == None:
        run = wandb.init(
            project = project_name, 
            config = inp
        )
    else:
        run = wandb.init(
            project = project_name, 
            config = {
                "inp": inp,
                "inp1": inp1,
                "inp2": inp2,
            },
        )
        # TODO
        # idea: put inp.update(with the info in inp1... make sure not to overwrite it)
        print("CAUTION. The inp1, inp2 File logging is not implemented for sweeping")
    return run



def transf_units(vec:np.array, id:str, forward:bool, linel = True):
    '''
    Transforms the units for input from simulation to training and back
    vec:        (np.array)          Vector to be transformed
    id:         (str)               Identifier 'sig', 'D' or 'eps-t' depending on the desired transformation of vec
                                    Expected shapes: sig: (n,8), eps-t: (n,9), D: (n, 8,8)
    forward:    (bool)              If true: Transformation is forward (i.e. from N, mm to MN, cm)
                                    If false: Transformation is backward (i.e. from MN, cm to N, mm)
    linel       (bool)              If true: Sets values of D_mb = 0 in stiffness matrix
                                    If false: Also transforms the units of D_mb according to the correct transformation
    '''

    if id == 'sig':
        sig_t = np.zeros_like(vec)
        if forward:
            sig_t[:, 0:3] = vec[:, 0:3]*(10**(-6))*(10**(1))
            sig_t[:, 3:6] = vec[:, 3:6]*(10**(-6))*1
            sig_t[:, 6:8] = vec[:, 6:8]*(10**(-6))*(10**(1))
            out_vec = sig_t
        else: 
            sig_t[:, 0:3] = vec[:, 0:3]*(10**(6))*(10**(-1))
            sig_t[:, 3:6] = vec[:, 3:6]*(10**(6))*1
            sig_t[:, 6:8] = vec[:, 6:8]*(10**(6))*(10**(-1))
            out_vec = sig_t
    elif id == 'sig-t':
        sig_t = np.zeros_like(vec)
        if forward:
            sig_t[:, 0:3] = vec[:, 0:3]*(10**(-6))*(10**(1))
            sig_t[:, 3:6] = vec[:, 3:6]*(10**(-6))*1
            sig_t[:, 6:8] = vec[:, 6:8]*(10**(-6))*(10**(1))
            sig_t[:,8:] = vec[:,8:]
            out_vec = sig_t
        else: 
            sig_t[:, 0:3] = vec[:, 0:3]*(10**(6))*(10**(-1))
            sig_t[:, 3:6] = vec[:, 3:6]*(10**(6))*1
            sig_t[:, 6:8] = vec[:, 6:8]*(10**(6))*(10**(-1))
            sig_t[:,8:] = vec[:,8:]
            out_vec = sig_t

    elif id == 'eps-t':
        eps_t = np.zeros_like(vec)
        if forward:
            eps_t[:, 0:3] = vec[:, 0:3]*1
            eps_t[:, 3:6] = vec[:, 3:6]*(10**1)
            eps_t[:, 6:8] = vec[:, 6:8]*1
            eps_t[:,8:] = vec[:,8:]
            out_vec = eps_t
        else:
            eps_t[:, 0:3] = vec[:, 0:3]*1
            eps_t[:, 3:6] = vec[:, 3:6]*(10**(-1))
            eps_t[:, 6:8] = vec[:, 6:8]*1
            eps_t[:,8:] = vec[:,8:]
            out_vec = eps_t
    elif id == 'eps': 
        eps_t = np.zeros_like(vec)
        if forward: 
            eps_t[:, 0:3] = vec[:, 0:3]*1
            eps_t[:, 3:6] = vec[:, 3:6]*(10**1)
            eps_t[:, 6:8] = vec[:, 6:8]*1
            out_vec = eps_t
        else: 
            eps_t[:, 0:3] = vec[:, 0:3]*1
            eps_t[:, 3:6] = vec[:, 3:6]*(10**(-1))
            eps_t[:, 6:8] = vec[:, 6:8]*1
            out_vec = eps_t
    
    elif id == 'D':

        D_t = np.zeros_like(vec)
        if linel:
            if forward:
                D_t[:, 0:3, 0:3] = vec[:, 0:3, 0:3]*(10**(-6))*(10**(1))
                D_t[:,3:6, 3:6] = vec[:, 3:6, 3:6]*(10**(-6))*(10**(-1))
                D_t[:, 6:8, 6:8] = vec[:, 6:8, 6:8]*(10**(-6))*(10**(1))
                out_vec = D_t
            else:
                D_t[:, 0:3, 0:3] = vec[:, 0:3, 0:3]*(10**(6))*(10**(-1))
                D_t[:,3:6, 3:6] = vec[:, 3:6, 3:6]*(10**(6))*(10**(1))
                D_t[:, 6:8, 6:8] = vec[:, 6:8, 6:8]*(10**(6))*(10**(-1))
                out_vec = D_t
        else:
            if forward:
                D_t[:, 0:3, 0:3] = vec[:, 0:3, 0:3]*(10**(-6))*(10**(1))
                D_t[:,3:6, 3:6] = vec[:, 3:6, 3:6]*(10**(-6))*(10**(-1))
                D_t[:, 6:8, 6:8] = vec[:, 6:8, 6:8]*(10**(-6))*(10**(1))
                D_t[:,0:3, 3:6] = vec[:, 0:3, 3:6]*(10**(-6))
                D_t[:,3:6, 0:3] = vec[:, 3:6, 0:3]*(10**(-6))
                out_vec = D_t
            else: 
                D_t[:, 0:3, 0:3] = vec[:, 0:3, 0:3]*(10**(6))*(10**(-1))
                D_t[:,3:6, 3:6] = vec[:, 3:6, 3:6]*(10**(6))*(10**(1))
                D_t[:, 6:8, 6:8] = vec[:, 6:8, 6:8]*(10**(6))*(10**(-1))
                D_t[:,0:3, 3:6] = vec[:, 0:3, 3:6]*(10**(6))
                D_t[:,3:6, 0:3] = vec[:, 3:6, 0:3]*(10**(6))
                out_vec = D_t

    return out_vec






'''--------------------------------------POSTPROCESSING--------------------------------------------------'''

def calculate_errors(Y, predictions, stats, transf, id = 'sig'):

    if id == 'sig':
        num_cols = 9
        num_cols_plt = 8
    elif id == 'De': 
        num_cols = 72
        num_cols_plt = 17
    elif id == 'De-NLRC': 
        num_cols = 72
        num_cols_plt = 38
        if stats['stats_y_train']['mean'].shape[0] == 38:
            num_cols = 38
            num_cols_plt = 38
    elif id == 'eps':
        num_cols = 8
        num_cols_plt = 8

    ### Transform the units of the statistics which relate back to the train set to the units desired in the diagonal plot
    if transf == 't' or transf == 't-inv':
        mean_train_ = 0*np.ones((1,num_cols))
        q_5_train_ = -1.645*np.ones((1,num_cols))
        q_95_train_ = 1.645*np.ones((1,num_cols))
    elif transf == 'o' or transf == 'o-inv':
        # the statistics are already in the units MN, cm
        mean_train_ = stats['stats_y_train']['mean'][0:num_cols].reshape((1,num_cols))
        q_5_train_ = stats['stats_y_train']['q_5'][0:num_cols].reshape((1,num_cols))
        q_95_train_ = stats['stats_y_train']['q_95'][0:num_cols].reshape((1,num_cols))
    elif transf == 'u' or transf == 'u-inv':
        stats_new = {}
        stats_new_2 = {}
        if id == 'sig' or id == 'eps':
            for key, value in stats['stats_y_train'].items():
                stats_new[key] = transf_units(value[0:num_cols].reshape(1,num_cols), id, forward=False).reshape(num_cols,)
            mean_train_ = stats_new['mean'][0:num_cols].reshape((1,num_cols))
            q_5_train_ = stats_new['q_5'][0:num_cols].reshape((1,num_cols))
            q_95_train_ = stats_new['q_95'][0:num_cols].reshape((1,num_cols))         
        elif id == 'De' or id == 'De-NLRC':
            if id == 'De':
                LINEL = True
            elif id == 'De-NLRC':
                LINEL = False
            if num_cols == 38:
                for key, value in stats['stats_y_train'].items():
                    value_ = np.zeros((1,8,8))
                    value_[:,:6,:6] = value[0:num_cols-2].reshape((1,6,6))
                    value_[:,6,6] = value[36].reshape((1,))
                    value_[:,7,7] = value[37].reshape((1,))
                    stats_new[key] = transf_units(value_, 'D', forward=False, linel = LINEL).reshape(64,)
                mean_train_ = stats_new['mean'].reshape((1,64))
                q_5_train_ = stats_new['q_5'].reshape((1,64))
                q_95_train_ = stats_new['q_95'].reshape((1,64))
            else: 
                for key, value in stats['stats_y_train'].items():
                    # stats['stats_y_train'][key] = transf_units(value[0:8].reshape(1,8), 'sig', forward=False, linel = LINEL).reshape(8,)
                    stats_new_2[key] = transf_units(value[0:8].reshape(1,8), 'sig', forward=False, linel = LINEL).reshape(8,)
                    stats_new[key] = transf_units(value[8:num_cols].reshape((1,8,8)), 'D', forward=False, linel = LINEL).reshape(64,)
                mean_train_ = np.hstack((stats_new_2['mean'][0:8], stats_new['mean'])).reshape((1,num_cols))
                q_5_train_ = np.hstack((stats_new_2['q_5'][0:8], stats_new['q_5'])).reshape((1,num_cols))
                q_95_train_ = np.hstack((stats_new_2['q_95'][0:8], stats_new['q_95'])).reshape((1,num_cols))


    if id == 'De':
        # Kick out irrelevant data (that should be zero) and reshape matrix to (num_rows x 12) format, for lin.el. calculation
        mean_train_De = mean_train_[:,8:72].reshape((-1, 8, 8))
        q_5_train_De = q_5_train_[:,8:72].reshape((-1, 8, 8))
        q_95_train_De = q_95_train_[:,8:72].reshape((-1, 8, 8))

        mean_train = np.hstack((mean_train_De[:,0:2, 0:2].reshape((-1,4)), mean_train_De[:,2, 2].reshape((-1,1)),
                                mean_train_De[:,3:5, 3:5].reshape((-1,4)), mean_train_De[:,5, 5].reshape((-1,1)),
                                mean_train_De[:,0:2, 3:5].reshape((-1,4)), mean_train_De[:,2, 5].reshape((-1,1)),
                                mean_train_De[:,6, 6].reshape((-1,1)), mean_train_De[:,7, 7].reshape((-1,1))))
        q_5_train = np.hstack((q_5_train_De[:,0:2, 0:2].reshape((-1,4)), q_5_train_De[:,2, 2].reshape((-1,1)),
                                q_5_train_De[:,3:5, 3:5].reshape((-1,4)), q_5_train_De[:,5, 5].reshape((-1,1)),
                                q_5_train_De[:,0:2, 3:5].reshape((-1,4)), q_5_train_De[:,2, 5].reshape((-1,1)),
                                q_5_train_De[:,6, 6].reshape((-1,1)), q_5_train_De[:,7, 7].reshape((-1,1))))
        q_95_train = np.hstack((q_95_train_De[:,0:2, 0:2].reshape((-1,4)), q_95_train_De[:,2, 2].reshape((-1,1)),
                                q_95_train_De[:,3:5, 3:5].reshape((-1,4)), q_95_train_De[:,5, 5].reshape((-1,1)),
                                q_95_train_De[:,0:2, 3:5].reshape((-1,4)), q_95_train_De[:,2, 5].reshape((-1,1)),
                                q_95_train_De[:,6, 6].reshape((-1,1)), q_95_train_De[:,7, 7].reshape((-1,1))))
    elif id == 'De-NLRC':
        # for nonlinear version of De (i.e. Dmb is not zero)
        if num_cols == 38:
            # if directly predicting D with the network
            if transf == 't' or transf == 'o': 
                mean_train = mean_train_
                q_5_train = q_5_train_
                q_95_train = q_95_train_
            else: 
                mean_train_De = mean_train_[:,0:64].reshape((-1, 8, 8))
                q_5_train_De = q_5_train_[:,0:64].reshape((-1, 8, 8))
                q_95_train_De = q_95_train_[:,0:64].reshape((-1, 8, 8))
                mean_train = np.concatenate((mean_train_De[:,:6,:6].reshape((-1, 36)), mean_train_De[:,6,6].reshape((-1, 1)), mean_train_De[:,7,7].reshape((-1, 1))), axis = 1)
                q_5_train = np.concatenate((q_5_train_De[:,:6,:6].reshape((-1,36)), q_5_train_De[:,6,6].reshape((-1,1)), q_5_train_De[:,7,7].reshape((-1,1))), axis = 1)
                q_95_train = np.concatenate((q_95_train_De[:,:6,:6].reshape((-1,36)), q_95_train_De[:,6,6].reshape((-1,1)), q_95_train_De[:,7,7].reshape((-1,1))), axis = 1)
        else:
            # if predicting sigma with network and D with derivatives
            mean_train_De = mean_train_[:,8:72].reshape((-1, 8, 8))
            q_5_train_De = q_5_train_[:,8:72].reshape((-1, 8, 8))
            q_95_train_De = q_95_train_[:,8:72].reshape((-1, 8, 8))
            mean_train = np.concatenate((mean_train_De[:,:6,:6].reshape((-1, 36)), mean_train_De[:,6,6].reshape((-1, 1)), mean_train_De[:,7,7].reshape((-1, 1))), axis = 1)
            q_5_train = np.concatenate((q_5_train_De[:,:6,:6].reshape((-1,36)), q_5_train_De[:,6,6].reshape((-1,1)), q_5_train_De[:,7,7].reshape((-1,1))), axis = 1)
            q_95_train = np.concatenate((q_95_train_De[:,:6,:6].reshape((-1,36)), q_95_train_De[:,6,6].reshape((-1,1)), q_95_train_De[:,7,7].reshape((-1,1))), axis = 1)
    
    elif id == 'sig' or id == 'eps':
        mean_train = mean_train_
        q_5_train = q_5_train_
        q_95_train = q_95_train_



    ### Calculate errors
    r_squared2 = np.zeros((1,num_cols_plt))
    rse_max=np.zeros((1,num_cols_plt))
    n_5p, n_10p, rmse, aux_, aux__, rrmse = np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt))
    rrse_max, nrse_max, nrmse, log_max, mean_log_err = np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt))
    rse, nrse, log_err = np.zeros((Y.shape[0], num_cols_plt)), np.zeros((Y.shape[0], num_cols_plt)), np.zeros((Y.shape[0], num_cols_plt))
    # Delta_max_i, Delta_max_max, Delta_max_mean = np.zeros((Y.shape[0], num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt))
    for i in range(num_cols_plt):
        Y_col = Y[:, i].flatten()
        pred_col = predictions[:,i].flatten()
        r_squared2[:,i] = np.corrcoef(Y_col, pred_col)[0, 1]**2
        rse[:,i] = np.sqrt((pred_col-Y_col)**2)
        rse_max[:,i] = np.max(rse[:,i])
        rmse[:,i] = np.sqrt(np.mean((pred_col - Y_col) ** 2))
        aux_[:,i] = np.sqrt(np.mean((mean_train[:,i]*np.ones(Y_col.shape) - Y_col) ** 2))
        if aux_[:,i].any() == 0:
            aux_[aux_[:,i] == 0] = 1
        rrmse[:,i] = np.divide(rmse[:,i],aux_[:,i])
        rrse_max[0,i] = np.max(np.divide(np.sqrt((pred_col-Y_col)**2), aux_[:,i]))

        # Delta_max_i[:,i] = np.abs(Y_col - pred_col)/np.maximum(np.abs(Y_col), np.abs(pred_col))
        # Delta_max_max[0,i] = np.max(Delta_max_i[:,i])
        # Delta_max_mean[0,i] = np.mean(Delta_max_i[:,i])

        # Calculate normalised RMSE
        aux__[:,i] = q_95_train[:,i]-q_5_train[:,i]
        if aux__[:,i].any() == 0:
            aux__[aux__[:,i] == 0] = 1
        nrse[:,i] = np.divide(np.sqrt((pred_col-Y_col)**2), aux__[:,i])*100
        nrmse[:,i] = np.divide(rmse[:,i], aux__[:,i])
        nrse_max[:,i] = np.max(nrse[:,i]/100)

        # Calculate log error (not used at the moment)
        log_err[:,i] = np.log(pred_col+1)-np.log(Y_col+1)
        mean_log_err[:,i] = np.mean(log_err[:,i])
        log_max[:,i] = np.max(log_err[:,i])


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


def multiple_diagonal_plots_wrapper(save_path: str, plot_data:dict, stats:dict, color='nrse'):
    raise Warning('this version of the MoE is outdated.')
    for model in ['exp1', 'exp2', 'exp3', 'MoE']:
        # normalised plots
        multiple_diagonal_plots(save_path+'\MoE\\'+model, plot_data[model]['all_test_labels_t'], plot_data[model]['all_predictions_t'], 't', stats, color)
        # original scale plots
        multiple_diagonal_plots(save_path+'\MoE\\'+model, plot_data[model]['all_test_labels'], plot_data[model]['all_predictions'], 'o', stats, color)
        # simulation scale plots
        plot_data_label_u = transf_units(plot_data[model]['all_test_labels'], 'sig', forward = False)
        plot_data_pred_u = transf_units(plot_data[model]['all_predictions'], 'sig', forward = False)
        multiple_diagonal_plots(save_path+'\MoE\\'+model, plot_data_label_u, plot_data_pred_u, 'u', stats, 'rse')
    return



def multiple_diagonal_plots(save_path: str, Y: np.array, predictions: np.array, transf:str, stats:dict, color='nrse', 
                            Y_train = None, pred_train = None, xlim = None, ylim = None):
    ''''
    save_path       (str)           path where images are saved
    Y               (np.array)      Ground truth
    predictions     (np.array)      Predictions
    transf          (str)           't': transformed(normalised), 'o': original scale [MN, cm], 'u': units for simulation [N,mm]
    stats           (dict)          data statistics for normalising / relativising the RMSE
    color           (str)           scatter color: if 'nrse': only one colour bar across all plots. If 'rse': 3 separate colorbars for n, m, v
    y_train         (np.array)      Values of training data (if want to test on training data)
    pred_train      (np.array)      Values of training data (if want to test on training data)
    xlim, ylim      (np.array)      Defines the xlim and ylim for all 8 stress values. For checking the smaller-range predictions. 
                                    If these values are chosen, then the error calculation will also be carried out just for the data in this range. 
                                    nRMSE will still be calcualated w.r.t. entire training data range. 
    '''

    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": 12,
        })

    if "inv" in transf:
        errors = calculate_errors(Y, predictions, stats, transf, id = 'eps')
    elif (xlim and ylim) is not None:
        mask_x = (Y[:,:8]>=xlim[0]) & (Y[:,:8]<=xlim[1])
        mask_y = (predictions[:,:8]>=ylim[0]) & (predictions[:,:8]<=ylim[1])
        mask = mask_x & mask_y
        valid_rows = mask.all(axis=1)
        if np.sum(valid_rows) == 0:
            raise Warning('No points found for this region. Please increase range.')
        errors = calculate_errors(Y[valid_rows,:8], predictions[valid_rows,:8], stats, transf, id = 'sig')
    else: 
        errors = calculate_errors(Y, predictions, stats, transf, id = 'sig')



    # Plot figure
    fig, axa = plt.subplots(3, 3, figsize=[15.5, 12], dpi=100)
    fig.subplots_adjust(wspace=0.5)
    index_mask = np.array([[0,1,2],
                           [3,4,5],
                           [6,7,7]])
    num_rows = Y.shape[0]
    mask_labels = np.zeros((Y.shape[0], 8))

    if transf == 'o':
        plotname = np.array([['$n_x$', '$n_y$', '$n_{xy}$'],
                            ['$m_x$', '$m_y$', '$m_{xy}$'],
                            ['$v_x$', '$v_y$', '$v_y$']])
        plotname_p = np.array([[r'$\tilde{n}_{x}$', r'$\tilde{n}_{y}$', r'$\tilde{n}_{xy}$'],
                            [r'$\tilde{m}_x$', r'$\tilde{m}_y$', r'$\tilde{m}_{xy}$'],
                            [r'$\tilde{v}_x$', r'$\tilde{v}_y$', r'$\tilde{v}_y$']])
        units = np.array([[r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$'],
                          [r'$\rm [MNcm/cm]$', r'$\rm [MNcm/cm]$', r'$\rm [MNcm/cm]$'],  
                          [r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$']])
        
    elif transf == 'u':
        plotname = np.array([['$n_x$', '$n_y$', '$n_{xy}$'],
                            ['$m_x$', '$m_y$', '$m_{xy}$'],
                            ['$v_x$', '$v_y$', '$v_y$']])
        plotname_p = np.array([[r'$\tilde{n}_{x}$', r'$\tilde{n}_{y}$', r'$\tilde{n}_{xy}$'],
                            [r'$\tilde{m}_x$', r'$\tilde{m}_y$', r'$\tilde{m}_{xy}$'],
                            [r'$\tilde{v}_x$', r'$\tilde{v}_y$', r'$\tilde{v}_y$']])
        units = np.array([[r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$'],
                          [r'$\rm [Nmm/mm]$', r'$\rm [Nmm/mm]$', r'$\rm [Nmm/mm]$'],  
                          [r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$']])

    elif transf =='t':
        plotname = np.array([[r'$n_{x,norm}$', r'$n_{y,norm}$', r'$n_{xy,norm}$'],
                        [r'$m_{x,norm}$', r'$m_{y,norm}$', r'$m_{xy,norm}$'],
                        [r'$v_{x,norm}$', r'$v_{y,norm}$', r'$v_{y,norm}$']])
        plotname_p = np.array([[r'$\tilde{n}_{x,norm}$', r'$\tilde{n}_{y,norm}$', r'$\tilde{n}_{xy,norm}$'],
                        [r'$\tilde{m}_{x,norm}$', r'$\tilde{m}_{y,norm}$', r'$\tilde{m}_{xy,norm}$'],
                        [r'$\tilde{v}_{x,norm}$', r'$\tilde{v}_{y,norm}$', r'$\tilde{v}_{y,norm}$']])
        units = np.array([['$[-]$', '$[-]$', '$[-]$'],
                          ['$[-]$', '$[-]$', '$[-]$'],  
                          ['$[-]$', '$[-]$', '$[-]$']])
        
    elif transf == 't-inv':
        plotname = np.array([[r'$\varepsilon_{x,norm}$', r'$\varepsilon_{y,norm}$', r'$\varepsilon_{xy,norm}$'],
                        [r'$\chi_{x,norm}$', r'$\chi_{y,norm}$', r'$\chi_{xy,norm}$'],
                        [r'$\gamma_{x,norm}$', r'$\gamma_{y,norm}$', r'$t$']])
        plotname_p = np.array([[r'$\tilde{\varepsilon}_{x,norm}$', r'$\tilde{\varepsilon}_{y,norm}$', r'$\tilde{\varepsilon}_{xy,norm}$'],
                        [r'$\tilde{\chi}_{x,norm}$', r'$\tilde{\chi}_{y,norm}$', r'$\tilde{\chi}_{xy,norm}$'],
                        [r'$\tilde{\gamma}_{x,norm}$', r'$\tilde{\gamma}_{y,norm}$', r'$\tilde{t}$']])
        units = np.array([['$[-]$', '$[-]$', '$[-]$'],
                          ['$[-]$', '$[-]$', '$[-]$'],  
                          ['$[-]$', '$[-]$', '$[-]$']])
    
    elif transf == 'u-inv':
        plotname = np.array([[r'$\varepsilon_{x}$', r'$\varepsilon_{y}$', r'$\varepsilon_{xy}$'],
                        [r'$\chi_{x}$', r'$\chi_{y}$', r'$\chi_{xy}$'],
                        [r'$\gamma_{x}$', r'$\gamma_{y}$', r'$t$']])
        plotname_p = np.array([[r'$\tilde{\varepsilon}_{x}$', r'$\tilde{\varepsilon}_{y}$', r'$\tilde{\varepsilon}_{xy}$'],
                        [r'$\tilde{\chi}_{x}$', r'$\tilde{\chi}_{y}$', r'$\tilde{\chi}_{xy}$'],
                        [r'$\tilde{\gamma}_{x}$', r'$\tilde{\gamma}_{y}$', r'$\tilde{t}$']])
        units = np.array([[r'$\rm [-]$', r'$\rm [-]$', r'$\rm [-]$'],
                          [r'$\rm [1/mm]$', r'$\rm [1/mm]$', r'$\rm [1/mm]$'],  
                          [r'$\rm [-]$', r'$\rm [-]$', r'$\rm [-]$']])
        
    elif transf == 'o-inv': 
        plotname = np.array([[r'$\varepsilon_{x}$', r'$\varepsilon_{y}$', r'$\varepsilon_{xy}$'],
                        [r'$\chi_{x}$', r'$\chi_{y}$', r'$\chi_{xy}$'],
                        [r'$\gamma_{x}$', r'$\gamma_{y}$', r'$t$']])
        plotname_p = np.array([[r'$\tilde{\varepsilon}_{x}$', r'$\tilde{\varepsilon}_{y}$', r'$\tilde{\varepsilon}_{xy}$'],
                        [r'$\tilde{\chi}_{x}$', r'$\tilde{\chi}_{y}$', r'$\tilde{\chi}_{xy}$'],
                        [r'$\tilde{\gamma}_{x}$', r'$\tilde{\gamma}_{y}$', r'$\tilde{t}$']])
        units = np.array([[r'$\rm [-]$', r'$\rm [-]$', r'$\rm [-]$'],
                          [r'$\rm [1/cm]$', r'$\rm [1/cm]$', r'$\rm [1/cm]$'],  
                          [r'$\rm [-]$', r'$\rm [-]$', r'$\rm [-]$']])


    # find max, min for colorbars
    norms = [mcolors.Normalize(vmin=np.min(errors[color][:, index_mask[i, :]]),
                           vmax=np.max(errors[color][:, index_mask[i, :]]))
                            for i in range(3)]
    scatters = []


    for i in range(3):
        for j in range(3):
            if i ==2 and j==2:
                    axa[i,j].set_title(' ')
            else: 
                if (xlim and ylim) is not None:
                    scatter = axa[i,j].scatter(Y[valid_rows,index_mask[i,j]], predictions[valid_rows,index_mask[i,j]], marker = 'o', s = 20, 
                                c = errors[color][:, index_mask[i,j]], cmap = 'plasma', linestyle='None', alpha = 0.4, 
                                norm = norms[i])
                    scatters.append(scatter)
                else:
                    scatter = axa[i,j].scatter(Y[:,index_mask[i,j]], predictions[:,index_mask[i,j]], marker = 'o', s = 20, 
                                c = errors[color][:, index_mask[i,j]], cmap = 'plasma', linestyle='None', alpha = 0.4, 
                                norm = norms[i])
                    scatters.append(scatter)
                if Y_train is not None:
                    scatter2 = axa[i,j].scatter(Y_train[:,index_mask[i,j]], pred_train[:,index_mask[i,j]], marker = 'o', 
                                  c = errors[color][:, index_mask[i,j]], cmap = 'viridis', fillstyle = 'none', 
                                  s = 8, linestyle='None', alpha = 0.4, norms = norms[i])
                axa[i,j].set_ylabel(plotname_p[i,j]+' '+ units[i,j])
                axa[i,j].set_xlabel(plotname[i,j]+' '+ units[i,j])
                
                if (xlim and ylim) is not None:
                    axa[i,j].set_xlim(xlim[0][index_mask[i,j]], xlim[1][index_mask[i,j]])
                    axa[i,j].set_ylim(ylim[0][index_mask[i,j]], ylim[1][index_mask[i,j]])
                else:
                    if Y_train is not None:
                        axa[i,j].set_xlim([np.min([np.min(Y_train[:,index_mask[i,j]]), np.min(pred_train[:,index_mask[i,j]])]), np.max([np.max(Y_train[:,index_mask[i,j]]), np.max(pred_train[:,index_mask[i,j]])])])
                        axa[i,j].set_ylim([np.min([np.min(Y_train[:,index_mask[i,j]]), np.min(pred_train[:,index_mask[i,j]])]), np.max([np.max(Y_train[:,index_mask[i,j]]), np.max(pred_train[:,index_mask[i,j]])])])
                    else:
                        axa[i,j].set_xlim([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])])
                        axa[i,j].set_ylim([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])])
                
                axa[i,j].grid(True, which='major', color='#666666', linestyle='-')
                at = AnchoredText('$R^2 = ' + np.array2string(errors['r_squared2'][:,index_mask[i,j]][0], precision=2) + '$ \n' +
                                  '$RMSE = ' + np.array2string(errors['rmse'][:,index_mask[i,j]][0], precision=2) + '$ \n' +
                                  '$rRMSE = ' + np.array2string(errors['rrmse'][:,index_mask[i,j]][0]*100, precision=0) + '\% $ \n' +
                                  '$||rRSE||_{\infty} = ' + np.array2string(errors['rrse_max'][:,index_mask[i,j]][0]*100, precision=0) + '\% $ \n'+
                                  '$nRMSE = ' + np.array2string(errors['nrmse'][:,index_mask[i,j]][0]*100, precision=0) + '\% $  \n'+
                                  '$||nRSE||_{\infty} = ' + np.array2string(errors['nrse_max'][:,index_mask[i,j]][0]*100, precision=0) + '\% $',
                                #    '$MALE = ' + np.array2string(mean_log_err[:,index_mask[i,j]][0], precision=2) +' $ \n'+
                                #   '$||ALE||_{\infty} = ' + np.array2string(log_max[:,index_mask[i,j]][0], precision=2)+ ' $',
                                #  '$||\Delta_{max}||_{\infty} = ' + np.array2string(Delta_max_max[0,index_mask[i,j]]*100, precision=0) + '\% $ \n' + 
                                #  '$\Delta_{max, avg} = ' + np.array2string(Delta_max_mean[0,index_mask[i,j]]*100, precision=0) + '\% $',
                                prop=dict(size=10), frameon=True,loc='upper left')
                at.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
                axa[i,j].add_artist(at)
                if Y_train is not None:
                    axa[i,j].plot([np.min([np.min(Y_train[:,index_mask[i,j]]), np.min(pred_train[:,index_mask[i,j]])]), np.max([np.max(Y_train[:,index_mask[i,j]]), np.max(pred_train[:,index_mask[i,j]])])], [np.min([np.min(Y_train[:,index_mask[i,j]]), np.min(pred_train[:,index_mask[i,j]])]), np.max([np.max(Y_train[:,index_mask[i,j]]), np.max(pred_train[:,index_mask[i,j]])])],
                                  color='white', linestyle='--', linewidth = 1)
                else:
                    axa[i,j].plot([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])], [np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])],
                                  color='white', linestyle='--', linewidth = 1)
                for l in range(Y.shape[0]):
                    if np.all(mask_labels[l]):
                        axa[i,j].text(Y[l,index_mask[i,j]], predictions[l, index_mask[i,j]], str(l), fontsize=12, color='red', ha='center', va='bottom')
            
    axa[-1, -1].axis('off')
    
    at_ = AnchoredText('avg $rRMSE = ' + np.array2string(np.mean(errors['rrmse'][0,:])*100, precision=0) + '\% $ \n'+
                       'avg $||rRSE||_{\infty}= '+ np.array2string(np.mean(errors['rrse_max'][0,:])*100, precision=0) + '\% $\n'+
                       'avg $nRMSE = ' + np.array2string(np.mean(errors['nrmse'][0,:])*100, precision=0) + '\% $ \n'+
                       'avg $||nRSE||_{\infty}= '+ np.array2string(np.mean(errors['nrse_max'][0,:])*100, precision=0) + '\% $\n' +
                       ('N = '+ str(np.sum(valid_rows)) if (xlim and ylim) is not None else 'N = ' +str(Y.shape[0])), 
                       prop=dict(size=10), frameon=True, loc='center')
    at_.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
    axa[-1, -1].add_artist(at_)

    for i in range(3):
        if color == 'rse': 
            if i == 0 or i == 2:
                if 'inv' in transf:
                    name = 'RSE \: [-]'
                else: 
                    name = 'RSE \: [N/mm]'
            if i == 1:
                if 'inv' in transf: 
                    name = 'RSE \: [1/mm]'
                else: 
                    name = 'RSE \: [Nmm/mm]'
        elif color == 'nrse':
           name = 'nRSE \: [\%]'
        cbar = fig.colorbar(scatters[i*3], ax=axa[i,:], orientation='vertical', label=f'Row {i+1} $'+name+'$')
        cbar.set_label('$'+name+'$')


    axa = plt.gca()
    axa.set_aspect('equal', 'box')
    axa.axis('square')


    # Save figure
    # plt.tight_layout()
    if save_path is not None:
        if transf == 't' or transf == 't-inv':
            filename = os.path.join(save_path, 'diagonal_match_'+'transformed.png')
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            wandb.log({"45°-plot, t": wandb.Image(filename)})
            if "inv" not in transf: 
                print('saved sig-t-plots')
            else: 
                print('saved eps-t-plots')
        elif transf == 'o' or transf == 'o-inv':
            filename = os.path.join(save_path, 'diagonal_match_'+'original.png')
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            wandb.log({"45°-plot, og": wandb.Image(filename)})
            if "inv" not in transf:
                print('saved sig-o-plots')
            else: 
                print('saved eps-o-plots')
        elif transf == 'u' or transf == 'u-inv':
            if (xlim and ylim) is not None:
                filename = os.path.join(save_path, 'diagonal_match_'+'original_units_newlim.png')
            else: 
                filename = os.path.join(save_path, 'diagonal_match_'+'original_units.png')
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            wandb.log({"45°-plot, og_u": wandb.Image(filename)})
            if "inv" not in transf and (xlim and ylim) is None:
                print('saved sig-u-plots')
            elif (xlim and ylim) is not None:
                print('saved sig-u-plots with adjusted limits')
            else: 
                print('saved eps-u-plots')
    # plt.show()
    # plt.close()

    return


def multiple_diagonal_plots_range(save_path: str, Y: np.array, predictions: np.array, transf:str, stats:dict, color='nrse',  Y_train = None, pred_train = None):

    return


def multiple_diagonal_plots_D(save_path: str, Y_inp: np.array, predictions_inp: np.array, transf:str, stats: dict, color:str,
                              xlim = None, ylim = None):
    '''
    displays all variables of D_m, D_mb, D_bm and D_b (6x6) + Ds (2x1) --> for nonlinear RC
    '''

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": 12,
        })
    

    plotname = np.array([['$D_{m,11}$', '$D_{m,12}$', '$D_{m,13}$', '$D_{mb,11}$', '$D_{mb,12}$', '$D_{mb,13}$'],
                        ['$D_{m,21}$', '$D_{m,22}$', '$D_{m,23}$', '$D_{mb,21}$', '$D_{mb,22}$', '$D_{mb,23}$'],
                        ['$D_{m,31}$', '$D_{m,32}$', '$D_{m,33}$', '$D_{mb,31}$', '$D_{mb,32}$', '$D_{mb,33}$'],
                        ['$D_{bm,11}$', '$D_{bm,12}$', '$D_{bm,13}$', '$D_{b,11}$', '$D_{b,12}$', '$D_{b,13}$'],
                        ['$D_{bm,21}$', '$D_{bm,22}$', '$D_{bm,23}$', '$D_{b,21}$', '$D_{b,22}$', '$D_{b,23}$'],
                        ['$D_{bm,31}$', '$D_{bm,32}$', '$D_{bm,33}$', '$D_{b,31}$', '$D_{b,32}$', '$D_{b,33}$'],
                        ['$D_{s,11}$', '$D_{s,22}$', '$D_{s,22}$', '$D_{s,22}$', '$D_{s,22}$', '$D_{s,22}$']
                        ])
    plotname_p = plotname

    if transf == 'o':
        units = np.array([[r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MN]$'],
                          [r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MN]$'],
                          [r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MN]$'],
                          [r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$'],
                          [r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$'],
                          [r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$'],
                          [r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$']
                          ])
    elif transf =='u':
        units = np.array([[r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$'],
                          [r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$'],
                          [r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$'],
                          [r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$'],
                          [r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$'],
                          [r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$'],
                          [r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$']
                          ])

    index_mask = np.array([[ 0,  1,  2,  3,  4,  5],
                           [ 6,  7,  8,  9, 10, 11],
                           [12, 13, 14, 15, 16, 17],
                           [18, 19, 20, 21, 22, 23],
                           [24, 25, 26, 27, 28, 29],
                           [30, 31, 32, 33, 34, 35],  
                           [36, 37, 37, 37, 37, 37]
                           ])
    
    
    
    Y = np.concatenate((Y_inp[:,:6, :6].reshape((-1,36)), Y_inp[:,6,6].reshape((-1,1)), Y_inp[:,7,7].reshape((-1,1))), axis = 1)
    predictions = np.concatenate((predictions_inp[:,:6,:6].reshape((-1,36)), predictions_inp[:,6,6].reshape((-1,1)), predictions_inp[:,7,7].reshape((-1,1))), axis = 1)
    if (xlim and ylim) is not None: 
        # convert lims to correct format:
        lims_min_flat = []
        lims_max_flat = [] 
        
        for i in range(6):
            for j in range(6):
                if i < 3 and j < 3:  # D_m
                    idx = 0 if i == j else 1
                elif i >= 3 and j >= 3:  # D_b
                    idx = 4 if i == j else 5
                else:  # D_mb
                    idx = 2 if abs(i-j)==3 else 3

                lims_min_flat.append(xlim[0][idx])
                lims_max_flat.append(xlim[1][idx])
        lims_min_flat.append(xlim[0][6])
        lims_min_flat.append(xlim[0][6])
        lims_max_flat.append(xlim[1][6])
        lims_max_flat.append(xlim[1][6])

        lims_all = [lims_min_flat, lims_max_flat]

        mask_x = (Y[:,:]>=lims_all[0]) & (Y[:,:]<=lims_all[1])
        mask_y = (predictions[:,:]>=lims_all[0]) & (predictions[:,:]<=lims_all[1])
        mask = mask_x & mask_y
        valid_rows = mask.all(axis=1)
        if np.sum(valid_rows) == 0:
            mask[0, :] = True
            valid_rows = mask.all(axis=1)
            print('There are no points in the adjusted ranges, masking differently to avoid error.')
            # raise Warning('No points found for this region. Please increase range.')
        errors = calculate_errors(Y[valid_rows,:], predictions[valid_rows,:], stats, transf, id = 'De-NLRC')

    else:
        errors = calculate_errors(Y, predictions, stats, transf, id = 'De-NLRC')

    
    block_positions = [(0, 0), (0, 3), (3, 0), (3, 3)]
    norms = []

    for bx, by in block_positions:
        block_errors = np.concatenate([
            errors[color][:, index_mask[b0, b1]].flatten()
            for b0 in range(bx, bx + 3) for b1 in range(by, by + 3)
            ])
        norms.append(mcolors.Normalize(vmin=np.min(block_errors), vmax=np.max(block_errors)))

    # Last row special case
    last_row_errors = np.concatenate([errors[color][:, index_mask[6, j]].flatten() for j in range(3)])
    vmin, vmax = np.min(last_row_errors), np.max(last_row_errors)
    norm_last_row = mcolors.Normalize(vmin=vmin, vmax=vmax)
    norms.extend([norm_last_row] * 3)


    exp_rmse = np.floor(np.log10(errors['rmse']+1)).astype(int)
    base_rmse = errors['rmse']/(10**exp_rmse)
    scatters = []
    

    fig, axa = plt.subplots(7, 6, figsize=[35, 30], dpi=100)
    fig.subplots_adjust(wspace=0.5)


    for i in range(7):
        for j in range(6):
            if i == 6 and j > 1:
                axa[i,j].set_title('  ')
            else: 
                if (xlim and ylim) is not None:
                    scatter = axa[i,j].scatter(Y[valid_rows,index_mask[i,j]], predictions[valid_rows,index_mask[i,j]], marker = 'o', s = 20, 
                                c = errors[color][:, index_mask[i,j]], cmap = 'plasma', linestyle='None', alpha = 1, 
                                norm = norms[i // 3 * 2 + j // 3])
                    scatters.append(scatter)
                else:
                    scatter = axa[i,j].scatter(Y[:,index_mask[i,j]], predictions[:,index_mask[i,j]], marker = 'o', s = 20, 
                                            c = errors[color][:, index_mask[i,j]], cmap = 'plasma', linestyle='None', alpha = 1, 
                                            norm = norms[i // 3 * 2 + j // 3])
                    scatters.append(scatter)
                # axa[i,j].plot(Y[:,index_mask[i,j]], predictions[:,index_mask[i,j]], marker = 'o', ms = 5, linestyle='None')
                axa[i,j].set_ylabel(plotname_p[i,j]+' '+ units[i,j])
                axa[i,j].set_xlabel(plotname[i,j]+' '+ units[i,j])
                if (xlim and ylim) is not None:
                    axa[i,j].set_xlim(lims_all[0][index_mask[i,j]], lims_all[1][index_mask[i,j]])
                    axa[i,j].set_ylim(lims_all[0][index_mask[i,j]], lims_all[1][index_mask[i,j]])
                else:
                    axa[i,j].set_xlim([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])])
                    axa[i,j].set_ylim([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])])
                axa[i,j].grid(True, which='major', color='#666666', linestyle='-')
                at = AnchoredText('$R^2 = ' + np.array2string(errors['r_squared2'][:,index_mask[i,j]][0], precision=3) + '$ \n' +
                                    '$RMSE = ' + np.array2string(base_rmse[:,index_mask[i,j]][0], precision=2)+ '\\times 10^{'+ np.array2string(exp_rmse[:,index_mask[i,j]][0], precision=0) +'}$ \n' +
                                    '$rRMSE = ' + np.array2string(errors['rrmse'][:,index_mask[i,j]][0]*100, precision=0) + '\% $ \n'+
                                    # '$||rRSE||_{\infty} = ' + np.array2string(errors['rrse_max'][:,index_mask[i,j]][0]*100, precision=0) + '\% $ \n'+
                                    '$nRMSE = ' + np.array2string(errors['nrmse'][:,index_mask[i,j]][0]*100, precision=0) + '\% $',
                                    # '$||nRSE||_{\infty} = ' + np.array2string(errors['nrse_max'][:,index_mask[i,j]][0]*100, precision=0) + '\% $',
                                prop=dict(size=10), frameon=True, loc='lower right')
                at.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
                axa[i,j].add_artist(at)
                axa[i,j].plot([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])], [np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])], 
                                color='grey', linestyle='--', linewidth = 1)
                axa[i,j].set_aspect('equal', 'box')
    
    axa[-1, -1].axis('off')
    axa[-1, -2].axis('off')
    axa[-1, -3].axis('off')
    axa[-1, -4].axis('off')


    at_ = AnchoredText('avg $rRMSE = ' + np.array2string(np.mean(errors['rrmse'][0,:])*100, precision=0) + '\% $ \n'+
                       'avg $||rRSE||_{\infty}= '+ np.array2string(np.mean(errors['rrse_max'][0,:])*100, precision=0) + '\% $\n'+
                       'avg $nRMSE = ' + np.array2string(np.mean(errors['nrmse'][0,:])*100, precision=0) + '\% $ \n'+
                       'avg $||nRSE||_{\infty}= '+ np.array2string(np.mean(errors['nrse_max'][0,:])*100, precision=0) + '\% $\n' +
                       ('N = '+ str(np.sum(valid_rows)) if (xlim and ylim) is not None else 'N = ' +str(Y.shape[0])),
                        prop=dict(size=10), frameon=True, loc='center')
    at_.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
    axa[-1, -1].add_artist(at_)
    
    # Define 3x3 blocks for colorbars
    colorbar_labels = {"rse": ["RSE [N/mm]", "RSE [N]", "RSE [N]", "RSE [Nmm]"], "nrse":  ["nRSE [%]", "nRSE [%]", "nRSE [%]", "nRSE [%]"]}
    colorbar_label = colorbar_labels.get(color, "Error Scale")

    for norm, (bx, by), label in zip(norms, block_positions, colorbar_label):
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axa[bx:bx+3, by:by+3], orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label(label, fontsize=12)

    # Last row special case
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norms[-1])
    sm.set_array([])
    cbar_last_row = fig.colorbar(sm, ax=axa[6, :3], orientation='vertical', fraction=0.02, pad=0.04)
    cbar_last_row.set_label("RSE [N/mm]", fontsize=12)

    axa = plt.gca()
    axa.axis('square')


    # Save figure
    if save_path is not None and transf == 'o':
        filename = os.path.join(save_path, 'diagonal_match_'+'D_nonzero_o.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        wandb.log({"45°-plot, D_o": wandb.Image(filename)})
        print('saved D-o-plot')
    if save_path is not None and transf == 'u':
        if (xlim and ylim) is not None:
            filename = os.path.join(save_path, 'diagonal_match_'+'D_nonzero_u_newlim.png')
        else:
            filename = os.path.join(save_path, 'diagonal_match_'+'D_nonzero_u.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        wandb.log({"45°-plot, D_u": wandb.Image(filename)})
        if (xlim and ylim) is not None: 
            print('saved D-u-plot with adjusted limits')
        else:
            print('saved D-u-plot')
    # plt.show()
    plt.close()


def multiple_diagonal_plots_Dnz(save_path: str, Y_inp: np.array, predictions_inp: np.array, transf:str, stats: dict, color:str):
    '''
    displays all variables which are not zero in a linear elastic stiffness matrix 
    --> use in case of glass or lin.el. material training / testing
    '''

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": 12,
        })


    # Kick out irrelevant data (that should be zero) and reshape matrix to (num.rows x 12) format
    mat_comp_m = {
        "Simulation": np.hstack(((Y_inp[:, 0:2, 0:2]).reshape((Y_inp.shape[0],4)), Y_inp[:,2,2].reshape((Y_inp.shape[0],1)))),
        "Prediction": np.hstack(((predictions_inp[:,0:2, 0:2]).reshape((Y_inp.shape[0],4)), predictions_inp[:,2,2].reshape((Y_inp.shape[0],1))))
        }
    mat_comp_b = {
        "Simulation": np.hstack(((Y_inp[:,3:5, 3:5]).reshape((Y_inp.shape[0],4)), Y_inp[:,5,5].reshape((Y_inp.shape[0],1)))),
        "Prediction": np.hstack(((predictions_inp[:,3:5, 3:5]).reshape((Y_inp.shape[0],4)), predictions_inp[:,5,5].reshape((Y_inp.shape[0],1))))
        }
    
    mat_comp_mb = {
        "Simulation": np.hstack(((Y_inp[:,0:2, 3:5]).reshape((Y_inp.shape[0],4)), Y_inp[:,2,5].reshape((Y_inp.shape[0],1)))),
        "Prediction": np.hstack(((predictions_inp[:,0:2, 3:5]).reshape((Y_inp.shape[0],4)), predictions_inp[:,2,5].reshape((Y_inp.shape[0],1))))
    }

    mat_comp_s = {
        "Simulation": np.hstack((Y_inp[:,6,6], Y_inp[:,7,7])).reshape((Y_inp.shape[0],2)),
        "Prediction": np.hstack((predictions_inp[:,6,6], predictions_inp[:,7,7])).reshape((Y_inp.shape[0],2))
        }
    
    Y = np.hstack((mat_comp_m['Simulation'], mat_comp_b['Simulation'], mat_comp_mb['Simulation'], mat_comp_s['Simulation']))
    predictions = np.hstack((mat_comp_m['Prediction'], mat_comp_b['Prediction'], mat_comp_mb['Prediction'], mat_comp_s['Prediction']))

    errors = calculate_errors(Y, predictions, stats, transf, id = 'De')
    exp_rmse = np.floor(np.log10(errors['rmse']+1)).astype(int)
    base_rmse = errors['rmse']/(10**exp_rmse)
    
    # Plot figure
    fig, axa = plt.subplots(4, 5, figsize=[20, 12], dpi=100)
    fig.subplots_adjust(wspace=0.5)
    
    num_rows = Y.shape[0]

    plotname = np.array([['$D_{m,11}$', '$D_{m,12}$', '$D_{m,21}$', '$D_{m,22}$', '$D_{m,33}$'],
                        ['$D_{b,11}$', '$D_{b,12}$', '$D_{b,21}$', '$D_{b,22}$', '$D_{b,33}$'],
                        ['$D_{mb,11}$', '$D_{mb,12}$', '$D_{mb,21}$', '$D_{mb,22}$', '$D_{mb,33}$'],
                        ['$D_{s,11}$', '$D_{s,22}$', '$D_{s,22}$', '$D_{s,22}$', '$D_{s,22}$']])
    plotname_p = np.array([[r'$\tilde{D}_{m,11}$', r'$\tilde{D}_{m,12}$', r'$\tilde{D}_{m,21}$', r'$\tilde{D}_{m,22}$', r'$\tilde{D}_{m,33}$'],
                        [r'$\tilde{D}_{b,11}$', r'$\tilde{D}_{b,12}$', r'$\tilde{D}_{b,21}$', r'$\tilde{D}_{b,22}$', r'$\tilde{D}_{b,33}$'],
                        [r'$\tilde{D}_{mb,11}$', r'$\tilde{D}_{mb,12}$', r'$\tilde{D}_{mb,21}$', r'$\tilde{D}_{mb,22}$', r'$\tilde{D}_{mb,33}$'],
                        [r'$\tilde{D}_{s,11}$', r'$\tilde{D}_{s,22}$', r'$\tilde{D}_{s,22}$', r'$\tilde{D}_{s,22}$', r'$\tilde{D}_{s,22}$']])
    
    if transf == 'o':
        units = np.array([[r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$'],
                            [r'$\rm [MNcm]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$'],
                            [r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MN]$'],
                            [r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$']])
    elif transf =='u':
        units = np.array([[r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$'],
                            [r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$'],
                            [r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$'],
                            [r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$']])
        
    index_mask = np.array([[0,  1,  2,  3,  4],
                           [5,  6,  7,  8,  9],
                           [10, 11, 12, 13, 14],
                           [15, 16, 16, 16, 16]])


    # find max, min for colorbars
    norms = [mcolors.Normalize(vmin=np.min(errors[color][:, index_mask[i, :]]),
                           vmax=np.max(errors[color][:, index_mask[i, :]]))
                            for i in range(4)]
    scatters = []


    for i in range(4):
        for j in range(5):
            if i == 3 and j > 1:
                axa[i,j].set_title('  ')
            else:
                scatter = axa[i,j].scatter(Y[:,index_mask[i,j]], predictions[:,index_mask[i,j]], marker = 'o', s = 20, 
                                            c = errors[color][:, index_mask[i,j]], cmap = 'plasma', linestyle='None', alpha = 1, 
                                            norm = norms[i])
                scatters.append(scatter)
                # axa[i,j].plot(Y[:,index_mask[i,j]], predictions[:,index_mask[i,j]], marker = 'o', ms = 5, linestyle='None')
                axa[i,j].set_ylabel(plotname_p[i,j]+' '+ units[i,j])
                axa[i,j].set_xlabel(plotname[i,j]+' '+ units[i,j])
                axa[i,j].set_xlim([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])])
                axa[i,j].set_ylim([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])])
                axa[i,j].grid(True, which='major', color='#666666', linestyle='-')
                at = AnchoredText('$R^2 = ' + np.array2string(errors['r_squared2'][:,index_mask[i,j]][0], precision=3) + '$ \n' +
                                  '$RMSE = ' + np.array2string(base_rmse[:,index_mask[i,j]][0], precision=2)+ '\\times 10^{'+ np.array2string(exp_rmse[:,index_mask[i,j]][0], precision=0) +'}$ \n' +
                                  '$rRMSE = ' + np.array2string(errors['rrmse'][:,index_mask[i,j]][0]*100, precision=0) + '\% $ \n'+
                                  # '$||rRSE||_{\infty} = ' + np.array2string(errors['rrse_max'][:,index_mask[i,j]][0]*100, precision=0) + '\% $ \n'+
                                  '$nRMSE = ' + np.array2string(errors['nrmse'][:,index_mask[i,j]][0]*100, precision=0) + '\% $',
                                  # '$||nRSE||_{\infty} = ' + np.array2string(errors['nrse_max'][:,index_mask[i,j]][0]*100, precision=0) + '\% $',
                                prop=dict(size=10), frameon=True, loc='lower right')
                at.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
                axa[i,j].add_artist(at)
                axa[i,j].plot([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])], [np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])], 
                              color='grey', linestyle='--', linewidth = 1)
                axa[i,j].set_aspect('equal', 'box')
    
    axa[-1, -1].axis('off')
    axa[-1, -2].axis('off')
    axa[-1, -3].axis('off')
    at_ = AnchoredText('avg $rRMSE = ' + np.array2string(np.mean(errors['rrmse'][0,:])*100, precision=0) + '\% $ \n'+
                       'avg $||rRSE||_{\infty}= '+ np.array2string(np.mean(errors['rrse_max'][0,:])*100, precision=0) + '\% $\n'+
                       'avg $nRMSE = ' + np.array2string(np.mean(errors['nrmse'][0,:])*100, precision=0) + '\% $ \n'+
                       'avg $||nRSE||_{\infty}= '+ np.array2string(np.mean(errors['nrse_max'][0,:])*100, precision=0) + '\% $',
                        prop=dict(size=10), frameon=True, loc='center')
    at_.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
    axa[-1, -1].add_artist(at_)

    for i in range(4):
        if color == 'rse': 
            if i == 0 or i == 3:
                name = 'RSE \: [N/mm]'
            if i == 1:
                name = 'RSE \: [Nmm]'
            if i ==2: 
                name = 'RSE \: [N]'
        elif color == 'nrse':
           name = 'nRSE \: [\%]'
        cbar = fig.colorbar(scatters[i*3], ax=axa[i,:], orientation='vertical', label=f'Row {i+1} $'+name+'$')
        cbar.set_label('$'+name+'$')

    axa = plt.gca()
    axa.axis('square')


    # Save figure
    if save_path is not None and transf == 'o':
        filename = os.path.join(save_path, 'diagonal_match_'+'D_nonzero_o.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        wandb.log({"45°-plot, D_o": wandb.Image(filename)})
        print('saved D-o-plot')
    if save_path is not None and transf == 'u':
        filename = os.path.join(save_path, 'diagonal_match_'+'D_nonzero_u.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        wandb.log({"45°-plot, D_u": wandb.Image(filename)})
        print('saved D-u-plot')
    # plt.show()
    plt.close()
    return



def multiple_diagonal_plots_paper(save_path: str, Y: np.array, predictions: np.array, transf:str, stats:dict, color='nrse', Y_train = None, pred_train = None):
    ''''
    save_path       (str)           path where images are saved
    Y               (np.array)      Ground truth
    predictions     (np.array)      Predictions
    transf          (str)           't': transformed(normalised), 'o': original scale [MN, cm], 'u': units for simulation [N,mm]
    stats           (dict)          data statistics for normalising / relativising the RMSE
    color           (str)           scatter color: if 'nrse': only one colour bar across all plots. If 'rse': 3 separate colorbars for n, m, v
    '''

    # for the paper convert the units to kN/m and kNm/m
    Y[:,3:6] = Y[:,3:6]*10**(-3)
    predictions[:,3:6] = predictions[:,3:6]*10**(-3)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times New Roman",
        "font.size": 9,
        # "legend.fontsize": 5,
        })

    plt.rc('text.latex', preamble=
        r'\usepackage{amsmath}' + 
        r'\usepackage{times}')

    errors = calculate_errors(Y, predictions, stats, transf, id = 'sig')

    # Plot figure
    fig = plt.figure(figsize=(16/2.54, 12/2.54), dpi=300)
    axa = fig.subplots(3, 3)
    fig.subplots_adjust(wspace=0.6)
    fig.subplots_adjust(hspace=0.4)

    index_mask = np.array([[0,1,2],
                           [3,4,5],
                           [6,7,7]])


    if transf == 'u':
        plotname = np.array([[r'\textit{n}$_\textit{x}$', r'\textit{n}$_\textit{y}$', r'\textit{n}$_{\textit{xy}}$'],
                            [r'\textit{m}$_\textit{x}$', r'\textit{m}$_\textit{y}$', r'\textit{m}$_{\textit{xy}}$'],
                            [r'\textit{v}$_{\textit{xz}}$', r'\textit{v}$_{\textit{yz}}$', '$v_\textit{y}$']])
        # plotname_p = np.array([[r'$\tilde{n}_{x}$', r'$\tilde{n}_{y}$', r'$\tilde{n}_{xy}$'],
         #                    [r'$\tilde{m}_x$', r'$\tilde{m}_y$', r'$\tilde{m}_{xy}$'],
         #                   [r'$\tilde{v}_x$', r'$\tilde{v}_y$', r'$\tilde{v}_y$']])
        plotname_p = np.array([[r'\textit{n}$_{\textit{x,pred}}$', r'\textit{n}$_{\textit{y,pred}}$', r'\textit{n}$_{\textit{xy,pred}}$'],
                            [r'\textit{m}$_{\textit{x,pred}}$', r'\textit{m}$_{\textit{y,pred}}$', r'\textit{m}$_{\textit{xy,pred}}$'],
                            [r'\textit{v}$_{\textit{xz,pred}}$', r'\textit{v}$_{\textit{yz,pred}}$', r'\textit{v}$_{\textit{yz,pred}}$']])
        units = np.array([[r'[kN/m]', r'[kN/m]', r'[kN/m]'],
                          [r'[kNm/m]', r'[kNm/m]', r'[kNm/m]'],  
                          [r'[kN/m]', r'[kN/m]', r'[kN/m]']])


    # find max, min for colorbars
    norms = [mcolors.Normalize(vmin=np.min(errors[color][:, index_mask[i, :]]),
                           vmax=np.max(errors[color][:, index_mask[i, :]]))
                            for i in range(3)]
    scatters = []


    for i in range(3):
        for j in range(3):
            # formatting axes
            for spine in axa[i,j].spines.values():
                spine.set_linewidth(0.5)  # Set the width of the outline
                spine.set_color('black')   # Set the color of the outline
            axa[i,j].tick_params(axis='both', labelsize=4, length=2, width=0.25, color = 'black', labelcolor = 'black')

            if i ==2 and j==2:
                    axa[i,j].set_title(' ')
            else: 
                scatter = axa[i,j].scatter(Y[:,index_mask[i,j]], predictions[:,index_mask[i,j]], marker = 'o', s = 5, 
                              c = errors[color][:, index_mask[i,j]], cmap = 'plasma', linestyle='None', alpha = 1, 
                              norm = norms[i])
                scatters.append(scatter)
                # Only plot three ticks per axis (min, max, mean)
                def round_to(x, round_by):
                    if x >= 0: 
                        return round(x-(x % round_by))
                    else:
                        return round(x-(x % round_by) + round_by)
                
                y_val = [round_to(num, 500) for num in axa[i,j].get_ylim()]
                x_val = [round_to(num, 500) for num in axa[i,j].get_xlim()]     #[min, max]
                y_val.append(round((y_val[0]+y_val[1])/2))
                x_val.append(round((x_val[0]+x_val[1])/2))

                axa[i,j].set_yticks([y_val[0], y_val[2], y_val[1]])
                axa[i,j].set_yticklabels([y_val[0], y_val[2], y_val[1]])
                axa[i,j].set_xticks([x_val[0], x_val[2], x_val[1]])
                axa[i,j].set_xticklabels([x_val[0], x_val[2], x_val[1]])

                if Y_train is not None:
                    print('The paper version of the test plot is not thought for plotting training predictions')
                
                # Labelling of axes
                axa[i,j].set_ylabel(plotname_p[i,j]+' '+ units[i,j])
                axa[i,j].set_xlabel(plotname[i,j]+' '+ units[i,j])
                # axa[i,j].set_xlim([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])])
                # axa[i,j].set_ylim([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])])
                axa[i,j].grid(True, which='major', color='lightgrey', linestyle='-', linewidth=0.25)
                at = AnchoredText(r'$||$\textit{RSE}$||_{\infty}$ = ' + np.array2string(errors['rse_max'][:,index_mask[i,j]][0], precision=1) + ' \n' +
                                  r'\textit{RMSE} = ' + np.array2string(errors['rmse'][:,index_mask[i,j]][0], precision=2) + ' \n' +
                                  r'\textit{rRMSE} = ' + np.array2string(errors['rrmse'][:,index_mask[i,j]][0]*100, precision=1) + '\% \n' +
                                  r'\textit{nRMSE} = ' + np.array2string(errors['nrmse'][:,index_mask[i,j]][0]*100, precision=1) + '\%',
                                prop=dict(size=5), 
                                frameon=True, loc='upper left')
                at.patch.set_edgecolor('lightgrey')
                at.patch.set_boxstyle('round,pad=0.,rounding_size=0.1')
                at.patch.set_linewidth(0.5)
                axa[i,j].add_artist(at)
                # Plot the 45°-dashed line
                axa[i,j].plot([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), 
                               np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])], 
                               [np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), 
                                np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])],
                                color='lightgrey', linestyle='--', linewidth = 1)
            
    axa[-1, -1].axis('off')
    at_ = AnchoredText(r'avg \textit{rRMSE} = ' + np.array2string(np.mean(errors['rrmse'][0,:])*100, precision=1) + '\% \n'+
                       r'avg $||$\textit{rRSE}$||_{\infty}$= '+ np.array2string(np.mean(errors['rrse_max'][0,:])*100, precision=1) + '\% \n'+
                       r'avg \textit{nRMSE} = ' + np.array2string(np.mean(errors['nrmse'][0,:])*100, precision=1) + '\% \n'+
                       r'avg $||$\textit{nRSE}$||_{\infty}$= '+ np.array2string(np.mean(errors['nrse_max'][0,:])*100, precision=1) + '\%',
                        prop=dict(size=7), 
                        frameon=True,loc='center')
    at_.patch.set_edgecolor('lightgrey')
    at_.patch.set_boxstyle('round,pad=0.,rounding_size=0.1')
    at_.patch.set_linewidth(0.5)
    axa[-1, -1].add_artist(at_)

    for i in range(3):
        if color == 'rse': 
            if i == 0 or i == 2:
                name = r'\textit{RSE}'
                unit = '[kN/m]'
            if i == 1:
                name = r'\textit{RSE}'
                unit = '[kNm/m]'
        elif color == 'nrse':
           name = 'nRSE \: [\%]'
        cbar = fig.colorbar(scatters[i*3], ax=axa[i,:], orientation='vertical', label=f'Row {i+1}'+name+' '+unit)
        cbar.set_label('$'+name+'$'+' '+unit)
        cbar.ax.tick_params(width=0.5, labelsize=5)
        cbar.outline.set_linewidth(0.5)

    # axa = plt.gca()
    # axa.set_aspect('equal', 'box')
    # axa.axis('square')


    # Save figure
    if save_path is not None:
        if transf == 'u':
            # filename = os.path.join(save_path, 'diagonal_match_'+'original_units_paper.png')
            filename = os.path.join(save_path, 'diagonal_match_'+'original_units_paper.tif')
            plt.savefig(filename, dpi = 600)
            wandb.log({"45°-plot, og_u": wandb.Image(filename)})
            print('saved sig-u-plots-paper')


    return



'''----------------------------------VERSION SAVING / COPYING------------------------------------------------------------'''



def get_latest_version_folder(base_folder):
        """
        Finds the latest version folder in the base folder with the format 'v_num'.
        """
        version_folders = [f for f in os.listdir(base_folder) if re.match(r'v_\d+', f)]
        version_numbers = [int(re.search(r'v_(\d+)', folder).group(1)) for folder in version_folders]
        return max(version_numbers) if version_numbers else 0


def copy_files_with_incremented_version(src_folder, base_dest_folder, files_to_copy):
    """
    Copies files from src_folder to a new folder in base_dest_folder with an incremented version number.
    """
    # Get the latest version number in the destination folder and increment it
    latest_version = get_latest_version_folder(base_dest_folder)
    new_version = latest_version + 1
    new_folder_name = f"v_{new_version}"
    new_folder_path = os.path.join(base_dest_folder, new_folder_name)
    
    # Create the new versioned folder
    os.makedirs(new_folder_path, exist_ok=True)
    
    # Copy files from the source folder to the new versioned folder
    for file_name in files_to_copy:
        src_path = os.path.join(src_folder, file_name)
        dest_path = os.path.join(new_folder_path, file_name)   

        if file_name == 'best_trained_model.pt':
            model_file = glob.glob(os.path.join(src_folder, 'best_trained_model_*.pt'))
            for i in range(len(model_file)):
                file_name_new = os.path.basename(model_file[i])
                src_path_new = os.path.join(src_folder, file_name_new)
                dest_path_new = os.path.join(new_folder_path, file_name_new)
                if os.path.exists(src_path_new):
                    shutil.copy2(src_path_new, dest_path_new)
                    os.remove(src_path_new)
                else: 
                    print(f"{file_name} does not exist in the source folder.")

        else: 
            if os.path.exists(src_path):
                shutil.copy2(src_path, dest_path)
            else:
                print(f"{file_name} does not exist in the source folder.")

    
    print(f"Files copied to {new_folder_path}")


def copy_files_to_plots_folder(src_folder, dest_folder, files_to_copy):
    """
    Copies specified files from src_folder to a new 'plots' folder within dest_folder.
    """
    # Create the 'plots' folder within the destination folder
    plots_folder = os.path.join(dest_folder, 'plots')
    os.makedirs(plots_folder, exist_ok=True)
    
    # Copy only the specified files to the 'plots' folder
    for file_name in files_to_copy:
        src_path = os.path.join(src_folder, file_name)
        dest_path = os.path.join(plots_folder, file_name)
        
        if os.path.exists(src_path):  # Ensure the file exists in the source folder
            shutil.copy2(src_path, dest_path)
        else:
            print(f"{file_name} does not exist in the source folder.")
    
    print(f"Specified files copied to {plots_folder}")


def copy_all_files(src_folder, dest_folder):
    # Ensure destination folder exists
    os.makedirs(dest_folder, exist_ok=True)

    # Walk through the source folder
    for root, _, files in os.walk(src_folder):
        for file in files:
            src_path = os.path.join(root, file)  # Full path of the source file
            rel_path = os.path.relpath(src_path, src_folder)  # Relative path
            dest_path = os.path.join(dest_folder, rel_path)  # Destination path

            # Ensure the subdirectories exist in the destination
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # Copy the file
            shutil.copy2(src_path, dest_path)  # copy2 preserves metadata
            print('Copied all files from ', src_folder, 'to', dest_folder)


'''---------------------------------- PLOTTING RAW SAMPLED DATA ------------------------------------------------------------'''


def plots_mike(X, predictions, true, save_path, tag = None):
    '''
    X:              (np.array)      input test vector (eps and t)
    predictions:    (np.array)      predictions (sig)
    sig:            (np.array)      ground truth (sig)
    '''
    plotname_sig = np.array(['$n_x$', '$n_y$', '$n_{xy}$',
                            '$m_x$', '$m_y$', '$m_{xy}$',
                            '$v_{xz}$', '$v_{yz}$', '$v_y$'])
    
    plotname_eps = np.array([r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$',
                             r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$',
                             r'$\gamma_x$', r'$\gamma_y$', r'$\gamma_{xy}$',
                             ])
    
    if tag == 'D':
        plotname_sig = np.array(['$D_{m,11}$', '$D_{m,12}$', '$D_{m,13}$', '$D_{mb,11}$', '$D_{mb,12}$', '$D_{mb,13}$',
                        '$D_{m,21}$', '$D_{m,22}$', '$D_{m,23}$', '$D_{mb,21}$', '$D_{mb,22}$', '$D_{mb,23}$',
                        '$D_{m,31}$', '$D_{m,32}$', '$D_{m,33}$', '$D_{mb,31}$', '$D_{mb,32}$', '$D_{mb,33}$',
                        '$D_{bm,11}$', '$D_{bm,12}$', '$D_{bm,13}$', '$D_{b,11}$', '$D_{b,12}$', '$D_{b,13}$',
                        '$D_{bm,21}$', '$D_{bm,22}$', '$D_{bm,23}$', '$D_{b,21}$', '$D_{b,22}$', '$D_{b,23}$',
                        '$D_{bm,31}$', '$D_{bm,32}$', '$D_{bm,33}$', '$D_{b,31}$', '$D_{b,32}$', '$D_{b,33}$',
                        '$D_{s,11}$', '$D_{s,22}$'
                        ])

    nRows = 8
    nCols = predictions.shape[1]
    fig, axs = plt.subplots(nCols, nRows, figsize=(2*nRows, 2*nCols))
    for i in range(nCols):
        for j in range(nRows):
            axs[i, j].plot(X[:,j], true[:,i], 'o', label = 'ground truth')
            axs[i, j].plot(X[:,j], predictions[:,i], 'ro', markerfacecolor = 'none', label='predictions')
            if i == nRows-1:
                axs[i, j].set_xlabel(plotname_eps[j])
            if j == 0:
                axs[i, j].set_ylabel(plotname_sig[i], rotation = 90)
    plt.legend()

    if save_path is not None:
        if tag == None:
            plt.savefig(os.path.join(save_path, "testo.png"))
            print('saved mike plot')
        elif tag == 'D':
            plt.savefig(os.path.join(save_path, "testo-D.png"))
            print('saved mike plot for D')
    plt.close()


def plots_mike_dataset(x_train, x_test, y_train, y_test, save_path, tag, tag2 = None, 
                       x_add = None, y_add = None, x_add2 = None, y_add2 = None, x_add3 = None, y_add3 = None, x_add4 = None, y_add4 = None, 
                       add_dict = None, outliers = False, linel = False):
    '''
    train       (array)     data for training (either x or y)
    test        (test)      data for testing or evaluation (either x or y, corresp. to train)
    save_path   (str)       save_path
    outliers    (bool)      if True: also plots outliers, only used for add_dict != None
    linel       (bool)      if True: additionally plots maximum linear elastic value of the corresponding stiffness matrix entry
    '''

    # plt.rcParams.update({
    #     "text.usetex": True,
    #    "font.family": "Helvetica",
    #     "font.size": 12,
    #     })
    
    units = np.array([r' $[MN/cm]$', r' $[MN/cm]$', r' $[MN/cm]$',
                          r' $[MNcm/cm]$', r' $[MNcm/cm]$', r' $[MNcm/cm]$',  
                          r' $[MN/cm]$', r' $[MN/cm]$', r' $[MN/cm]$'])
    
    plotname_sig = np.array(['$n_x$', '$n_y$', '$n_{xy}$',
                            '$m_x$', '$m_y$', '$m_{xy}$',
                            '$v_{xz}$', '$v_{yz}$', '$t_1$', '$t_2$', '$n_{lay}$'])
    
    plotname_eps = np.array([r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$',
                             r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$',
                             r'$\gamma_{xz}$', r'$\gamma_{yz}$', r'$t_1$', r'$t_2$', r'$n_{lay}$'
                             ])

    if tag2 == 'D':
        plotname_sig = np.array(['$D_{m,11}$', '$D_{m,12}$', '$D_{m,13}$', '$D_{mb,11}$', '$D_{mb,12}$', '$D_{mb,13}$',
                        '$D_{m,21}$', '$D_{m,22}$', '$D_{m,23}$', '$D_{mb,21}$', '$D_{mb,22}$', '$D_{mb,23}$',
                        '$D_{m,31}$', '$D_{m,32}$', '$D_{m,33}$', '$D_{mb,31}$', '$D_{mb,32}$', '$D_{mb,33}$',
                        '$D_{bm,11}$', '$D_{bm,12}$', '$D_{bm,13}$', '$D_{b,11}$', '$D_{b,12}$', '$D_{b,13}$',
                        '$D_{bm,21}$', '$D_{bm,22}$', '$D_{bm,23}$', '$D_{b,21}$', '$D_{b,22}$', '$D_{b,23}$',
                        '$D_{bm,31}$', '$D_{bm,32}$', '$D_{bm,33}$', '$D_{b,31}$', '$D_{b,32}$', '$D_{b,33}$',
                        '$D_{s,11}$', '$D_{s,22}$'
                        ])

    index_mask = np.array([0,1,2,3,4,5, 6, 7, 7])

    if tag2 == None:
        nRows = 8
        nCols = 8
    elif tag2 == 'D':
        nRows = 11
        nCols = 38

    
    fig, axs = plt.subplots(nCols, nRows, figsize=(2*nRows, 2*nCols))
    used_labels = set()
    for i in range(nCols):
        for j in range(nRows):
            
            if x_add is not None:
                axs[i, j].plot(x_add[:,j], y_add[:,i], 'o', 
                           markerfacecolor = 'lightblue', markeredgecolor = 'lightblue', 
                           label = tag+'_add', alpha = 0.05, markersize = 3)
            if x_add2 is not None:
                axs[i, j].plot(x_add2[:,j], y_add2[:,i], 'o', 
                           markerfacecolor = 'lightgreen', markeredgecolor = 'lightgreen', 
                           label = tag+'_add', alpha = 0.05, markersize = 3)
            if x_add3 is not None:
                axs[i, j].plot(x_add3[:,j], y_add3[:,i], 'o', 
                           markerfacecolor = 'orange', markeredgecolor = 'orange', 
                           label = tag+'_add', alpha = 0.05, markersize = 3)
            if x_add4 is not None:
                axs[i, j].plot(x_add4[:,j], y_add4[:,i], 'o', 
                           markerfacecolor = 'yellow', markeredgecolor = 'yellow', 
                           label = tag+'_add', alpha = 0.05, markersize = 3)
                
            if add_dict is not None: 
                colors = plt.cm.get_cmap('viridis', len(add_dict)+1)(np.arange(len(add_dict)+1))
                for k, l in zip(add_dict.keys(), range(len(add_dict))):
                    label = tag+k
                    if label not in used_labels:
                        if tag2 == None:
                            axs[i,j].plot(add_dict[k]['x_data'][:,j], add_dict[k]['y_data'][:,i], 'o', 
                                        markerfacecolor = colors[l], markeredgecolor = colors[l],
                                        label = label, alpha = 0.05, markersize = 3)
                            if outliers: 
                                # CAREFUL. marking the **D**-outliers in the sig-eps plots!
                                x_vals = add_dict[k]['x_data'][:,j]
                                y_vals = add_dict[k]['y_data'][:,i]
                                mask = (add_dict[k]['D_outliers'] != 0).all(axis=1).astype(int).reshape(-1,1)
                                print('Amount of D-outliers: ', sum(mask))
                                x_vals_nonzero = x_vals[mask]
                                y_vals_nonzero = y_vals[mask]
                                axs[i,j].plot(x_vals_nonzero, y_vals_nonzero, 'o', 
                                            markerfacecolor = 'red', markeredgecolor = 'red',
                                            label = label, alpha = 0.05, markersize = 3)

                        elif tag2 == 'D':
                            axs[i,j].plot(add_dict[k]['x_data'][:,j], add_dict[k]['D_data'][:,i], 'o', 
                                        markerfacecolor = colors[l], markeredgecolor = colors[l],
                                        label = label, alpha = 0.05, markersize = 3)
                            if outliers:
                                x_vals = add_dict[k]['x_data'][:,j]
                                y_vals = add_dict[k]['D_outliers'][:,i]
                                nonzero_mask = y_vals != 0
                                print('Amount of outliers: ', sum(nonzero_mask))
                                x_vals_nonzero = x_vals[nonzero_mask]
                                y_vals_nonzero = y_vals[nonzero_mask]
                                axs[i,j].plot(x_vals_nonzero, y_vals_nonzero, 'o', 
                                            markerfacecolor = 'red', markeredgecolor = 'red',
                                            label = label, alpha = 0.05, markersize = 3)
                            if linel: 
                                for t, E in zip([200,450], [32000, 39000]):
                                    D_linel_MN_cm = find_D_linel(t,E)
                                    D_linel_N_mm = transf_units(D_linel_MN_cm.reshape((1,8,8)), 'D', forward=False, linel=True)
                                    D_linel_ = np.concatenate((D_linel_N_mm[:,:6,:6].reshape((-1,36)), D_linel_N_mm[:,6,6].reshape((-1,1)), D_linel_N_mm[:,7,7].reshape((-1,1))), axis=1)
                                    D_linel = D_linel_[0,i]
                                    x_line = np.linspace(add_dict[k]['x_data'][:,j].min(), add_dict[k]['x_data'][:,j].max(), 100)
                                    y_line = D_linel*np.ones_like(x_line)
                                    axs[i,j].plot(x_line, y_line, color = 'black', linestyle = '--', label = '$D_{linel, t = '+str(t)+'}$', linewidth = 1)

                        used_labels.add(label)
                            
                        # print(f'plotted ', tag2, f': row {i}, column {j} for data no {k}, with labels')
                        # if k == 1: print(f'({i}, {j}), k=1: with labels')
                    else: 
                        if tag2 == None:
                            axs[i,j].plot(add_dict[k]['x_data'][:,j], add_dict[k]['y_data'][:,i], 'o', 
                                        markerfacecolor = colors[l], markeredgecolor = colors[l],
                                        label = '_nolegend_', alpha = 0.05, markersize = 3)
                            if outliers: 
                                # CAREFUL. marking the **D**-outliers in the sig-eps plots!
                                x_vals = add_dict[k]['x_data'][:,j]
                                y_vals = add_dict[k]['y_data'][:,i]
                                mask = (add_dict[k]['D_outliers'] != 0).all(axis=1).astype(int).reshape(-1,1)
                                x_vals_nonzero = x_vals[mask]
                                y_vals_nonzero = y_vals[mask]
                                if j == 0:
                                    print('Amount of D-outliers: ', sum(mask))
                                axs[i,j].plot(x_vals_nonzero, y_vals_nonzero, 'o', 
                                            markerfacecolor = 'red', markeredgecolor = 'red',
                                            label = '_nolegend_', alpha = 0.05, markersize = 3)
                        elif tag2 == 'D':
                            axs[i,j].plot(add_dict[k]['x_data'][:,j], add_dict[k]['D_data'][:,i], 'o', 
                                        markerfacecolor = colors[l], markeredgecolor = colors[l],
                                        label = '_nolegend_', alpha = 0.05, markersize = 3)
                            if outliers:
                                x_vals = add_dict[k]['x_data'][:,j]
                                y_vals = add_dict[k]['D_outliers'][:,i]
                                nonzero_mask = y_vals != 0
                                if j == 0:
                                    print('Amount of outliers: ', sum(nonzero_mask))
                                x_vals_nonzero = x_vals[nonzero_mask]
                                y_vals_nonzero = y_vals[nonzero_mask]
                                axs[i,j].plot(x_vals_nonzero, y_vals_nonzero, 'o', 
                                        markerfacecolor = 'red', markeredgecolor = 'red',
                                        label = '_nolegend_', alpha = 0.05, markersize = 3)
                                
                            if linel: 
                                for t, E in zip([200,450], [32000, 39000]):
                                    D_linel_MN_cm = find_D_linel(t,E)
                                    D_linel_N_mm = transf_units(D_linel_MN_cm.reshape((1,8,8)), 'D', forward=False, linel=True)
                                    D_linel_ = np.concatenate((D_linel_N_mm[:,:6,:6].reshape((-1,36)), D_linel_N_mm[:,6,6].reshape((-1,1)), D_linel_N_mm[:,7,7].reshape((-1,1))), axis=1)
                                    D_linel = D_linel_[0,i]
                                    x_line = np.linspace(add_dict[k]['x_data'][:,j].min(), add_dict[k]['x_data'][:,j].max(), 100)
                                    y_line = D_linel*np.ones_like(x_line)
                                    axs[i,j].plot(x_line, y_line, color = 'black', linestyle = '--', label = '_nolegend_', linewidth = 1)

                        # print(f'plotted ', tag2, f': row {i}, column {j} for data no {k}, without labels')
                        # if k == 1: print(f'({i}, {j}), k=1: without labels')

        
            if x_train is not None:
                axs[i, j].plot(x_train[:,j], y_train[:,i], 'o', 
                           label = 'train', color = 'blue', markersize = 3)     
            if x_test is not None:
                axs[i, j].plot(x_test[:,j], y_test[:,i], 'o', 
                           markerfacecolor = 'lightcoral', markeredgecolor = 'lightcoral', 
                           label = tag, alpha = 0.05, markersize = 3)

            if i == nRows-1:
                axs[i, j].set_xlabel(plotname_eps[j])
            if j == 0:
                axs[i, j].set_ylabel(plotname_sig[i])

        print(f'plotted ', tag2, f': row {i} for all columns and all data')


    plt.title('training vs '+tag+' data')
    # plt.tight_layout()
    if add_dict is not None:
        legend_handles = []
        for key, l in zip(add_dict.keys(), range(len(add_dict))): 
            pt = Line2D([0], [0], marker='o', color=colors[l], markerfacecolor=colors[l], markersize=3, alpha=1)
            legend_handles.append(pt)
        # fig.legend(legend_handles, colors, loc = 'upper right')
        fig.legend(legend_handles, list(add_dict.keys()), loc = 'upper right')
    else:
        fig.legend()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "data_scatter_"+tag+".png"))
        print('saved data scatter plot ', tag)
    plt.close() 


def plot_nathalie(data_in, data_in_test = None, save_path=None, tag=None):
    if tag == 'eps+t':
        if data_in.shape[1] == 9:
            col = ['epsx', 'epsy', 'epsxy', 'chix', 'chiy', 'chixy', 'gamx', 'gamy', 't']
            label = np.array([r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$',
                                r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$',
                                r'$\gamma_x$', r'$\gamma_y$', r'$t$'
                                ])
        elif data_in.shape[1] == 10:
            col = ['epsx', 'epsy', 'epsxy', 'chix', 'chiy', 'chixy', 'gamx', 'gamy', 't_1', 't_2']
            label = np.array([r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$',
                                r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$',
                                r'$\gamma_x$', r'$\gamma_y$', r'$t_1$', r'$t_2$'
                                ])
        elif data_in.shape[1] == 11:
            col = ['epsx', 'epsy', 'epsxy', 'chix', 'chiy', 'chixy', 'gamx', 'gamy', 't_1', 't_2', 'nl']
            label = np.array([r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$',
                                r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$',
                                r'$\gamma_x$', r'$\gamma_y$', r'$t_1$', r'$t_2$', r'$n_{lay}$'
                                ])
    elif tag == 'eps+t_RC':
        col = ['epsx', 'epsy', 'epsxy', 'chix', 'chiy', 'chixy', 'gamx', 'gamy', 't', 'rho', 'CC']
        label = np.array([r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$',
                            r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$',
                            r'$\gamma_x$', r'$\gamma_y$', r'$t$', r'$\rho$', r'$CC$'
                            ])
    if tag == 'sig':
        col = ['nx', 'ny', 'nxy', 'mx', 'my', 'mxy', 'vx', 'vy']
        label = np.array(['$n_x$', '$n_y$', '$n_{xy}$',
                            '$m_x$', '$m_y$', '$m_{xy}$',
                            '$v_{xz}$', '$v_{yz}$', '$v_y$'])
    elif tag == 'eps':
        col = ['epsx', 'epsy', 'epsxy', 'chix', 'chiy', 'chixy', 'gamx', 'gamy']
        label = np.array([r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$',
                             r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$',
                             r'$\gamma_x$', r'$\gamma_y$', r'$\gamma_{xy}$',
                             ])
    elif tag == 't':
        col = ['t_1', 'rho', 'CC', 'Ec', 'tb0', 'tb1', 'ect', 'ec0', 'fcp', 'fct']
        label = np.array([r'$t_1$', r'$\rho$', r'$CC$', r'$Ec$', r'$tb0$', r'$tb1$',
                           r'$\varepsilon_{ct}$', r'$\varepsilon_{c0}$', r'$f_{cp}$', 
                           r'$f_{ct}$'])
    df_in = pd.DataFrame(data_in, columns = col)
    df_in = df_in.convert_dtypes()
    mat_fig = pd.plotting.scatter_matrix(df_in, alpha = 0.2, figsize= (8,8), diagonal='kde')
    if data_in_test is not None:
        df_in_eval = pd.DataFrame(data_in_test, columns=col)
        df_in_eval = df_in_eval.convert_dtypes()

        for i in range(len(df_in_eval.columns)):
            for j in range(len(df_in_eval.columns)):
                ax = mat_fig[i, j]
                ax.scatter(df_in_eval[df_in_eval.columns[j]], df_in_eval[df_in_eval.columns[i]], color='lightcoral', alpha=0.5, s=3)

    for i, ax in enumerate(mat_fig[:,0]):
        ax.set_ylabel(label[i], labelpad = 10, rotation=90)
    for j, ax in enumerate(mat_fig[-1,:]):
        ax.set_xlabel(label[j], labelpad = 10)
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'scatter_matrix_'+tag+'.png'), dpi = 300)
        print('Saved scatter matrix plot for '+tag)
    plt.tight_layout
    plt.close()
    return



def plot_paper_comp(data, save_path, ticks = False, number = 4000, id='m'):
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "font.size": 8,
        })

    plt.rc('text.latex', preamble=
       r'\usepackage{amsmath}' + 
       r'\usepackage{times}')
    
    if id == 'm':
        label_eps = [r'$\chi_\textit{x}$', r'$\chi_\textit{y}$', r'$\chi_{\textit{xy}}$']
        label_sig = [r'$\textit{m}_\textit{x}$', r'$\textit{m}_\textit{y}$', r'$\textit{m}_{\textit{xy}}$']
        nrow, ncol = 3, 3
    elif id == 'n':
        label_eps = [r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$']
        label_sig = ['$n_x$', '$n_y$', '$n_{xy}$']
        nrow, ncol = 3, 3
    elif id == 'v':
        label_eps = [r'$\gamma_{xz}$', r'$\gamma_{yz}$']
        label_sig = ['$v_{xz}$', '$v_{yz}$']
        nrow, ncol= 2, 2

    def create_subplot_nat(ax, data_sub, row, ticks):
        """Create a 3x3 grid of subplots in the given axis."""
        for i in range(nrow):
            for j in range(ncol):
                
                # Plot scatter plots in all off-diagonal plots
                if i != j: 
                    ax[i][j].scatter(data_sub[0][:, j], data_sub[0][:, i], alpha=0.2, color = '#3b528b', s = 2.25, edgecolors="none")
                    
                    # make nice ticks of just max and min, and round them
                    if id=='m':
                        round_by = [1, 50]
                    elif (id=='n') or (id == 'v'):
                        round_by = [0.1,100]
                    y_val = [round(num/ round_by[row])*round_by[row] for num in ax[i][j].get_ylim()]
                    x_val = [round(num/ round_by[row])*round_by[row] for num in ax[i][j].get_xlim()]     #[min, max]

                    ax[i][j].set_yticks([y_val[0], y_val[1]])
                    ax[i][j].set_yticklabels([y_val[0], y_val[1]])
                    ax[i][j].set_xticks([x_val[0], x_val[1]])
                    ax[i][j].set_xticklabels([x_val[0], x_val[1]])

                    # Adjusting the label position
                    dx, dy = 9/72, 4/72
                    offset_y = matplotlib.transforms.ScaledTranslation(0, dy, fig.dpi_scale_trans)
                    offset_x = matplotlib.transforms.ScaledTranslation(dx, 0, fig.dpi_scale_trans)
                    labely = ax[i][j].yaxis.get_majorticklabels()
                    labely[0].set_transform(labely[0].get_transform()+offset_y)
                    labely[1].set_transform(labely[1].get_transform()-offset_y)
                    labelx = ax[i][j].xaxis.get_majorticklabels()
                    labelx[0].set_transform(labelx[0].get_transform()+offset_x)
                    labelx[1].set_transform(labelx[1].get_transform()-offset_x)
                    

                # Plot histograms in diagonal plots
                if i == j: 
                    df_sub = pd.DataFrame(data_sub[0])
                    df_sub.iloc[:,i].hist(ax=ax[i][j], alpha=0.2, color = '#1f9e89', edgecolor = '#1f9e89', bins = 10, grid = False)
                    ax[i][j].set_ylabel(' ')

                    # make nice ticks of just max and min, round them
                    if id=='m':
                        round_by = [1, 50]
                    elif (id=='n') or (id == 'v'):
                        round_by = [0.1,100]
                    x_val = [round(num/ round_by[row])*round_by[row] for num in ax[i][j].get_xlim()]
                    ax[i][j].set_xticks([x_val[0], x_val[1]])
                    ax[i][j].set_xticklabels([x_val[0], x_val[1]])
                    dx = 8/72
                    offset_x = matplotlib.transforms.ScaledTranslation(dx, 0, fig.dpi_scale_trans)
                    labelx = ax[i][j].xaxis.get_majorticklabels()
                    labelx[0].set_transform(labelx[0].get_transform()+offset_x)
                    labelx[1].set_transform(labelx[1].get_transform()-offset_x)
                    
                    

                    # To create the axis of scatter plot, create invisible scatter here and create double set of axes.
                    if i == 0 and j == 0: 
                        ax_ = ax[i][j].twinx()
                        ax_.scatter(data_sub[0][:, j], data_sub[0][:, i], alpha=0, color = '#3b528b', s = 2.25, edgecolors="none")
                        ax_.tick_params(axis = 'y', which = 'both', labelsize=5, length=2, width=0.25, color = 'lightgrey', labelcolor = 'grey', left=True)
                        for spine in ax_.spines.values():
                            spine.set_linewidth(0.5)  # Set the width of the outline
                            spine.set_color('lightgrey')   # Set the color of the outline
                        ax_.yaxis.set_label_position("left")
                        ax_.yaxis.tick_left()

                        # make nice ticks of just max and min, and round them
                        if id=='m':
                            round_by = [1, 50]
                        elif (id=='n') or (id == 'v'):
                            round_by = [0.1,100]
                        y_val = [round(num/ round_by[row])*round_by[row] for num in ax_.get_ylim()]
                        x_val = [round(num/ round_by[row])*round_by[row] for num in ax_.get_xlim()]     #[min, max]
                        ax_.set_yticks([y_val[0], y_val[1]])
                        ax_.set_yticklabels([y_val[0], y_val[1]])
                        ax_.set_xticks([x_val[0], x_val[1]])
                        ax_.set_xticklabels([x_val[0], x_val[1]])
                        # Adjusting the label position
                        dx, dy = 8/72, 3/72
                        offset_y = matplotlib.transforms.ScaledTranslation(0, dy, fig.dpi_scale_trans)
                        offset_x = matplotlib.transforms.ScaledTranslation(dx, 0, fig.dpi_scale_trans)
                        labely = ax_.yaxis.get_majorticklabels()
                        labely[0].set_transform(labely[0].get_transform()+offset_y)
                        labely[1].set_transform(labely[1].get_transform()-offset_y)
                        labelx = ax_.xaxis.get_majorticklabels()
                        labelx[0].set_transform(labelx[0].get_transform()+offset_x)
                        labelx[1].set_transform(labelx[1].get_transform()-offset_x)
                        

                # set labels for epsilon and sigma(row == 0)
                labels = [label_eps, label_sig]
                if i == nrow-1:
                    ax[i][j].set_xlabel(labels[row][j], labelpad = 1.5)
                if j == 0:
                    if i == 0: 
                        ax[i][j].set_ylabel(labels[row][i], labelpad = 20)
                    else: 
                        ax[i][j].set_ylabel(labels[row][i])

                if ticks:
                    # Hide x-axis labels for top and middle rows
                    if i < nrow-1:  
                        ax[i][j].set_xticklabels([])
                        ax[i][j].tick_params(axis="x", which="both", bottom=False, top=False)
                    # Hide y-axis labels for all but the leftmost column
                    if (j > 0) and (j != i):
                        ax[i][j].set_yticklabels([])
                        ax[i][j].tick_params(axis="y", which="both", left=False, right=False)

                    # Set y-axis ticks of histograms:
                    if i == j: 
                        max_y_val = ax[i][j].get_ylim()[1]
                        max_y_1_4 = round(max_y_val/4 / 50) * 50
                        max_y_3_4 = round(max_y_val*3/4 / 50) * 50
                        ax[i][j].set_yticks([max_y_1_4, max_y_3_4])
                        ax[i][j].set_yticklabels([max_y_1_4, max_y_3_4])
                        ax[i][j].tick_params(axis='y', which = 'both', direction = 'in', right = True, colors = '#1f9e89')
                        ax[i][j].yaxis.set_ticks_position('right')
                        ax[i][j].yaxis.set_tick_params(pad=-12, labelcolor = '#1f9e89')

                # don't plot any ticks
                else: 
                    ax[i][j].set_xticklabels([])
                    ax[i][j].set_yticklabels([])
                    ax[i][j].tick_params(axis="x", which="both", bottom=False, top=False)
                    ax[i][j].tick_params(axis="y", which="both", left=False, right=False)

    
    def create_subplot_mik(ax, data_sub, ticks):
        """Create a 3x3 grid of subplots in the given axis."""
        for i in range(nrow):
            for j in range(ncol):
                
                ax[i][j].scatter(data_sub[0][:, j], data_sub[1][:, i], alpha=0.2, color = '#3b528b', s = 2.25, edgecolors="none")

                # make nice ticks of just max and min, and round them
                if id=='m':
                    round_by = [1, 50]
                elif (id=='n') or (id == 'v'):
                    round_by = [0.1,100]
                y_val = [round(num/ round_by[1])*round_by[1] for num in ax[i][j].get_ylim()]
                x_val = [round(num/ round_by[0])*round_by[0] for num in ax[i][j].get_xlim()]     #[min, max]

                ax[i][j].set_yticks([y_val[0], y_val[1]])
                ax[i][j].set_yticklabels([y_val[0], y_val[1]])
                ax[i][j].set_xticks([x_val[0], x_val[1]])
                ax[i][j].set_xticklabels([x_val[0], x_val[1]])

                # Adjusting the label position
                dx, dy = 8/72, 4/72
                offset_y = matplotlib.transforms.ScaledTranslation(0, dy, fig.dpi_scale_trans)
                offset_x = matplotlib.transforms.ScaledTranslation(dx, 0, fig.dpi_scale_trans)
                labely = ax[i][j].yaxis.get_majorticklabels()
                labely[0].set_transform(labely[0].get_transform()+offset_y)
                labely[1].set_transform(labely[1].get_transform()-offset_y)
                labelx = ax[i][j].xaxis.get_majorticklabels()
                labelx[0].set_transform(labelx[0].get_transform()+offset_x)
                labelx[1].set_transform(labelx[1].get_transform()-offset_x)

                if ticks: 
                    # Hide x-axis labels for top and middle rows
                    if i < 2:  
                        ax[i][j].set_xticklabels([])
                        ax[i][j].tick_params(axis="x", which="both", bottom=False, top=False)
                    # Hide y-axis labels for all but the leftmost column
                    if j > 0:
                        ax[i][j].set_yticklabels([])
                        ax[i][j].tick_params(axis="y", which="both", left=False, right=False)
                
                else: # don't plot any ticks
                    ax[i][j].set_xticklabels([])
                    ax[i][j].set_yticklabels([])
                    ax[i][j].tick_params(axis="x", which="both", bottom=False, top=False)
                    ax[i][j].tick_params(axis="y", which="both", left=False, right=False)

                if i == nrow-1:
                    ax[i][j].set_xlabel(label_eps[j])
                if j == 0: 
                    ax[i][j].set_ylabel(label_sig[i])


    fig = plt.figure(figsize=(14/2.54, 18/2.54))
    outer_grid = gridspec.GridSpec(3, 2, wspace=0.3, hspace=0.2)
    if ticks:
        plt.subplots_adjust(left = 0.15)

    for i in range(3):
        for j in range(2):
            data_sub = data[str(i)+'-'+str(j)]
            
            if ticks:
                inner_grid = gridspec.GridSpecFromSubplotSpec(
                    nrow,ncol, subplot_spec=outer_grid[i,j], wspace=0, hspace=0
                )
            else: 
                inner_grid = gridspec.GridSpecFromSubplotSpec(
                    nrow,ncol, subplot_spec=outer_grid[i,j], wspace=0, hspace=0
                )
            # Create space for plots
            axes = []
            for row in range(nrow):
                axes_row = []
                for col in range(ncol):
                    ax = fig.add_subplot(inner_grid[row, col])
                    for spine in ax.spines.values():
                        spine.set_linewidth(0.5)  # Set the width of the outline
                        spine.set_color('lightgrey')   # Set the color of the outline
                    ax.tick_params(axis='both', labelsize=5, length=2, width=0.25, color = 'lightgrey', labelcolor = 'grey')
                    axes_row.append(ax)
                axes.append(axes_row)

            # Fill plots with desired data
            if i < 2:
                create_subplot_nat(axes, data_sub, i, ticks)
            elif i == 2:
                create_subplot_mik(axes, data_sub, ticks)
    
        

    # plt.tick_params(axis='both', labelsize=4, length=2, width=0.25)

    fig.text(0.31, 0.91, "Global \n $(N = "+str(number)+")$", ha='center', va='center', fontsize=9)
    fig.text(0.73, 0.91, "Local \n $(N = "+str(number)+")$", ha='center', va='center', fontsize=9)
    if id == 'm':
        title = fig.text(0.05, 0.23, "Curvature vs. Moment", ha='center', va='center', rotation='vertical', fontsize=9)
        fig.text(0.06, 0.5, "Moment vs. Moment \n [kNm/m]", ha='center', va='center', rotation='vertical', fontsize=9) 
        fig.text(0.06, 0.77, "Curvature vs. Curvature \n [mrad/m]", ha='center', va='center', rotation='vertical', fontsize=9) 
        fig.text(0.02, 0.31, "(c)", ha='center', va='center', rotation='horizontal', fontsize=9)
        fig.text(0.02, 0.57, "(b)", ha='center', va='center', rotation='horizontal', fontsize=9)
        fig.text(0.02, 0.86, "(a)", ha='center', va='center', rotation='horizontal', fontsize=9)
        # or [10^{-3}$ $mrad/mm]
    elif id == 'n' or id == 'v':
        fig.text(0.06, 0.23, "Strains vs. Stresses", ha='center', va='center', rotation='vertical', fontsize=9)
        fig.text(0.06, 0.5, "Stresses vs. Stresses \n [kN/m]", ha='center', va='center', rotation='vertical', fontsize=9) 
        fig.text(0.06, 0.77, "Strains vs. Strains \n [‰]", ha='center', va='center', rotation='vertical', fontsize=9)

    # font_properties = title.get_fontproperties()
    # print(f"Font family: {font_properties.get_name()}")

    if save_path is not None:
        # plt.savefig(os.path.join(save_path, 'comp_paper_'+id+'.png'), dpi = 300)
        plt.savefig(os.path.join(save_path, 'comp_paper_'+id+'.tif'), dpi = 600)
        print('Saved comparison plot for paper')
    plt.tight_layout
    plt.close()


'''---------------------------------- PLOTTING RAW SAMPLED 2D DATA as 3D scatter ------------------------------------------------------------'''

def plotting_sampled_scatter_2D(path_collection, path_deployment = None, 
                                filter_type = None, plot_type = 'eps', clean_data = None, save_new_data = False, save_path = None):
    '''
    Plot the sampled data [eps_x, eps_y, eps_xy] in a 3D scatter plot
    path_collection     (str-list)          location where epsilon is stored
    path_deployment     (str-list)          to also plot data from deployment additionally to the data from original dataset.
    filter_type         (str)               can be 'small', 'large' or None if want to use different filter altogether, need to define new category.
    plot_type           (str)               can be 'eps' or 'sig' depending on which data to plot
    clean_data          (str)               path to data which doesn't contain outliers. Basis for creating hull to determine which points are outliers
    save_new_data       (bool)              if True: saves data without outliers to new folder.
    save_path           (str)               location where to save the image.
    
    '''

    # 0 - Read data
    points = {}
    for path in path_collection:
        path_extended = os.path.join('04_Training\\data\\', path)
        data = read_data(path_extended, id = plot_type)
        points[path] = data[:,:3]

    # 1a - Filter data (with filter_type)
    if clean_data is None:
        criteria_mask = get_filter_mask(path_collection, points, filter_type)
        filtered_points, points_ = filter_2D_scatterdata(points, criteria_mask)
    
    # 1b - Determine outliers (if clean_data is not None)
    elif clean_data is not None: 
        if filter_type is not None: 
            raise UserWarning('Please use either the filter_type or the clean_data but not both at the same time.')
        outlier_mask = get_outliers(path_collection, clean_data)
        outlier_points, points_ = filter_2D_scatterdata(points, outlier_mask)

    # 1c - Read deployment points
    if path_deployment is not None:
        deployment_points = {}
        for path in path_deployment:
            path_extended = os.path.join('05_Deploying\\data_out\\', path)
            loadsteps = get_loadsteps(path_extended)
            deployment_points[path] = get_data_deployment(loadsteps, path_extended, plot_type)
            print(f'Amount of points for deployment points in {path}: {deployment_points[path].shape[0]}')


    # 2 - Plot data
    if clean_data is not None:
        plotly_2D_scatter(outlier_points, points_, plot_type, save_path)
    elif path_deployment is not None: 
        plotly_2D_scatter(deployment_points, points_, plot_type, save_path)
    else: 
        plotly_2D_scatter(filtered_points, points_, plot_type, save_path)
       


    # 3 - Save data without outliers if desired
    if save_new_data:
        save_data_without_outliers(outlier_mask, path_collection)


    return

def get_filter_mask(path_collection, points, filter_type, eps_filter = True):
    # note: this filter will always filter according to epsilon (eps_filter = True)
    # this would need to be redefined if needed for filtering according to sigma. 

    criteria_mask = {}

    if eps_filter: 
        points_ = {}
        for path in path_collection:
            path_extended = os.path.join('04_Training\\data\\', path)
            data = read_data(path_extended, id = 'eps')
            points_[path] = data[:,:3]
    else: 
        points_ = points
        # do nothing, i.e. use the filter criteria on points = sigma instead of points = epsilon.

    for path in points_.keys():
        if filter_type is None:
            criteria_mask[path] = np.ones((points_[path].shape[0],), dtype = bool)          # to plot entire dataset
        elif filter_type == 'small':
            criteria_mask[path] = (abs(points_[path][:,1]) <0.2e-4) & (abs(points_[path][:,2]) <0.2e-4)
        elif filter_type == 'large':
            criteria_mask[path] = (abs(points_[path][:,1]) <2e-4) & (abs(points_[path][:,2]) <2e-4)
        if np.sum(criteria_mask[path]) == 0:
            raise UserWarning('No points left to plot. Please choose different filter criterion.')
        else:
            print(f'Amount of data points after filtering dataset {path}: {np.sum(criteria_mask[path])}')

        
    return criteria_mask

def filter_2D_scatterdata(points, criteria_mask):
    filtered_points = {}
    for key in points.keys():
        filtered_points[key] = points[key][criteria_mask[key]]
    
    for path in points.keys():
        if points[path].shape[0]>500000:
            # if plotting entire dataset and one dataset is larger than 500k points: only plot every 10th point.
            points[path] = points[path][::10,:3]
            print('Attention: Reduced amount of points for plotting.')
        if filtered_points[path].shape[0]>500000:
            # if plotting entire dataset and one dataset is larger than 500k points: only plot every 10th point.
            filtered_points[path] = filtered_points[path][::10,:3]
            print('Attention: Reduced amount of points for plotting.')

    return filtered_points, points


def plotly_2D_scatter(filtered_points, points = None, plot_type = None, save_path = None):
    '''
    plot the figure as 3d scatter
    '''
    import plotly.graph_objects as go
    import os
    _, color_schemes1,_ = get_colors_from_map(points)
    _, _, color_schemes2 = get_colors_from_map(filtered_points)
    

    scene_dict = {
        'eps': dict(xaxis_title='eps_x',
                    yaxis_title='eps_y',
                    zaxis_title='gamma_xy'),
        'sig':dict(xaxis_title='n_x',
                    yaxis_title='n_y',
                    zaxis_title='n_xy')
    }
    title_dict = {
        'eps': 'Visualisation Sampled Epsilon',
        'sig': 'Visualisation Sampled Sigma'
    }

    fig = go.Figure()
    for path1, path2 in zip(points.keys(), filtered_points.keys()):
        if points is not None: 
            r,g,b,_ = color_schemes1[path1]
            fig.add_trace(
                go.Scatter3d(
                    x=points[path1][:,0], y=points[path1][:,1], z=points[path1][:,2],
                    mode = 'markers',
                    marker = dict(
                        size = 2, 
                        opacity= 0.5,
                        color = f'rgb({r},{g},{b})',
                    ),
                    name = path1
                )
            )
        r,g,b,_ = color_schemes2[path2]
        fig.add_trace(
            go.Scatter3d(
                x=filtered_points[path2][:,0], y=filtered_points[path2][:,1], z=filtered_points[path2][:,2],
                mode = 'markers',
                marker = dict(
                    size = 2, 
                    opacity= 0.7,
                    color = f'rgb({r},{g},{b})',
                ),
                name = path2+'_filtered'
            )
        )


    fig.update_layout(
    scene=scene_dict[plot_type],
    title=title_dict[plot_type]
    )

    fig.show()

    if save_path is not None: 
        filename = "Membrane_epsilon_ScatterCloud.html"
        save_path = os.path.join(save_path, filename)

        fig.write_html(save_path)
        print(f"✅ Figure saved as interactive HTML: {save_path}")
    
    return  


def get_colors_from_map(inp_vector):
    cmap1, cmap2, cmap3 = plt.cm.gist_yarg, plt.cm.Blues, plt.cm.RdPu
    values = np.linspace(0.4,0.8,len(inp_vector.keys()))
    colors1, colors2, colors3 = {}, {}, {}
    for v, key in zip(values, inp_vector.keys()):
        colors1[key], colors2[key], colors3[key] = cmap1(v), cmap2(v), cmap3(v)
    return colors1, colors2, colors3


def get_outliers(path, clean_data):
    '''
    get outliers, i.e. points outside of the convex hull formed by "clean_data"
    '''
    from scipy.spatial import ConvexHull, Delaunay

    clean_pts = read_data(os.path.join('04_Training\\data\\', clean_data), 'sig')
    test_pts = {}
    for key in path:
        test_pts[key] = read_data(os.path.join('04_Training\\data\\', key), 'sig')[:,:3]

    hull = ConvexHull(clean_pts[:,:3])
    delaunay = Delaunay(clean_pts[:,:3][hull.vertices])

    outside_mask = {}
    for key in path:
        inside_mask = delaunay.find_simplex(test_pts[key]) >= 0
        outside_mask[key] = ~inside_mask
        print(f'Total amount of outliers / total points in dataset {key}: ',
              f'{np.sum(outside_mask[key])}/{test_pts[key].shape[0]}',
              f'= {(np.sum(outside_mask[key])/test_pts[key].shape[0]*100):.1f}%')

    return outside_mask


def save_data_without_outliers(outlier_mask, path_collection):
    '''
    saves data without outliers to same folder where all generated data is saved: 04_Training\data
    outlier_mask    (dict-bool)     mask for which points are outliers
    '''

    no_outlier_points={}

    for path in path_collection:
        for id in ['eps', 't', 'sig', 'De']:
            path_extended = os.path.join('04_Training\\data\\', path)
            data = read_data(path_extended, id)
            no_outlier_points[path] = data[~outlier_mask[path]]
            path_datasave = os.path.join("04_Training\\data", path+"_cleaned")
            os.makedirs(path_datasave, exist_ok=True)
            filename = os.path.join(path_datasave, 'new_data_'+id+'.pkl')
            with open(filename, "wb") as f:
                pickle.dump(no_outlier_points[path], f)
            print(f'Saved {filename}.')


    return


def find_corresponding_eps(sig, data_path):
    points = {}

    # extract data from data_path
    path_extended = os.path.join('04_Training\\data\\', data_path)
    data_sig = read_data(path_extended, id = 'sig')
    data_eps = read_data(path_extended, id = 'eps')
    points['sig'] = data_sig[:,:3]
    points['eps'] = data_eps[:,:3]

    # find correct indices of sig
    mask_all = np.zeros((points['sig'].shape[0], ), dtype = bool)
    for i in range(len(sig)): 
        mask = abs((abs(points['sig']) - abs(sig[i,:]))) < 0.001*np.ones((1,3))
        mask_i = mask.all(axis=1)
        mask_all |= mask_i

    print(f'Amount of epsilon rows: {sum(mask_all)}')

    # return eps at these indices

    return points['eps'][mask_all]


def get_loadsteps(path_depl): 

    subfolders = [name for name in os.listdir(path_depl)
              if os.path.isdir(os.path.join(path_depl, name))]

    load_steps = np.array(subfolders, dtype = np.float32)

    sorted_load_steps = load_steps[np.argsort(np.abs(load_steps))]

    return sorted_load_steps

def get_data_deployment(load_steps, path_depl, plot_type, tag = 'NN'):
    '''
    collect vectors for scatter plots of deployment data
    force_i         (int)       force level, corresponding to name of folder
    path_deployment (str-list)  list of paths of deployment data
    plot_type       (str)       either 'eps' or 'sig', depending on which data to extract.
    tag             (str)       either 'NN' or 'norm', depending on whether NLFEA or NN result is desired
    '''
    depl_data = {}

    for i, force_i in enumerate(load_steps):
        with open(os.path.join(path_depl+'\\'+str(int(force_i)), 'mat_res_'+tag+'.pkl'),'rb') as handle:
                    mat_res = pickle.load(handle)

        depl_data[str(force_i)] = mat_res[plot_type+'_g'][:,0,0,:3]
    
    combined_data = np.concatenate(list(depl_data.values()), axis=0)

    return combined_data

