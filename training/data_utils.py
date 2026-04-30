# vb, 25.03.2026

import h5py
import math
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
    path = os.path.join(save_data_path, 'output_' + name + '.h5')
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
    _plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(_plot_dir, exist_ok=True)
    plt.savefig(os.path.join(_plot_dir, filename + ".png"))
    print(f'Saved {filename} to {_plot_dir}{os.sep}{filename}.png')
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


# ══════════════════════════════════════════════════════════════════════════════
# Streaming / out-of-core data loading
# Use when the dataset is too large to fit in RAM (> ~50 GB).
# ══════════════════════════════════════════════════════════════════════════════

def get_dataset_size(path_data: str) -> int:
    """Return total number of samples by reading only the HDF5 header."""
    with h5py.File(os.path.join(path_data, 'output_eps_g.h5'), 'r') as f:
        return int(f['eps_g'].shape[0])


def get_streaming_splits(n_total: int,
                         test_size: float = 0.1,
                         eval_size: float = 0.2,
                         seed: int = SEED) -> tuple:
    """
    Return sorted index arrays for train / eval / test without loading any data.

    Indices are shuffled once so each split covers the full data range, then
    sorted so HDF5 reads are always sequential (fast I/O).

    Returns:
        idx_train, idx_eval, idx_test  —  np.ndarray, dtype int64, each sorted.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total).astype(np.int64)

    n_test = int(n_total * test_size)
    n_eval = int((n_total - n_test) * eval_size)

    idx_test  = np.sort(idx[:n_test])
    idx_eval  = np.sort(idx[n_test: n_test + n_eval])
    idx_train = np.sort(idx[n_test + n_eval:])

    print(f'[splits] train={len(idx_train)/1e6:.1f} M  '
          f'eval={len(idx_eval)/1e6:.1f} M  '
          f'test={len(idx_test)/1e6:.1f} M')
    return idx_train, idx_eval, idx_test


def compute_stats_from_sample(path_data: str,
                               sobolev: bool,
                               n_sample: int = 1_000_000,
                               seed: int = SEED) -> dict:
    """
    Compute normalisation statistics from a random subsample of the HDF5 dataset.
    Only n_sample rows are read — safe for arbitrarily large datasets.

    Returns a dict with keys 'stats_X_train' and 'stats_y_train', compatible
    with transform_data(), save_stats(), and test_utils throughout the pipeline.
    """
    n_total = get_dataset_size(path_data)
    rng = np.random.default_rng(seed)
    sample_idx = np.sort(
        rng.choice(n_total, size=min(n_sample, n_total), replace=False).astype(np.int64)
    )

    with h5py.File(os.path.join(path_data, 'output_eps_g.h5'), 'r') as f:
        X_s = f['eps_g'][sample_idx].astype(np.float32)
    with h5py.File(os.path.join(path_data, 'output_sig_g.h5'), 'r') as f:
        y_sig_s = f['sig_g'][sample_idx].astype(np.float32)

    if sobolev:
        with h5py.File(os.path.join(path_data, 'output_D.h5'), 'r') as f:
            y_D_s = f['D'][sample_idx].reshape(len(sample_idx), -1).astype(np.float32)
        y_s = np.concatenate([y_sig_s, y_D_s], axis=1)
    else:
        y_s = y_sig_s

    stats = {
        'stats_X_train': statistics(X_s),
        'stats_y_train': statistics(y_s),
    }
    print(f'[stats] Computed from {len(sample_idx) / 1e6:.2f} M sampled points.')
    return stats


def compute_stats_full_dataset(path_data: str,
                                sobolev: bool,
                                chunk_size: int = 1_000_000,
                                percentile_n_samples: int = 1_000_000,
                                seed: int = SEED) -> dict:
    """
    Compute normalisation statistics by scanning the *entire* HDF5 dataset in chunks.

    - mean and std  : exact, via Chan's parallel algorithm (numerically stable).
    - max and min   : exact, running min/max per chunk.
    - q_5 and q_95  : from a 1 M-point random reservoir (very accurate for 4e9 pts).
    - log_mean/std  : from the same reservoir.

    Only one chunk (~chunk_size rows) lives in RAM at a time.

    Returns a dict with keys 'stats_X_train' and 'stats_y_train', compatible
    with the rest of the pipeline (transform_data, save_stats, test_utils, …).
    """
    n_total = get_dataset_size(path_data)
    n_chunks = math.ceil(n_total / chunk_size)
    rng = np.random.default_rng(seed)

    # Pre-select reservoir indices (sorted so we can use searchsorted during
    # the single sequential pass below — no second HDF5 read needed).
    res_idx = np.sort(
        rng.choice(n_total, size=min(percentile_n_samples, n_total),
                   replace=False).astype(np.int64)
    )

    # ── single pass: exact mean / variance (Chan) + running max / min
    #                + reservoir rows collected on-the-fly via searchsorted ────
    count = 0
    mean_X = mean_y = M2_X = M2_y = None
    max_X  = max_y  = min_X = min_y = None
    res_X_list: list = []
    res_y_list: list = []

    print(f'[stats] Scanning {n_total / 1e6:.1f} M samples in {n_chunks} chunks ...')
    for ci in range(n_chunks):
        chunk_start = ci * chunk_size
        chunk_end   = min(chunk_start + chunk_size, n_total)
        n_c = chunk_end - chunk_start

        with h5py.File(os.path.join(path_data, 'output_eps_g.h5'), 'r') as f:
            X_c = f['eps_g'][chunk_start:chunk_end].astype(np.float64)
        with h5py.File(os.path.join(path_data, 'output_sig_g.h5'), 'r') as f:
            y_sig_c = f['sig_g'][chunk_start:chunk_end].astype(np.float64)
        if sobolev:
            with h5py.File(os.path.join(path_data, 'output_D.h5'), 'r') as f:
                y_D_c = f['D'][chunk_start:chunk_end].reshape(n_c, -1).astype(np.float64)
            y_c = np.concatenate([y_sig_c, y_D_c], axis=1)
        else:
            y_c = y_sig_c

        # Collect reservoir rows that fall inside this chunk (no random seeks).
        lo = int(np.searchsorted(res_idx, chunk_start))
        hi = int(np.searchsorted(res_idx, chunk_end))
        if lo < hi:
            local_idx = res_idx[lo:hi] - chunk_start
            res_X_list.append(X_c[local_idx].astype(np.float32))
            res_y_list.append(y_c[local_idx].astype(np.float32))

        # Chunk statistics
        mX_b = X_c.mean(axis=0);  M2X_b = ((X_c - mX_b) ** 2).sum(axis=0)
        my_b = y_c.mean(axis=0);  M2y_b = ((y_c - my_b) ** 2).sum(axis=0)

        if count == 0:
            mean_X, M2_X = mX_b.copy(), M2X_b.copy()
            mean_y, M2_y = my_b.copy(), M2y_b.copy()
            max_X, min_X = X_c.max(axis=0), X_c.min(axis=0)
            max_y, min_y = y_c.max(axis=0), y_c.min(axis=0)
        else:
            # Chan's parallel combination
            new_count = count + n_c
            dX = mX_b - mean_X
            mean_X += dX * n_c / new_count
            M2_X   += M2X_b + dX ** 2 * count * n_c / new_count

            dy = my_b - mean_y
            mean_y += dy * n_c / new_count
            M2_y   += M2y_b + dy ** 2 * count * n_c / new_count

            max_X = np.maximum(max_X, X_c.max(axis=0))
            min_X = np.minimum(min_X, X_c.min(axis=0))
            max_y = np.maximum(max_y, y_c.max(axis=0))
            min_y = np.minimum(min_y, y_c.min(axis=0))

        count += n_c
        if (ci + 1) % 500 == 0 or ci == n_chunks - 1:
            print(f'[stats]   {count / 1e6:.1f} M / {n_total / 1e6:.1f} M samples ...')

    std_X = np.sqrt(M2_X / (count - 1))
    std_y = np.sqrt(M2_y / (count - 1))

    # Assemble reservoir arrays (already collected during the pass above).
    X_res   = np.concatenate(res_X_list, axis=0) if res_X_list else np.empty((0, mean_X.shape[0]), dtype=np.float32)
    y_res   = np.concatenate(res_y_list, axis=0) if res_y_list else np.empty((0, mean_y.shape[0]), dtype=np.float32)
    print(f'[stats] Reservoir collected: {len(X_res) / 1e6:.2f} M points.')

    delta = 1e-8
    log_X   = np.log(X_res   + np.abs(min_X.astype(np.float32))   + delta)
    log_y   = np.log(y_res   + np.abs(min_y.astype(np.float32))   + delta)

    def _build_stat(mean, std, max_, min_, reservoir):
        return {
            'mean':     mean.astype(np.float32),
            'std':      std.astype(np.float32),
            'max':      max_.astype(np.float32),
            'min':      min_.astype(np.float32),
            'q_5':      np.percentile(reservoir, 5,  axis=0).astype(np.float32),
            'q_95':     np.percentile(reservoir, 95, axis=0).astype(np.float32),
            'log_mean': np.mean(np.log(reservoir + np.abs(min_.astype(np.float32)) + delta), axis=0).astype(np.float32),
            'log_std':  np.std( np.log(reservoir + np.abs(min_.astype(np.float32)) + delta), axis=0).astype(np.float32),
        }

    stats = {
        'stats_X_train': _build_stat(mean_X, std_X, max_X, min_X, X_res),
        'stats_y_train': _build_stat(mean_y, std_y, max_y, min_y, y_res),
    }
    print(f'[stats] Full-dataset statistics computed from {count / 1e9:.3f} B samples.')
    return stats


def load_test_sample(path_data: str,
                     sobolev: bool,
                     test_indices: np.ndarray,
                     n_sample: int = 500_000,
                     chunk_size: int = 1_000_000,
                     seed: int = SEED) -> dict:
    """
    Load a random subsample from the test split (raw, un-normalised).

    Uses a sequential chunked scan with searchsorted to avoid slow HDF5
    fancy indexing. Only rows matching `chosen` are kept per chunk.

    Returns {'X_test': np.ndarray, 'y_test': np.ndarray}, compatible with
    save_test_data() and test_NN_model().
    """
    rng = np.random.default_rng(seed)
    chosen = np.sort(
        rng.choice(test_indices, size=min(n_sample, len(test_indices)),
                   replace=False).astype(np.int64)
    )

    n_total = get_dataset_size(path_data)
    n_chunks = math.ceil(n_total / chunk_size)

    X_list: list = []
    y_sig_list: list = []
    y_D_list: list = []

    for ci in range(n_chunks):
        chunk_start = ci * chunk_size
        chunk_end   = min(chunk_start + chunk_size, n_total)

        lo = int(np.searchsorted(chosen, chunk_start))
        hi = int(np.searchsorted(chosen, chunk_end))
        if lo >= hi:
            continue

        local_idx = chosen[lo:hi] - chunk_start

        with h5py.File(os.path.join(path_data, 'output_eps_g.h5'), 'r') as f:
            X_list.append(f['eps_g'][chunk_start:chunk_end][local_idx].astype(np.float32))
        with h5py.File(os.path.join(path_data, 'output_sig_g.h5'), 'r') as f:
            y_sig_list.append(f['sig_g'][chunk_start:chunk_end][local_idx].astype(np.float32))
        if sobolev:
            with h5py.File(os.path.join(path_data, 'output_D.h5'), 'r') as f:
                n_c = hi - lo
                y_D_list.append(f['D'][chunk_start:chunk_end][local_idx].reshape(n_c, -1).astype(np.float32))

    X     = np.concatenate(X_list,     axis=0)
    y_sig = np.concatenate(y_sig_list, axis=0)
    if sobolev:
        y = np.concatenate([y_sig, np.concatenate(y_D_list, axis=0)], axis=1)
    else:
        y = y_sig

    print(f'[test-sample] Loaded {len(X) / 1e6:.2f} M test points.')
    return {'X_test': X, 'y_test': y}


def _std_normalise(data: np.ndarray, stat: dict) -> np.ndarray:
    """Zero-mean / unit-variance normalisation using pre-computed statistics."""
    std = np.where(stat['std'] == 0, 1.0, stat['std'])
    return (data - stat['mean']) / std


class HDF5StreamingDataset(torch.utils.data.IterableDataset):
    """
    Out-of-core dataset that streams (X, y) pairs from HDF5 files.

    Only one chunk of rows lives in RAM at a time — the full dataset is never
    loaded.  Normalisation is applied on-the-fly with pre-computed stats.
    Multi-worker DataLoader is supported: each worker automatically handles a
    disjoint slice of the index array.

    Args:
        path_data  (str):          Folder containing output_eps_g.h5,
                                   output_sig_g.h5, and output_D.h5.
        stats      (dict):         From compute_stats_from_sample(). Must have
                                   keys 'stats_X_train' and 'stats_y_train'.
        sobolev    (bool):         Include stiffness D in y if True.
        indices    (np.ndarray):   Sorted int64 row indices for this split.
        chunk_size (int):          Rows read per HDF5 access.  Default 500 000
                                   ≈ 96 MB per chunk for float32 features.
        shuffle    (bool):         Shuffle samples within each chunk.
                                   True for training, False for eval / test.
    """

    def __init__(self, path_data: str, stats: dict, sobolev: bool,
                 indices: np.ndarray, chunk_size: int = 500_000,
                 shuffle: bool = True):
        super().__init__()
        self.path_data  = path_data
        self.stats      = stats
        self.sobolev    = sobolev
        self.indices    = indices       # sorted int64 array
        self.chunk_size = chunk_size
        self.shuffle    = shuffle

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        # ── split work across DataLoader workers ──────────────────────────────
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            indices = self.indices
        else:
            per_worker = int(math.ceil(len(self.indices) / worker_info.num_workers))
            start      = worker_info.id * per_worker
            indices    = self.indices[start: start + per_worker]

        if len(indices) == 0:
            return

        # ── sequential scan with searchsorted ────────────────────────────────
        # HDF5 fancy indexing with large scattered index arrays is very slow
        # (essentially one seek per row). Instead we read contiguous slices and
        # pick out the matching rows in memory — same approach as compute_stats.
        # We only scan [indices[0], indices[-1]+1] to skip irrelevant regions.
        scan_start = int(indices[0])
        scan_end   = int(indices[-1]) + 1

        for chunk_start in range(scan_start, scan_end, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, scan_end)

            lo = int(np.searchsorted(indices, chunk_start))
            hi = int(np.searchsorted(indices, chunk_end))
            if lo >= hi:
                continue

            local_idx = indices[lo:hi] - chunk_start
            n_c = hi - lo

            with h5py.File(os.path.join(self.path_data, 'output_eps_g.h5'), 'r') as f:
                X = f['eps_g'][chunk_start:chunk_end][local_idx].astype(np.float32)
            with h5py.File(os.path.join(self.path_data, 'output_sig_g.h5'), 'r') as f:
                y_sig = f['sig_g'][chunk_start:chunk_end][local_idx].astype(np.float32)
            if self.sobolev:
                with h5py.File(os.path.join(self.path_data, 'output_D.h5'), 'r') as f:
                    y_D = f['D'][chunk_start:chunk_end][local_idx].reshape(n_c, -1).astype(np.float32)
                y = np.concatenate([y_sig, y_D], axis=1)
            else:
                y = y_sig

            X = _std_normalise(X, self.stats['stats_X_train'])
            y = _std_normalise(y, self.stats['stats_y_train'])

            if self.shuffle:
                perm = np.random.permutation(n_c)
                X, y = X[perm], y[perm]

            X_t = torch.from_numpy(X)
            y_t = torch.from_numpy(y)
            for i in range(n_c):
                yield X_t[i], y_t[i]