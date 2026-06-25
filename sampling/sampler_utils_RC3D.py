# vb, 06.03.2026

import os
import math
import numpy as np
import cupy as cp
import pyDOE as doe
from itertools import permutations
import time
import h5py

import matplotlib.pyplot as plt

try:
    from .concrete_classes import dict_CC
    from .simulating_sig_vec_RC3D import SigSimulator, sig_simulation_batchwise
except ImportError:
    from concrete_classes import dict_CC
    from simulating_sig_vec_RC3D import SigSimulator, sig_simulation_batchwise


########################################## Sampling strains ##########################################

def get_constant_sampling_params(sample_2d:bool) -> tuple:
    '''
    collect constant parameters for sampling (material parameters, ranges, ...)
    
    Args:
        sample_2d   (bool):    True for sampling in 2D

    Returns: 
        constants   (dict):    Containing relevant input parameters for sampling

    '''

    from constant_sampling_params import c, c_3D

    if not sample_2d:
        c.update(c_3D)

    
    # select values for concrete: 
    idx = dict_CC['CC'].index(c['CC'])
    dict_CC_one = {key: values[idx] for key, values in dict_CC.items()}

    dict_CC_one.update({'fsy': c["fsy"], 'fsu': c["fsu"], 'Es': c["Es"], 'Esh': c["Esh"], 'D': c["D"], 'Dmax': c["Dmax"], 's': c["s"]})
    mat_dict = dict_CC_one

    return c, mat_dict


def setup_sampler_dirs(constants: dict) -> tuple:
    """
    Apply the N_SAMPLES_3D env-var override (if set) and resolve the output
    directory from DATA_DIR (falls back to the local Windows path for dev runs).

    Args:
        constants (dict): sampling constants as returned by get_constant_sampling_params.

    Returns:
        constants     (dict): same dict, with n_samples_3D updated if overridden.
        save_data_dir (str):  resolved output directory (created if it does not exist).
    """
    if os.environ.get('N_SAMPLES_3D'):
        constants['n_samples_3D'] = int(float(os.environ['N_SAMPLES_3D']))
        print(f"[sampler] N_SAMPLES_3D override -> {constants['n_samples_3D']:,}")

    save_data_dir = os.environ.get('DATA_DIR') or os.path.join('D:\\', 'VeraBalmer', 'ShellSim3D')
    os.makedirs(save_data_dir, exist_ok=True)
    print(f'[sampler] Saving data to: {save_data_dir}')

    return constants, save_data_dir


def sample_eps(sampler:str, constants: dict, sampler_type_log = 'lhs', zero_value_epsx = 0.5e-6) -> np.array:
    """
    Samples 2D-strains.
    
    Args: 
        sampler             (str) : uniform, uniform_3D, log or LHS
        constants           (dict): material and geom constants, amount of samples
        sampler_type_log    (str):  if log-sampler, select sampler with which log-data is sampled (lhs or uniform_3D_grouped)
        zero_value_epsx     (float): value allocated to eps_x = 0 in case of log-sampling (mostly not == 0)

    Returns: 
        eps_l (np.arr): eps_layer for one layer (n_samples_2D, 3)
    
    """

    # Sample points

    if sampler == 'uniform':
        t0 = time.perf_counter()
        par_names = ['eps_x', 'eps_y', 'eps_xy']
        uniform_sampler = samplers(par_names, constants['min'], constants['max'], samples= constants['n_samples_2D'])
        data = uniform_sampler.uniform()
        print(f'Sampled {int(constants["n_samples_2D"]/1e9)}*1e9 values for 2D-epsilon')
        t_elapsed = time.perf_counter() - t0
        print(f'2D Sampling done in {t_elapsed/60:.2f}min')

    elif sampler == 'uniform_3D':
        t0 = time.perf_counter()
        par_names = ['eps_x', 'eps_y', 'eps_xy', 'chi_x', 'chi_y', 'chi_xy']
        uniform_sampler = samplers(par_names, constants['min'], constants['max'], samples= constants['n_samples_3D'])
        data = uniform_sampler.uniform_multi()
        print(f'Sampled {int(constants["n_samples_3D"]/1e9)}*1e9 values for 3D-epsilon')
        t_elapsed = time.perf_counter() - t0
        print(f'3D Sampling done in {t_elapsed/60:.2f}min')
    
    elif sampler == 'uniform_3D_grouped':
        t0 = time.perf_counter()
        par_names = ['eps_x', 'eps_y', 'eps_xy', 'chi_x', 'chi_y', 'chi_xy']
        uniform_sampler = samplers(par_names, constants['min'], constants['max'], samples= constants['n_samples_3D'])
        data = uniform_sampler.uniform_multi_grouped()
        print(f'Sampled {int(constants["n_samples_3D"]/1e9)}*1e9 values for 3D-epsilon')
        t_elapsed = time.perf_counter() - t0
        print(f'3D Sampling done in {t_elapsed/60:.2f}min')

    elif sampler == 'lhs':
        t0 = time.perf_counter()
        par_names = ['eps_x', 'eps_y', 'eps_xy', 'chi_x', 'chi_y', 'chi_xy']
        lhs_sampler = samplers(par_names, constants['min'], constants['max'], samples= constants['n_samples_3D'])
        data = lhs_sampler.lhs(criterion = 'c')
        print(f'Sampled {int(constants["n_samples_3D"]/1e9)}*1e9 values for 3D-epsilon')
        t_elapsed = time.perf_counter() - t0
        print(f'3D Sampling with LHS done in {t_elapsed/60:.2f}min')

    elif sampler == 'combined_lhs_uniform':
        half_samples = int(constants['n_samples_3D']/2)

        t0 = time.perf_counter()
        par_names = ['eps_x', 'eps_y', 'eps_xy', 'chi_x', 'chi_y', 'chi_xy']
        lhs_sampler = samplers(par_names, constants['min'], constants['max'], samples= half_samples)
        data_0 = lhs_sampler.lhs(criterion = 'c')
        print(f'Sampled {int(half_samples/1e9)}*1e9 values for 3D-epsilon')
        t_elapsed_0 = time.perf_counter() - t0
        print(f'3D Sampling with LHS done in {t_elapsed_0/60:.2f}min')

        t1 = time.perf_counter()
        uniform_sampler = samplers(par_names, constants['min'], constants['max'], samples= half_samples)
        data_1 = uniform_sampler.uniform_multi_grouped()
        print(f'Sampled {int(half_samples/1e9)}*1e9 values for 3D-epsilon')
        t_elapsed_1 = time.perf_counter() - t1
        print(f'3D Sampling uniformly done in {t_elapsed_1/60:.2f}min')

        data = np.concatenate((data_0, data_1), axis = 0)

    elif sampler == 'log-sampler':
        data_, max_log_neg, max_log_pos = sample_exponents(constants, sampler_type_log, zero_value_epsx)
        data = convert_log_data_to_eps(data_, max_log_neg, max_log_pos)                                                                               # TODO: Write function

    elif sampler == 'combined_log_uniform':
        if constants['p_samples_log'] is not None:
            log_samples = int(constants['n_samples_3D']*constants['p_samples_log'])
            uniform_samples = int(constants['n_samples_3D']*(1-constants['p_samples_log']))
        else: 
            log_samples = int(constants['n_samples_3D']/2)
            uniform_samples = log_samples.copy()

        data_, max_log_neg, max_log_pos = sample_exponents(constants, sampler_type_log, zero_value_epsx, num_samples = log_samples)
        data_0 = convert_log_data_to_eps(data_, max_log_neg, max_log_pos) 

        t1 = time.perf_counter()
        par_names = ['eps_x', 'eps_y', 'eps_xy', 'chi_x', 'chi_y', 'chi_xy']
        uniform_sampler = samplers(par_names, constants['min'], constants['max'], samples= uniform_samples)
        data_1 = uniform_sampler.uniform_multi_grouped()
        print(f'Sampled {int(uniform_samples/1e9)}*1e9 values for 3D-epsilon')
        t_elapsed_1 = time.perf_counter() - t1
        print(f'3D Sampling uniformly done in {t_elapsed_1/60:.2f}min')

        data = np.concatenate((data_0, data_1), axis = 0)

    elif sampler == 'combined_log_lhs':
        if constants['p_samples_log'] is not None:
            log_samples = int(constants['n_samples_3D']*constants['p_samples_log'])
            uniform_samples = int(constants['n_samples_3D']*(1-constants['p_samples_log']))
        else: 
            log_samples = int(constants['n_samples_3D']/2)
            uniform_samples = log_samples.copy()

        data_, max_log_neg, max_log_pos = sample_exponents(constants, sampler_type_log, zero_value_epsx, num_samples = log_samples)
        data_0 = convert_log_data_to_eps(data_, max_log_neg, max_log_pos) 

        t1 = time.perf_counter()
        par_names = ['eps_x', 'eps_y', 'eps_xy', 'chi_x', 'chi_y', 'chi_xy']
        lhs_sampler = samplers(par_names, constants['min'], constants['max'], samples= uniform_samples)
        data_1 = lhs_sampler.lhs(criterion = 'c')
        print(f'Sampled {int(uniform_samples/1e9)}*1e9 values for 3D-epsilon')
        t_elapsed = time.perf_counter() - t1
        print(f'3D Sampling with LHS done in {t_elapsed/60:.2f}min')

        data = np.concatenate((data_0, data_1), axis = 0)

    
    else: 
        raise UserWarning('This has not yet been implemented.')
    

    # Check smallest value and increase if required
    data = filter_small_epsilon(data)

    return data


def convert_log_data_to_eps(data_, max_log_neg, max_log_pos):
    """
    Converts sampled exponents (data_) into values of epsilon.
    
    Args: 
        data_       (np.arr): Contains sampled exponents
    
    Returns: 
        data        (np.arr): Contains values of epsilon
    """
    exp = 10

    # calculate real epsilon values (absolute values)
    eps_no_sign = exp**(data_)

    # get mask and assign (+) or (-) sign to epsilon values
    mask_sign = ~((data_ < max_log_pos) & (data_ > max_log_neg))        # see sketch hand notes, 27.04.2026
    sign_eps = np.random.randint(0,2, size = eps_no_sign.shape)
    sign_eps[sign_eps==0] = -1

    # collect epsilon values with added sign into one data array
    data = eps_no_sign.copy()
    data[mask_sign] = sign_eps[mask_sign]*eps_no_sign[mask_sign]


    return data

def sample_exponents(constants, sampler_type_log, zero_value_epsx, num_samples = None):
    """
    sampling exponent instead of epsilon directly. 
    """

    t = constants['t']
    zero_vec = [zero_value_epsx]*2+[zero_value_epsx/10]+[zero_value_epsx/(t/2)]*2 + [(zero_value_epsx/10)/(t/2)]

    min = constants['min_log']
    max = constants['max_log']
    max_log_neg = np.log10([abs(x) for x in min])
    max_log_pos = np.log10([abs(x) for x in max])

    min_log = np.log10(zero_vec)
    max_log = np.maximum(max_log_neg, max_log_pos)


    if num_samples is not None:
        n_samples = num_samples
    else: 
        n_samples = constants['n_samples_3D']


    t0 = time.perf_counter()
    par_names = ['eps_x', 'eps_y', 'eps_xy', 'chi_x', 'chi_y', 'chi_xy']
    if sampler_type_log == 'lhs': 
        lhs_sampler = samplers(par_names, min_log, max_log, samples= n_samples)
        data = lhs_sampler.lhs(criterion = 'c')
        print(f'Sampled {int(n_samples/1e9)}*1e9 values for 3D-epsilon')
        t_elapsed = time.perf_counter() - t0
        print(f'3D Sampling exponents with LHS done in {t_elapsed/60:.2f}min')
    elif sampler_type_log == 'uniform_3D_grouped': 
        uniform_sampler = samplers(par_names, min_log, max_log, samples= n_samples)
        data = uniform_sampler.uniform_multi_grouped()
        print(f'Sampled {int(n_samples/1e9)}*1e9 values for 3D-epsilon')
        t_elapsed = time.perf_counter() - t0
        print(f'3D Sampling exponents uniformly done in {t_elapsed/60:.2f}min')
    else:   
        raise UserWarning('Please choose either lhs or uniform_3D_grouped as sampler for log sampling.')
    
    return data, max_log_neg, max_log_pos

def filter_small_epsilon(data, threshold = 1e-10):
    """
    filters out small values of epsilon to avoid large stiffness values. 

    """
    mask = abs(data) < threshold
    print(f'Detected {np.sum(mask)} points below threshold of {threshold}. These values are set to {threshold} in the dataset.')
    data[mask] = threshold
    

    return data

class samplers:
    def __init__(self, parnames, min, max, samples):
        self.parnames = parnames
        self.min = min
        self.max = max
        self.samples = samples

    def lhs(self, criterion):
        """
        Returns LHS samples.

        Args:
            min        list:        List of lower bounds
            max        list:        List of upper bounds 
            samples     int:        Amount of samples
            criterion   str:        A string that tells lhs how to sample the points. See docs for pyDOE.lhs().
        
        Returns:
            points      np.arr:    Sampled points
        """
        dim = len(self.min)
        n_i = int(np.round((self.samples)**(1/dim), 0))

        bounds = np.vstack((self.min, self.max))
        bounds = bounds.T
        

        lhs = doe.lhs(dim, samples=n_i**dim, criterion=criterion)
        par_vals = np.zeros((n_i**dim,dim))
        for i in range(dim):
            par_min = bounds[i][0]
            par_max = bounds[i][1]
            par_vals[:,i] = np.array(lhs[:, i]) * (par_max - par_min) + par_min

        points = np.array(par_vals)

        return points
    
    def uniform(self):
        n_i = int(np.round((self.samples)**(1/3),0))
        
        x = np.linspace(self.min[0], self.max[0], n_i)
        y = np.linspace(self.min[1], self.max[1], n_i)
        z = np.linspace(self.min[2], self.max[2], n_i)

        X,Y,Z = np.meshgrid(x,y,z,indexing = 'ij')
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        return points
    
    def uniform_multi(self):
        n_dims = len(self.min)
        n_i = int(np.round((self.samples)**(1/n_dims), 0))
        
        axes = [np.linspace(self.min[i], self.max[i], n_i) for i in range(n_dims)]
        grids = np.meshgrid(*axes, indexing='ij')
        points = np.column_stack([g.ravel() for g in grids])
        
        return points

    def uniform_multi_grouped(self, group_size=3):
        n_dims = len(self.min)
        n_groups = n_dims // group_size
        n_i = int(np.round((self.samples)**(1/n_dims), 0))

        group_grids = []
        for g in range(n_groups):
            start0 = g * group_size
            end0 = start0 + group_size
            axes = [np.linspace(self.min[i], self.max[i], n_i) for i in range(start0, end0)]
            grids = np.meshgrid(*axes, indexing='ij')
            group_points = np.column_stack([gr.ravel() for gr in grids])  # (40^3, 3)
            group_grids.append(group_points)

        # Efficient cartesian product via repeat/tile instead of meshgrid on indices
        g0, g1 = group_grids[0], group_grids[1]
        n0, n1 = len(g0), len(g1)

        # print(f'g0 shape: {g0.shape}')  # should be (40^3, 3) = (64000, 3)
        # print(f'g1 shape: {g1.shape}')  # should be (40^3, 3) = (64000, 3)
        # print(f'g1 unique values per dim:')
        # for j in range(3):
        #     print(f'  dim {j}: {len(np.unique(g1[:,j]))} unique values')

        points = np.empty((n0 * n1, n_dims), dtype=np.float32)
        points[:, :group_size] = np.repeat(g0, n1, axis=0)   # repeat each row of g0 n1 times
        points[:, group_size:] = np.tile(g1, (n0, 1))         # tile g1 n0 times

        # print(f'After tile, unique values in last dim: {len(np.unique(points[:, -1]))}')

        return points


def permute_eps_2D(n_perm: int, eps_l: np.array) -> np.array:
    """
    Creates pairs of top and bottom strains by permutation
    
    Args:
        n_perm  (int)   : amount of permutations 
        eps_l   (np.arr): strains per layer (n_samples_2D, 3)
    
    Returns: 
        eps_l_top_bot (np.arr): strains per layer at top and bottom of element (n_tot, 3, 2)

    """

    n_samples = eps_l.shape[0]
    
    # Create n_perm shuffled versions of eps_l
    shuffled = []
    for _ in range(n_perm):
        eps_copy = eps_l.copy()
        np.random.shuffle(eps_copy)
        shuffled.append(eps_copy)  # each: (n_samples, 3)
    
    print('Created shuffled 2D eps vectors')

    # All ordered pairs (i, j) where i != j
    all_pairs = list(permutations(range(n_perm), 2))  # n_perm * (n_perm - 1) pairs

    n_tot         = n_samples * len(all_pairs)
    eps_l_top_bot = np.zeros((n_tot, 3, 2))

    for k, (i, j) in enumerate(all_pairs):
        start = k * n_samples
        end   = start + n_samples
        eps_l_top_bot[start:end, :, 0] = shuffled[i]  # top
        eps_l_top_bot[start:end, :, 1] = shuffled[j]  # bot

    print('Created pairs of top-bottom 2D eps vectors')

    return eps_l_top_bot


def permute_eps_2D_batched(n_perm: int, eps_l: np.array, batch_size: int = 50):
    """
    Generator that yields batches of top-bottom strain pairs.
    Use this when the full output is too large to fit in memory.

    Yields:
        batch (np.arr): (n_samples * batch_size, 3, 2)
    """
    n_samples = eps_l.shape[0]
    indices   = [np.random.permutation(n_samples) for _ in range(n_perm)]
    all_pairs = list(permutations(range(n_perm), 2))

    print('Created indices for shuffled 2D eps vectors')

    for batch_start in range(0, len(all_pairs), batch_size):
        batch_pairs = all_pairs[batch_start : batch_start + batch_size]
        batch       = np.empty((n_samples * len(batch_pairs), 3, 2), dtype=eps_l.dtype)

        for k, (i, j) in enumerate(batch_pairs):
            start = k * n_samples
            end   = start + n_samples
            batch[start:end, :, 0] = eps_l[indices[i]]  # top
            batch[start:end, :, 1] = eps_l[indices[j]]  # bot

        print(f'Yielding batch {batch_start // batch_size + 1} / {len(all_pairs) // batch_size + 1}, with batchsize {batch_size}')
        yield batch


def get_all_eps_2D(eps_l_top_bot: np.array, n_layer: int) -> np.array:
    """
    Creates array of strains per layer for all layers

    Args: 
        eps_l_top_bot (np.arr): strains per layer at top and bottom of element (n_tot, 3, 2)
        n_layer       (int)   : amount of layers in shell element

    Returns: 
        eps_l_all (np.arr): strains per layer (n_tot, 3, 20)
 
    """
    
    weights = np.linspace(0.0, 1.0, n_layer)
    
    eps_bottom = eps_l_top_bot[:, :, 0]  # (n_tot, 3)
    eps_top    = eps_l_top_bot[:, :, 1]  # (n_tot, 3)
    
    eps_l_all = eps_bottom[:, :, np.newaxis] + weights * (eps_top - eps_bottom)[:, :, np.newaxis]

    return eps_l_all


def get_eps(eps_l_all:np.array, eps_l_top_bot: np.array, n_layer:int, t:int) -> np.array:
    """
    Creates array of strains per layer for all layers

    Args: 
        eps_l_all     (np.arr): strains per layer in all layers (n_tot, 3, 20)
        eps_l_top_bot (np.arr): strains per layer at top and bottom of element (n_tot, 3, 2)
        n_layer       (int)   : number of layers
        t             (int)   : thickness        

    Returns: 
        eps_g (np.arr): generalised strains (n_tot, 6)
 
    """
    eps_mid = (eps_l_all[:, :, 9] + eps_l_all[:, :, 10]) / 2
    eps_top    = eps_l_top_bot[:, :, 1]  # (n_tot, 3)
    eps_bottom = eps_l_top_bot[:, :, 0]  # (n_tot, 3)

    z = t/2-(t/n_layer)
    chi_all = (1/z)*(np.maximum(eps_top, eps_bottom)-eps_mid)

    eps_g = np.hstack((eps_mid, chi_all))

    return eps_g


def permute_and_save(eps_l: np.array, constants:dict, save_dir, save_batchwise: bool = False) -> None:
    """
    Permutes and interpolates 2D eps data and generates 3d eps data.

    Args:
        eps_l       (np.arr): Sampled 2D eps data
        constants   (dict):   constants dict vector
    
    Returns: 
        Saved files in "save_dir"
    
    """

    n_perm = int(np.sqrt(constants['n_samples_3D']/constants['n_samples_2D']))

    if save_batchwise:
        for k, batch in enumerate(permute_eps_2D_batched(n_perm, eps_l)):
            # calculate per-layer strains
            t0 = time.perf_counter()
            eps_l_all = get_all_eps_2D(batch, constants['n_layer'])
            print(f'time get_all_eps_2D: {time.perf_counter()-t0:.2f}s')
            
            # calculate generalised strains
            t1 = time.perf_counter()
            eps_g     = get_eps(eps_l_all, batch, constants['n_layer'], constants['t'])
            print(f'time get_eps: {time.perf_counter()-t1:.2f}s')

            # save
            t2 = time.perf_counter()
            with h5py.File(os.path.join(save_dir,f'output_eps_g_batch_{k}.h5'), 'w') as f:
                f.create_dataset('eps_g',     data = eps_g,     dtype='float32')
            print(f'time save_eps: {time.perf_counter()-t2:.2f}s')

            t_elapsed = time.perf_counter() - t0
            print(f'Batch {k+1} done in {t_elapsed/60:.2f}min')
    else: 
        start = 0
        with h5py.File(os.path.join(save_dir,f'output_eps_g.h5'), 'w') as f:
            ds_eps_g     = f.create_dataset('eps_g', shape=(constants['n_samples_3D'], 6), dtype='float32')

            for k, batch in enumerate(permute_eps_2D_batched(n_perm, eps_l)):
                # calculate per-layer strains
                t0 = time.perf_counter()
                eps_l_all = get_all_eps_2D(batch, constants['n_layer'])
                print(f'time get_all_eps_2D: {time.perf_counter()-t0:.2f}s')
                
                # calculate generalised strains
                t1 = time.perf_counter()
                eps_g     = get_eps(eps_l_all, batch, constants['n_layer'], constants['t'])
                print(f'time get_eps: {time.perf_counter()-t1:.2f}s')

                # save
                t2 = time.perf_counter()
                end = start + eps_g.shape[0]
                ds_eps_g[start:end]     = eps_g
                print(f'time save_eps: {time.perf_counter()-t2:.2f}s')

                start = end
                t_elapsed = time.perf_counter() - t0
                print(f'Batch {k+1} done in {t_elapsed/60:.2f}min')

    
    print(f'Done — saved {k+1} batches to {save_dir}')


def save_3D_data(data_:np.array, save_dir:str, filename:str):
    t2 = time.perf_counter()
    with h5py.File(os.path.join(save_dir,f'output_'+ filename +'.h5'), 'w') as f:
        f.create_dataset(filename,     data = data_,     dtype='float32')
    print(f'time save_data {filename}: {(time.perf_counter()-t2)/60:.2f}min')


def save_3D_data_append(data_: np.ndarray, save_dir: str, filename: str):
    """
    Append rows to an HDF5 dataset, creating it (with an unlimited first axis)
    on the first call and resizing on every subsequent call.

    Replaces save_3D_data() for the chunked sampling pipeline so that only
    one chunk of data needs to be in RAM at a time.

    Args:
        data_     (np.ndarray): Chunk to append. Shape (n_chunk, ...).
        save_dir  (str):        Directory containing the HDF5 files.
        filename  (str):        Dataset name, e.g. 'eps_g', 'sig_g', 'D'.
    """
    t0   = time.perf_counter()
    path = os.path.join(save_dir, f'output_{filename}.h5')
    data_f32 = np.asarray(data_).astype(np.float32)

    with h5py.File(path, 'a') as f:
        if filename in f:
            ds       = f[filename]
            old_size = ds.shape[0]
            ds.resize(old_size + data_f32.shape[0], axis=0)
            ds[old_size:] = data_f32
        else:
            maxshape = (None,) + data_f32.shape[1:]
            f.create_dataset(filename, data=data_f32,
                             maxshape=maxshape, dtype='float32',
                             chunks=True)

    print(f'[save] appended {data_f32.shape[0]:,} rows to {filename}.h5  '
          f'({(time.perf_counter()-t0)/60:.2f} min)')


def run_chunked_sampling(
    constants: dict,
    mat_dict: dict,
    save_data_dir: str,
    simulatesig,
    cm: int,
    sampling_type: str,
    chunk_size: int,
    filter_data: bool = True,
    save_D: bool = True,
    remove_outliers: bool = True,
) -> tuple:
    """
    Full chunked sampling loop: sample → simulate → filter → append to HDF5.

    Iterates over the total requested samples in chunks of `chunk_size` so that
    only ~1.7 GB of CPU RAM is needed at any one time, regardless of the total
    dataset size.  GPU memory is freed after every chunk.

    Stale output HDF5 files are removed at the start so appending always begins
    from an empty file.

    Args:
        constants     (dict):  sampling constants (must contain 'n_samples_3D').
        mat_dict      (dict):  material parameter dict.
        save_data_dir (str):   directory where the HDF5 files are written.
        simulatesig          : SigSimulator instance.
        cm            (int):   cross-section model index passed to the simulator.
        sampling_type (str):   sampler name, e.g. 'combined_lhs_uniform'.
        chunk_size    (int):   number of samples per chunk.
        filter_data   (bool):  if True, apply filter_3d_data to each chunk.
        save_D        (bool):  if True, also append the stiffness tensor D to disk.

    Returns:
        eps_g_last (np.ndarray | None): filtered strains from the last chunk.
        sig_g_last (np.ndarray | None): filtered stresses from the last chunk.
        D_last     (np.ndarray | None): filtered stiffness from the last chunk.
        All three are None if n_total == 0.
    """
    n_total  = constants['n_samples_3D']
    n_chunks = math.ceil(n_total / chunk_size)

    # Remove any stale output files so appending always starts clean.
    for _name in ['eps_g', 'sig_g', 'D']:
        _p = os.path.join(save_data_dir, f'output_{_name}.h5')
        if os.path.exists(_p):
            os.remove(_p)
            print(f'[sampler] Removed existing {_name}.h5')

    eps_g_last = sig_g_last = D_last = None

    for chunk_i in range(n_chunks):
        chunk_n = min(chunk_size, n_total - chunk_i * chunk_size)
        print(f'\n[sampler] ── Chunk {chunk_i + 1}/{n_chunks}  '
              f'({chunk_n / 1e6:.1f} M samples) ──')

        # 1 - Sample strains
        constants_chunk = {**constants, 'n_samples_3D': chunk_n}
        eps_g = sample_eps(sampler=sampling_type, constants=constants_chunk)
        eps_g[:, 2] = eps_g[:, 2] * 2          # eps_xy → gamma_xy

        # 2 - Filter
        if filter_data:
            eps_g_f = filter_3d_data(eps_g, constants = constants, prefilter=True)

        # 3 - Simulate stresses & stiffness
        n_sub = max(1, int(chunk_n) // 1_000_000)
        sig_g, dh = sig_simulation_batchwise(
            cp.asarray(eps_g), simulatesig, cm, mat_dict, n_batches=n_sub
        )

        eps_g_f, sig_g_f, D_f = eps_g, sig_g, dh

        # 4 - Remove outliers of D:
        if remove_outliers:
            _, _, mask_outliers = find_outlier_d(D_f, eps_g_f, 1000)
            eps_g_f = eps_g_f[~mask_outliers]
            sig_g_f = sig_g_f[~mask_outliers]
            D_f = D_f[~mask_outliers]
            print(f'Removed {np.sum(mask_outliers)} outliers from dataset ({np.sum(mask_outliers)/eps_g_f.shape[0]*100:.3f} %)')


        # 4 - Append to HDF5
        save_3D_data_append(eps_g_f, save_data_dir, filename='eps_g')
        save_3D_data_append(sig_g_f, save_data_dir, filename='sig_g')
        if save_D:
            save_3D_data_append(D_f, save_data_dir, filename='D')

        # Keep the last chunk in memory for post-loop visualisation.
        eps_g_last, sig_g_last, D_last = eps_g_f, sig_g_f, D_f

        # Free GPU and CPU memory before the next chunk.
        del eps_g, sig_g, dh, eps_g_f, sig_g_f, D_f
        cp.get_default_memory_pool().free_all_blocks()

    return eps_g_last, sig_g_last, D_last


########################################## Visualising strains, stresses ##########################################

def plot_3D_data(save_data_path, filename, n_every: int = int(1e3)):
    """
    visualise sampled strains

    Args:
        save_data_path (str): location where to save the plot
        filename (str): Either "scatter_eps_g" or "scatter_sig_g"     

    """
    
    
    fig = plt.figure(figsize=(14, 7))
    
    t0 = time.perf_counter()
    data = read_h5_file(save_data_path, filename, n_every)
    print(f'time reading file: {(time.perf_counter()-t0)/60:.2f}min')

    for i in range(2):
        t1 = time.perf_counter()
        x = data[:,i*3]
        y = data[:,i*3+1]
        z = data[:,i*3+2]
        print(f'Plotting {len(x)/1e6}*1e6/{len(x)/(1e6)*n_every}*1e6 points')
        
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        ax.scatter(x, y, z, s=2, alpha=0.1)
        figure_formatting(ax, i, filename)
        print(f'time plotting: {(time.perf_counter()-t1)/60:.2f}min')

    t2 = time.perf_counter()
    plt.tight_layout()
    _plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(_plot_dir, exist_ok=True)
    plt.savefig(os.path.join(_plot_dir, filename + ".png"))
    print(f'Saved {filename} to {_plot_dir}{os.sep}{filename}.png')
    print(f'time saving figure: {(time.perf_counter()-t2)/60:.2f}min')

    return


def read_h5_file(save_data_path, filename, n_every:int) -> tuple:
    name = filename[-5:]
    with h5py.File(save_data_path, 'r') as f:
        data = f[name][::n_every,:]

    # this is quite slow...
    # with h5py.File(save_data_path, 'r') as f:
    #     n_total = f['eps_g'].shape[0]
    #     idx = np.sort(np.random.choice(n_total, size=n_total//n_every, replace=False))
    #     data = f['eps_g'][idx, :]

    return data


def figure_formatting(ax, i, filename):
    if 'eps' in filename:
        if i == 0:
            ax.set_xlabel('eps_x')
            ax.set_ylabel('eps_y')
            ax.set_zlabel('gamma_xy')
        elif i == 1:
            ax.set_xlabel('chi_x')
            ax.set_ylabel('chi_y')
            ax.set_zlabel('2*chi_xy')

    elif 'sig' in filename: 
        if i == 0:
            ax.set_xlabel('n_x')
            ax.set_ylabel('n_y')
            ax.set_zlabel('n_xy')
        elif i == 1:
            ax.set_xlabel('m_x')
            ax.set_ylabel('m_y')
            ax.set_zlabel('m_xy')


########################################## Visualising stiffnesses ##########################################

def plot_filtered_stiffness(data_eps, data_D, idx_eps, save_path, remove_outliers: bool = False):
    """
    Plots filtered versions of stiffness data

    Args: 
        data_eps    (np.arr):   to create the mask according to which the D-data is filtered, shape: (ntot, 6)
        data_D      (np.arr):   data for plotting, shape: (ntot, 6,6)
        idx_eps     (int):      Non-zero element of epsilon
        save_path   (str):      Location where to save plot

    Returns: 
        plot containing idx_eps on x-axis and all corresponding stiffnesses on y-axis
    """


    # filter data
    data_f_eps,mask = get_mask_strain(data_eps, idx_eps)
    data_f_D = data_D[mask]

    # sort data
    data_s_eps, data_s_D = sort_data(data_f_eps, data_f_D, idx_eps)

    # deduplicate data (if still two values left inside tolerance, always use the first one):
    data_d_eps, data_d_D = deduplicate_by_eps(data_s_eps, data_s_D, idx_eps)
    if not remove_outliers:
        # otherwise I will plot a random subset of the outliers.
        data_d_D_outlier, data_d_eps_outlier, _ = find_outlier_d(data_d_D, data_d_eps, 1000)
    else: 
        data_d_D_outlier, data_d_eps_outlier = None, None

    # plot data
    plot_data_stiffness(data_d_eps, data_d_D, data_d_D_outlier, data_d_eps_outlier, idx_eps,save_path)

    return

def deduplicate_by_eps(data_s_eps, data_s_D, idx_eps, decimals=6):
    x = np.round(data_s_eps[:, idx_eps], decimals=decimals)
    unique_x = np.unique(x)

    unique_eps = []
    unique_D = []
    for val in unique_x:
        group_mask = x == val
        unique_eps.append(data_s_eps[group_mask][0])
        unique_D.append(data_s_D[group_mask][0])

    return np.array(unique_eps), np.array(unique_D)

def get_mask_strain(data_eps, idx_eps, tol = [0.5e-3, 0.5e-3, 0.9e-3, 0.4e-5, 0.4e-5, 0.4e-5]):
    # tol for 6 points per direction: [0.5e-3, 0.5e-3, 1.6e-3, 0.5e-5, 0.5e-5, 0.7e-5]
    tol = np.array(tol)
    cols = np.arange(data_eps.shape[1])!=idx_eps
    mask = np.all(np.abs(data_eps[:,cols])<tol[cols], axis =1)

    data_f_eps = data_eps[mask]

    print(f'After filtering data for plotting D, {mask.sum()} datapoints are left.')
    if mask.sum() < 1:
        raise UserWarning('No datapoints found in given range. Please change the filtering tolerance.')

    return data_f_eps, mask

def sort_data(data_f_eps, data_f_D, idx_eps):
    """
    sorts data in ascending order according to idx_eps values

    Args:
        data_f_eps  (np.arr): filtered eps-data
        data_f_D    (np.arr): filtered D-data
        idx_eps     (int):    index for which to sort

    """
    data_s_eps = data_f_eps[np.argsort(data_f_eps[:, idx_eps])]
    data_s_D = data_f_D[np.argsort(data_f_eps[:, idx_eps])]

    return data_s_eps, data_s_D

def plot_data_stiffness(data_s_eps, data_s_D, data_d_D_outlier, data_d_eps_outlier, idx_eps, save_path):

    fig, axs = plt.subplots(6,6, figsize = [30,20])

    for i in range(6): 
        for j in range(6): 
            axs[i,j].plot(data_s_eps[:,idx_eps], data_s_D[:,i,j], marker = 'o')
            if data_d_eps_outlier is not None:
                axs[i,j].plot(data_d_eps_outlier[:,idx_eps], data_d_D_outlier[:,i,j], marker = 'o', color = 'coral')

    figure_formatting_D(axs, idx_eps)

    if save_path is not None: 
        filename = 'filtered_dataset_D.png'
        fig.savefig(os.path.join(save_path, filename))
        print(f'Saved {filename} to {save_path}')


    return

def figure_formatting_D(axs, idx_eps):
    names_D = np.array([['$D_{m,11}$', '$D_{m,12}$', '$D_{m,13}$', '$D_{mb,11}$', '$D_{mb,12}$', '$D_{mb,13}$'],
                        ['$D_{m,21}$', '$D_{m,22}$', '$D_{m,23}$', '$D_{mb,21}$', '$D_{mb,22}$', '$D_{mb,23}$'],
                        ['$D_{m,31}$', '$D_{m,32}$', '$D_{m,33}$', '$D_{mb,31}$', '$D_{mb,32}$', '$D_{mb,33}$'],
                        ['$D_{bm,11}$', '$D_{bm,12}$', '$D_{bm,13}$', '$D_{b,11}$', '$D_{b,12}$', '$D_{b,13}$'],
                        ['$D_{bm,21}$', '$D_{bm,22}$', '$D_{bm,23}$', '$D_{b,21}$', '$D_{b,22}$', '$D_{b,23}$'],
                        ['$D_{bm,31}$', '$D_{bm,32}$', '$D_{bm,33}$', '$D_{b,31}$', '$D_{b,32}$', '$D_{b,33}$'],
                        ])
    names_eps = np.array(['$\\varepsilon_x$', '$\\varepsilon_y$', '$\\gamma_{xy}$', '$\\chi_x$', '$\\chi_y$', '$\\chi_{xy}$'])
    
    for i in range(6):
        for j in range(6):
            axs[i,j].set_ylabel(names_D[i,j])
            axs[i,j].set_xlabel(names_eps[idx_eps])

    return

def imshow_D_filtered(data_eps, data_D, idx_eps, save_path): 
    # filter data
    data_f_eps,mask = get_mask_strain(data_eps, idx_eps)
    data_f_D = data_D[mask]

    # sort data
    data_s_eps, data_s_D = sort_data(data_f_eps, data_f_D, idx_eps)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data_s_D.reshape((-1,36)), aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax)

    if save_path is not None: 
        filename = 'matrix_imshow.png'
        fig.savefig(os.path.join(save_path, filename))
        print(f'Saved {filename} to {save_path}')

    return

def imshow_D_all(dh, save_path):
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    im1 = ax1.imshow(dh[:,:3,:3].reshape((-1,9)), aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(im1, ax=ax1)

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    im2 = ax2.imshow(dh[:,3:6,3:6].reshape((-1,9)), aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(im2, ax=ax2)

    if save_path is not None: 
        filename1 = 'matrix_imshow_all_Dm.png'
        fig1.savefig(os.path.join(save_path, filename1))
        print(f'Saved {filename1} to {save_path}')

        filename2 = 'matrix_imshow_all_Db.png'
        fig2.savefig(os.path.join(save_path, filename2))
        print(f'Saved {filename2} to {save_path}')

    return

def imshow_sig_eps_all(sig_g, eps_g, save_path):
    data_ = [sig_g[:,:3], eps_g[:,:3], sig_g[:,3:6], eps_g[:,3:6]]
    filenames_ = ['n_i', 'eps_i', 'm_i', 'chi_i']

    for data, filename in zip (data_,filenames_):
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        im1 = ax1.imshow(data, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(im1, ax=ax1)

        if save_path is not None: 
            filename1 = 'matrix_imshow_all_'+filename+'.png'
            fig1.savefig(os.path.join(save_path, filename1))
            print(f'Saved {filename1} to {save_path}')

    return



########################################## Filtering 3d generated data ##########################################


def filter_3d_data(eps_g, sig_g = None, dh = None, constants = None, prefilter = True, principal = True):
    """
    filter out physically meaningless datapoints (datapoints where the top and bottom layer have too large or too small strains)

    Args:
        eps_g     (np.arr): original generalised strain matrix (n, 6)
        sig_g     (np.arr): original generalised stress (n, 6)
        dh        (np.arr): original stiffness matrix (n, 6, 6)
        t         (int):    thickness of element (constant for now)
        principal (bool):   if True, filter by principal strains; otherwise filter by eps_x/eps_y/gamma_xy ranges
    Returns:
        eps_g_f (np.arr): filtered eps_g
        sig_g_f (np.arr): filtered sig_g
        dh_f    (np.arr): filtered stiffness
    """

    # 1 - Calculate top and bottom strains from eps_g
    simulatesig = SigSimulator(constants)
    e = simulatesig.find_e_vec(cp.array(eps_g))
    eps_top, eps_bot = e[:,-1,:].get(), e[:,0,:].get()

    # 2 - Check whether strains lie in desired ranges
    if principal:
        mask = get_mask_strains_principal(eps_top, eps_bot, eps_2_min = -3e-3, eps_x_y_max = 50e-3)
    else:
        eps_x_y_range = [-3e-3, 50e-3]
        gamma_xy_range = [-20e-3, 20e-3]
        mask = get_mask_strains(eps_top, eps_bot, eps_x_y_range, gamma_xy_range)

    # 3 - Remove values that do not live in desired ranges
    eps_g_f = eps_g[mask]
    if not prefilter:
        sig_g_f = sig_g[mask]
        dh_f = dh[mask]
        return eps_g_f, sig_g_f, dh_f
    
    else:
        return eps_g_f

def get_mask_strains(eps_top, eps_bot, eps_range, gamma_range):  
    """
    check whether the calculated strains lie within desired ranges
    
    Args: 
        eps_top     (np.arr): strains in top layer (n, 3)
        eps_bot     (np.arr): strains in bottom layer (n, 3)
        eps_range   (list):   min and max value that is in acceptable epsilon range [min, max]
        gamma_range (list):   min and max value that is in acceptable gamma range [min, max]

    Returns: 
        mask        (np.arr): bool (n, 1)

    """

    mask_eps_x = (eps_top[:,0] > eps_range[0]) & (eps_top[:,0] < eps_range[1]) & (eps_bot[:,0] > eps_range[0]) & (eps_bot[:,0] < eps_range[1])
    mask_eps_y = (eps_top[:,1] > eps_range[0]) & (eps_top[:,1] < eps_range[1]) & (eps_bot[:,1] > eps_range[0]) & (eps_bot[:,1] < eps_range[1])
    mask_gam = (eps_top[:,2] > gamma_range[0]) & (eps_top[:,2] < gamma_range[1]) & (eps_bot[:,2] > gamma_range[0]) & (eps_bot[:,2] < gamma_range[1])

    mask = mask_eps_x & mask_eps_y & mask_gam

    print(f'Amount of points left after filtering: {np.sum(mask)}/{eps_top.shape[0]} = {np.sum(mask)/eps_top.shape[0]*100:.2f}\%')

    return mask

def get_mask_strains_principal(eps_top, eps_bot, eps_2_min = -5e-3, eps_x_y_max = 55e-3):
    """
    get mask based on principal strain filtering (instead of just filtering like in "get_mask_strain")

    Args:
        eps_top     (np.arr): strains in top layer (n, 3) -> (eps_x, eps_y, gamma_xy)
        eps_bot     (np.arr): strains in bottom layer (n, 3) -> (eps_x, eps_y, gamma_xy)
        eps_2_min   (float):  minimum allowed minor principal strain (most compressive)
        eps_x_y_max (float):  maximum allowed major principal strain (most tensile)

    Returns:
        mask        (np.arr): bool (n,)
    """

    eps_1_top, eps_2_top = get_principal_strains(eps_top[:, 0], eps_top[:, 1], eps_top[:, 2])
    eps_1_bot, eps_2_bot = get_principal_strains(eps_bot[:, 0], eps_bot[:, 1], eps_bot[:, 2])

    mask_eps_2 = (eps_2_top > eps_2_min) & (eps_2_bot > eps_2_min)
    mask_eps_1 = (eps_1_top < eps_x_y_max) & (eps_1_bot < eps_x_y_max)

    mask = mask_eps_2 & mask_eps_1

    print(f'Amount of points left after principal filtering: {np.sum(mask)}/{eps_top.shape[0]} = {np.sum(mask)/eps_top.shape[0]*100:.2f}\%')

    return mask


def get_principal_strains(eps_x, eps_y, gamma_xy):
    """
    get principal strains (eps_1, eps_2) from a 2D strain state (eps_x, eps_y, gamma_xy).
    eps_1 >= eps_2.
    """
    eps_avg = 0.5 * (eps_x + eps_y)
    R = np.sqrt(((eps_x - eps_y) * 0.5) ** 2 + (gamma_xy * 0.5) ** 2)

    eps_1 = eps_avg + R
    eps_2 = eps_avg - R

    return eps_1, eps_2




def filter_3D_data_batchwise(eps_g, sig_g, dh, constants, n_batches):
    # function not in use.

    batch_size = int(eps_g.shape[0]/n_batches)
    eps_list, sig_list, dh_list = [], [], []
    n = eps_g.shape[0]

    for i in range(n_batches):
        start = i*batch_size
        end = min(start + batch_size, n)
        t0 = time.perf_counter()

        eps_batch = eps_g[start:end]
        sig_batch = sig_g[start:end]
        dh_batch  = dh[start:end]

        eps_f_b, sig_f_b, dh_f_b = filter_3d_data(eps_batch, sig_batch, dh_batch, constants)

        eps_list.append(np.asarray(eps_f_b))
        sig_list.append(np.asarray(sig_f_b))
        dh_list.append(np.asarray(dh_f_b))
    
        eps_g_f = np.concatenate(eps_list, axis=0)
        sig_g_f = np.concatenate(sig_list, axis=0)
        dh_f    = np.concatenate(dh_list,  axis=0)

        if i%10 == 0:
            t_batch = time.perf_counter() - t0
            print(f'Finished batch {i+1}/{n_batches} with batchsize = {batch_size} in {t_batch:.2f} sec.')

    return eps_g_f, sig_g_f, dh_f


'''
#### Not in use ####

def get_neutral_axis(eps_g, t):
    """
    calculates z_sup and z_inf (neutral axis) for every element in the array
    
    Args: 
        eps_g   (np.arr): generalised strains (n, 6)
        t       (int):    thickness, constant

    Returns: 
        z_sup   (np.arr): superior height to neutral axis (n, 1)
        z_inf   (np.arr): inferior height to neutral axis (n, 1)

    """ 
    print('Warning: This function has not been tested.')
    z_sup = np.zeros((eps_g.shape[0], 1))

    chi = eps_g[:,3:6]
    eps_mid = eps_g[:,0:3]

    mask_pos = ((chi > 0) & (eps_mid > 0)) | ((chi < 0) & (eps_mid < 0))
    mask_zero = ((chi < 1e-9) & (chi > -1e-9))
    mask_neg = ~mask_pos & ~mask_zero

    z_sup[mask_pos] = t/2 - eps_mid[mask_pos]/np.tan(chi[mask_pos])
    z_sup[mask_neg] = t/2 + eps_mid[mask_neg]/np.tan(chi[mask_neg])
    z_sup[mask_zero] = 0

    z_inf = t-z_sup
    
    return z_sup, z_inf

def get_top_bottom_strains(eps_g, z_sup, z_inf):  
    """
    get top and bottom layer stresses for every element in dataset
    
    Args: 
        eps_g       (np.arr): generalised strains (n, 6)
        z_sup       (np.arr): neutral axis from the top (n, 1)
        z_inf       (np.arr): neutral axis from the bottom (n, 1)
    
    Returns: 
        eps_top   (np.arr): layer stresses in top layer (n, 3)
        eps_bottom(np.arr): layer stresses in bottom layer (n, 3)

    """

    print('Warning: This function has not been tested.')

    eps_mid = eps_g[:,0:3]
    chi = eps_g[:,3:6]
    
    eps_top = eps_mid + z_sup*chi
    eps_bot = eps_mid + z_inf*chi

    return eps_top, eps_bot
'''



########################################## Findig outliers in D ##########################################

def find_outlier_d(dh: np.array, eps_g: np.array, factor: float = 1.5) -> np.array:
    """
    determines outliers in D-dataset
    
    Args: 
        dh          (np.arr / float):   (n, 6, 6) stiffness data generated by sampler

    Returns:
        outliers    (np.arr / bool):     (n, 6, 6) outliers in dataset

    """

    q1 = np.quantile(dh, 0.25, axis = 0)
    q3 = np.quantile(dh, 0.75, axis = 0)
    iqr = q3-q1

    outlier_min = q1-(factor*iqr)
    outlier_max = q3+(factor*iqr)
    
    outlier_mask = (dh < outlier_min) | (dh > outlier_max)
    
    mask_1d = outlier_mask.any(axis=(1, 2))
    outliers = dh[mask_1d]
    eps_g_outlier = eps_g[mask_1d]

    print(f'Found {np.sum(mask_1d)} outliers for D in dataset with N = {dh.shape[0]/1e3:.2f}*1e3 points ({np.sum(mask_1d)/(dh.shape[0])*100:.3f}%)')

    return outliers, eps_g_outlier, mask_1d