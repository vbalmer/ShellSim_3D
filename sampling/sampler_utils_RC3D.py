# vb, 06.03.2026

import os
import numpy as np
import pyDOE as doe
from itertools import permutations
import time
import h5py

import matplotlib.pyplot as plt


from concrete_classes import dict_CC

########################################## Sampling strains ##########################################

def get_constant_sampling_params(sample_2d:bool) -> tuple:
    '''
    collect constant parameters for sampling (material parameters, ranges, ...)
    
    Args:
        sample_2d   (bool):    True for sampling in 2D

    Returns: 
        constants   (dict):    Containing relevant input parameters for sampling

    '''

    c = {
        'n_samples_2D': 1e6, 
        'n_samples_3D': 1e6,         #4e9,47e3 (for 6 elements per dimension), 64e6 (20 elements per dimension)

        'min': [-3e-3]*2 + [-4e-3],
        'max': [5e-3]*2  + [4e-3],
        't': 300,
        'CC': 1,
        'n_layer': 20,
        'nu': 0,
        'fsy': 435,
        'fsu': 470,
        'Es': 205e3,
        'Esh': 9.4e3,
        'D': 16,
        'Dmax': 16,
        's': 200,
        'rho_x': [0.025]*4 + [0]*12 + [0.025]*4,    # length of the array needs to correspond to the amount of layers.
        'rho_y': [0.025]*4 + [0]*12 + [0.025]*4,
        'rho_sublayer': True,

    }

    c_3D = {        
        'min': [-3e-3]*2 + [-4e-3] + [-0.02e-3]*2 + [-0.027e-3],        # units: [-], [1/mm]
        'max': [5e-3]*2  + [4e-3] +  [0.033e-3]*2 + [0.027e-3],         # units: [-], [1/mm]
    }

    if not sample_2d:
        c.update(c_3D)

    
    # select values for concrete: 
    idx = dict_CC['CC'].index(c['CC'])
    dict_CC_one = {key: values[idx] for key, values in dict_CC.items()}

    dict_CC_one.update({'fsy': c["fsy"], 'fsu': c["fsu"], 'Es': c["Es"], 'Esh': c["Esh"], 'D': c["D"], 'Dmax': c["Dmax"], 's': c["s"]})
    mat_dict = dict_CC_one

    return c, mat_dict


def sample_eps(sampler:str, constants: dict) -> np.array:
    """
    Samples 2D-strains.
    
    Args: 
        sampler      (str) : uniform, uniform_3D, log or LHS - so far only uniform implemented.
        constants    (dict): material and geom constants, amount of samples

    Returns: 
        eps_l (np.arr): eps_layer for one layer (n_samples_2D, 3)
    
    """
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

    else: 
        raise UserWarning('This has not yet been implemented.')

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

        :param parnames: List of parameter names
        :type parnames: list(str)
        :param bounds: List of lower/upper bounds,
                        must be of the same length as par_names
        :type bounds: list(tuple(float, float))
        :param int samples: Number of samples
        :param str criterion: A string that tells lhs how to sample the
                                points. See docs for pyDOE.lhs().
        :return: DataFrame
        """
        raise UserWarning('This code is not yet implemented. Please check # TODO')
        bounds = np.vstack((self.min, self.max))
        bounds = bounds.T
        

        lhs = doe.lhs(len(self.parnames), samples=self.samples, criterion=criterion)
        par_vals = {}
        for par, i in zip(self.parnames, range(len(self.parnames))):
            par_min = bounds[i][0]
            par_max = bounds[i][1]
            par_vals[par] = np.array(lhs[:, i]) * (par_max - par_min) + par_min

        # Convert dict(str: np.ndarray) to pd.DataFrame 
        # TODO: remove this. get data in np format directly.
        
        par_df = pd.DataFrame(columns=self.parnames, index=np.arange(int(self.samples)))
        for i in range(self.samples):
            for p in self.parnames:
                par_df.loc[i, p] = par_vals[p][i]

        return par_df
    
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
    plt.savefig(os.path.join(os.getcwd(), "sampling\\plots\\" + filename + ".png"))
    print(f'Saved {filename} to sampling\\plots\\{filename}.png')
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


########################################## Visualising stiffnesses ##########################################

def plot_filtered_stiffness(data_eps, data_D, idx_eps, save_path):
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

    # plot data
    plot_data_stiffness(data_s_eps, data_s_D, idx_eps,save_path)

    return

def get_mask_strain(data_eps, idx_eps, tol = [0.5e-3, 0.5e-3, 0.9e-3, 0.4e-5, 0.4e-5, 0.4e-5]):
    # tol for 6 points per direction: [0.5e-3, 0.5e-3, 1.6e-3, 0.5e-5, 0.5e-5, 0.7e-5]
    tol = np.array(tol)
    cols = np.arange(data_eps.shape[1])!=idx_eps
    mask = np.all(np.abs(data_eps[:,cols])<tol[cols], axis =1)

    data_f_eps = data_eps[mask]

    print(f'After filtering data, {mask.sum()} datapoints are left.')
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

def plot_data_stiffness(data_s_eps, data_s_D, idx_eps, save_path):

    fig, axs = plt.subplots(6,6, figsize = [30,20])

    for i in range(6): 
        for j in range(6): 
            axs[i,j].plot(data_s_eps[:,idx_eps], data_s_D[:,i,j], marker = 'o')

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