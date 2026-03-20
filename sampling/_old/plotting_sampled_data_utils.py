# plotting_sampled_data_utils.py

import os
import numpy as np
from data_work import read_data, find_D_linel, transf_units
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sampler_utils import Sampler_utils_vb, SamplerUtils
from concrete_classes import dict_CC


def get_paths_data(mat_data_names):
    mat_data_paths = {}
    for key in mat_data_names.keys():
        mat_data_paths[key] =  os.path.join(os.getcwd(), mat_data_names[key])

    return mat_data_paths


def read_all_data(mat_data_paths):
    numpy_data = {}
    for key in mat_data_paths.keys():
        numpy_data[key] = {
            'eps': read_data(mat_data_paths[key], 'eps'),
            't': read_data(mat_data_paths[key], 't'), 
            'sig': read_data(mat_data_paths[key], 'sig'), 
            'D': read_data(mat_data_paths[key], 'De').reshape((-1,8,8))
        }
    
    return numpy_data


def format_all_data(numpy_data,n, outliers):
    plot_data = {}
    plot_data_cut = {}
    for key in numpy_data.keys():
        plot_data[key] = {
        'x_data': np.concatenate((numpy_data[key]['eps'], numpy_data[key]['t']), axis = 1),
        'y_data': numpy_data[key]['sig'],
        'D_data': np.concatenate((numpy_data[key]['D'][:, :6, :6].reshape((-1,36)), numpy_data[key]['D'][:, 6, 6].reshape((-1,1)),numpy_data[key]['D'][:, 7, 7].reshape((-1,1))), axis = 1)
        }

        plot_data_cut[key] = {
            'x_data': plot_data[key]['x_data'][:n,:],
            'y_data': plot_data[key]['y_data'][:n,:],
            'D_data': plot_data[key]['D_data'][:n,:]
        }

        if outliers:
            plot_data[key]['D_outliers'] = find_outliers_D(plot_data[key]['D_data'])
            plot_data_cut[key]['D_outliers'] = plot_data[key]['D_outliers'][:n,:]

    return plot_data, plot_data_cut



def combine_all_data(numpy_data, n1, n2, n3):
    # combine the data sets and crop until n'th entry of every dataset
    n = {}
    shape_list = {}
    for key in numpy_data.keys():
        if key in ['67', '68_1', '68_2', '68_3', '69_1', '69_2', '69_3', '70_1', '70_2', '70_3']:
            n[key] = n1
        elif key in ['64']:
            n[key] = n2
        else: 
            n[key] = n3
        
    # ensure that geom is four-dimensional before combining the data.
    padded_t = {
        key: pad_for_t(numpy_data[key]['t'][0:n[key],:], 11)
        for key in numpy_data.keys()
    }
    
    new_data_eps_np = np.concatenate([numpy_data[key]['eps'][0:n[key],:] for key in numpy_data.keys()], axis = 0)
    new_data_t_np = np.concatenate([padded_t[key][0:n[key],:] for key in numpy_data.keys()], axis = 0)
    new_data_sig_np = np.concatenate([numpy_data[key]['sig'][0:n[key],:] for key in numpy_data.keys()], axis = 0)
    new_data_De_np = np.concatenate([numpy_data[key]['D'][0:n[key],:,:].reshape((-1,64)) for key in numpy_data.keys()], axis = 0)

    return new_data_eps_np, new_data_sig_np, new_data_t_np, new_data_De_np



def pad_for_t(arr,target_cols):
    """
    Pad array along axis 1 by repeating second-to-last column until reaching target_cols.
    """
    curr_cols = arr.shape[1]
    if curr_cols == target_cols:
        return arr
    pad = np.repeat(arr[:, 1].reshape((-1,1)), target_cols - curr_cols, axis=1)
    return np.concatenate([arr[:,:2], pad, arr[:,2:]], axis=1)



def find_outliers_D(np_data):
    '''
    creates mask for outliers per column of stiffness matrix
    np_data:       array of D-data, expected shape (n, 64)

    NB. the shape actually doesn't matter.
    '''

    q_25 = np.percentile(np_data,25, axis = 0)
    q_75 = np.percentile(np_data,75, axis= 0)
    iqr = q_75-q_25

    lower_bound = q_25-1.5*iqr
    upper_bound = q_75+1.5*iqr

    outlier_mask = (np_data < lower_bound) | (np_data > upper_bound)

    outliers = np_data*outlier_mask

    return outliers




def three_D_histogram(np_data, idx_eps, idx_D, alpha_ = 0.5, save_path = None, linel = False):
    '''
    plots a 3d histogram of any combination of eps and D given by idx_eps and idx_D
    if linel = True: draws a line of the corresponding linear elastic stiffness

    '''

    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    colors = plt.cm.get_cmap('viridis', len(np_data)+1)(np.arange(len(np_data)+1))
    alphas = alpha_*np.ones(len(np_data))



    for key, l in zip(np_data.keys(), range(len(np_data))):
        x,y = np_data[key]['x_data'][:,idx_eps], np_data[key]['D_data'][:,idx_D]
        hist, xedges, yedges = np.histogram2d(x,y, bins = 10)

        xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = np.zeros_like(xpos)

        dx = (xedges[1]-xedges[0])*np.ones_like(zpos)
        dy = (yedges[1]-yedges[0])*np.ones_like(zpos)
        dz = hist.ravel()

        gap = 0.1

        dx = dx*(1-gap)
        dy = dy*(1-gap)
        xpos = xpos + (dx*gap/2)
        ypos = ypos + (dy*gap/2)

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color = colors[l], alpha = alphas[l])
        print('plotted key ', key)

    if linel: 
        for t, E in zip([200,450], [32000, 39000]):
            D_linel_MN_cm = find_D_linel(t, E)
            D_linel_N_mm = transf_units(D_linel_MN_cm.reshape((1,8,8)), 'D', forward=False, linel=True)
            D_linel_ = np.concatenate((D_linel_N_mm[:,:6,:6].reshape((-1,36)), D_linel_N_mm[:,6,6].reshape((-1,1)), D_linel_N_mm[:,7,7].reshape((-1,1))), axis=1)
            D_linel = D_linel_[0,idx_D]
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = D_linel*np.ones_like(x_line)
            z_line = np.zeros_like(x_line)
            ax.plot3D(x_line, y_line, z_line, color = 'red', label = '$D_{linel, t = '+str(t)+'}$', linewidth = 3)


    ax.set_xlabel('eps_'+str(idx_eps))
    ax.set_ylabel('D_'+str(idx_D))
    ax.set_zlabel('Frequency')
    ax.legend()
    

    if save_path is not None:
        ax.view_init(elev=30, azim=60)
        plt.savefig(os.path.join(save_path,'3d_hist_'+str(idx_eps)+'_'+str(idx_D)))
        print('Saved 3d-hist at ', save_path)
    

    return




################### Plotting filtered data and subfunctions ###################

def plot_filtered_data(plot_data, idx_eps, geom, stiffness_plots = False, quantiles=None, save_path=None):
    '''
    plots figures of dataset for given variation of idx_eps and filtered according to geom.
    idx_eps     (list)      index of variable that was varied in input dataset
    geom        (list)      [t, rho, CC] that was varied in the dataset
    quantiles   (list)      [lower, upper]: range of y-axis quantiles depending on the max. values in the respective datasets.
    save_path   (str)       location to save figures
    '''

    # filter data according to geom and idx_eps
    data_f = filter_data(plot_data, geom)

    # sort data in ascending order according to idx_eps
    data_s = sort_data(data_f, idx_eps)

    # plot generalised_stresses
    n_tot = {}
    for key in plot_data.keys():
        n_tot[key] = plot_data[key]['x_data'].shape[0]
    plot_sh(data_s, idx_eps, save_path, n_tot)

    # plot D_m, D_mb, D_bm, D_b
    if stiffness_plots:
        plot_stiffness(data_s, idx_eps,save_path)

    return


def filter_data(data, geom): 
    '''
    subfunction for filtered_dataset_plotting function
    filters data according to given constant t, rho, cc (=geom)
    '''
    new_data = {}
    for key in data.keys():
        mask = np.all(abs(data[key]['x_data'][:,8:11] - np.array(geom)) < 1e-5, axis = 1)
        new_data[key] = {}
        for key2 in ['x_data', 'y_data', 'D_data']: 
            new_data[key][key2] = data[key][key2][mask]
        print(f'Remaining rows after filtering: {mask.sum()}')
        if mask.sum() < 1:
            raise UserWarning('The filtering yielded zero points. Please reconsider your choice of "geom".')
    return new_data
    
    
def sort_data(data, idx_eps):
    '''
    subfunction for filtered_dataset_plotting function
    sorts data according to values in x_data, idx_eps    
    '''
    new_data = {}
    for key in data.keys():
        x_rel = data[key]['x_data'][:,idx_eps[0]]
        x_flat = x_rel.ravel()
        idx = np.argsort(x_flat)
        new_data[key] = {}
        for key2 in ['x_data', 'y_data', 'D_data']: 
            new_data[key][key2] = data[key][key2][idx]
    return new_data


def plot_sh(data, idx_eps, save_path, n_tot, units = 'kN'):
    '''
    subfunction for filtered_dataset_plotting function
    units       (str)       'kN' or 'N' - choose between original units N, mm and more intuitive units kN, m
    '''
    fig, axs = plt.subplots(3,3, figsize = [15, 10])
    
    if units == 'kN':
        labels_y = [r'$n_x [kN/m]$', r'$n_y [kN/m]$', r'$n_{xy} [kN/m]$', 
                    r'$m_x [kNm/m]$', r'$m_y [kNm/m]$', r'$m_{xy} [kNm/m]$',
                    r'$v_{x} [kN/m]$', r'$v_{y} [kN/m]$']
    else: 
        labels_y = [r'$n_x [N/mm]$', r'$n_y [N/mm]$', r'$n_{xy} [N/mm]$', 
                r'$m_x [Nmm/mm]$', r'$m_y [Nmm/mm]$', r'$m_{xy} [Nmm/mm]$',
                r'$v_{x} [N/mm]$', r'$v_{y} [N/mm]$']
    labels_x = [r'$\epsilon_x$', r'$\epsilon_y$', r'$\epsilon_{xy}$',
                r'$\chi_x$', r'$\chi_y$', r'$\chi_xy$',
                r'$\gamma_{xy}$', r'$\gamma_{xz}$']
    label_x = labels_x[idx_eps[0]]
    idx = np.array([[0,1,2],
                    [3,4,5],
                    [6,7,7]])
    
    if units == 'kN':
        for key in data.keys():
            data[key]['y_data'][:,3:6] = data[key]['y_data'][:,3:6]*1e-3

    for i in range(3): 
        for j in range(3):
            if i == 2 and j == 2: 
                pass
            else: 
                for key in data.keys():
                    axs[i,j].plot(data[key]['x_data'][:,idx_eps[0]], data[key]['y_data'][:,idx[i,j]], label = key)
                    axs[i,j].set_ylabel(labels_y[idx[i,j]])
                    axs[i,j].set_xlabel(label_x)

    axs[-1, -1].axis('off')
    axs[-1, -2].legend()
    plt.tight_layout()

    n_filt = {}
    for key in data.keys():
        n_filt[key] = data[key]['x_data'].shape[0]
    text = "".join([
                f"$n_{{{key}}}/n_{{\\text{{tot}}}}$ = {n_filt[key]} / {n_tot[key]}\n"
                for key in n_filt.keys()
            ])
    at_ = AnchoredText(text,prop=dict(size=10), frameon=True, loc='center')
    at_.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
    axs[-1, -1].add_artist(at_)

    if save_path is not None: 
        fig.savefig(os.path.join(save_path, 'filtered_dataset_sh.png'))
        print(f'Saved filtered_dataset_sh to {save_path}')


    

    return


def plot_stiffness(data, idx_eps, save_path, units = 'kN'):
    '''
    Plots all stiffness matrix entries of Dm, Dmb, Dbm, Db (but not Ds)
    '''

    fig, axs = plt.subplots(6,6, figsize = [30, 20])

    names_y = np.array([['$D_{m,11}$', '$D_{m,12}$', '$D_{m,13}$', '$D_{mb,11}$', '$D_{mb,12}$', '$D_{mb,13}$'],
                        ['$D_{m,21}$', '$D_{m,22}$', '$D_{m,23}$', '$D_{mb,21}$', '$D_{mb,22}$', '$D_{mb,23}$'],
                        ['$D_{m,31}$', '$D_{m,32}$', '$D_{m,33}$', '$D_{mb,31}$', '$D_{mb,32}$', '$D_{mb,33}$'],
                        ['$D_{bm,11}$', '$D_{bm,12}$', '$D_{bm,13}$', '$D_{b,11}$', '$D_{b,12}$', '$D_{b,13}$'],
                        ['$D_{bm,21}$', '$D_{bm,22}$', '$D_{bm,23}$', '$D_{b,21}$', '$D_{b,22}$', '$D_{b,23}$'],
                        ['$D_{bm,31}$', '$D_{bm,32}$', '$D_{bm,33}$', '$D_{b,31}$', '$D_{b,32}$', '$D_{b,33}$'],
                        # ['$D_{s,11}$', '$D_{s,22}$', '$D_{s,22}$', '$D_{s,22}$', '$D_{s,22}$', '$D_{s,22}$']
                        ])
    if units == 'kN':
        unit_label = np.array([[r'$\rm [kN/m]$', r'$\rm [kN/m]$', r'$\rm [kN/m]$', r'$\rm [kN]$', r'$\rm [kN]$', r'$\rm [kN]$'],
                            [r'$\rm [kN/m]$', r'$\rm [kN/m]$', r'$\rm [kN/m]$', r'$\rm [kN]$', r'$\rm [kN]$', r'$\rm [kN]$'],
                            [r'$\rm [kN/m]$', r'$\rm [kN/m]$', r'$\rm [kN/m]$', r'$\rm [kN]$', r'$\rm [kN]$', r'$\rm [kN]$'],
                            [r'$\rm [kN]$', r'$\rm [kN]$', r'$\rm [kN]$', r'$\rm [kNm]$', r'$\rm [kNm]$', r'$\rm [kNm]$'],
                            [r'$\rm [kN]$', r'$\rm [kN]$', r'$\rm [kN]$', r'$\rm [kNm]$', r'$\rm [kNm]$', r'$\rm [kNm]$'],
                            [r'$\rm [kN]$', r'$\rm [kN]$', r'$\rm [kN]$', r'$\rm [kNm]$', r'$\rm [kNm]$', r'$\rm [kNm]$'],
                            #   [r'$\rm [kN/m]$', r'$\rm [kN/m]$', r'$\rm [kN/m]$', r'$\rm [kNm]$', r'$\rm [kNm]$', r'$\rm [kNm]$']
                            ]) 
    else: 
        unit_label = np.array([[r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$'],
                            [r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$'],
                            [r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$'],
                            [r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$'],
                            [r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$'],
                            [r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$'],
                            #   [r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$']
                            ])
    
    labels_y = np.char.add(np.char.add(names_y, ' '), unit_label)
    labels_x = [r'$\epsilon_x$', r'$\epsilon_y$', r'$\epsilon_{xy}$',
                r'$\chi_x$', r'$\chi_y$', r'$\chi_xy$',
                r'$\gamma_{xy}$', r'$\gamma_{xz}$']
    label_x = labels_x[idx_eps[0]]
    

    if units == 'kN':
        for key in data.keys():
            D_data = data[key]['D_data'][:,:36].reshape((-1,6,6))
            D_data[:,0:3,3:6] = D_data[:,0:3,3:6]*1e-3
            D_data[:,3:6,0:3] = D_data[:,3:6,0:3]*1e-3
            D_data[:,3:6,3:6] = D_data[:,3:6,3:6]*1e-6
            data[key]['D_data'][:,:36] = D_data.reshape((-1,36))


    for i in range(6): 
        for j in range(6):
            for key in data.keys():
                D_data = data[key]['D_data'][:,:36].reshape((-1,6,6))
                axs[i,j].plot(data[key]['x_data'][:,idx_eps[0]], D_data[:,i,j], label = key)
                axs[i,j].set_ylabel(labels_y[i,j])
                axs[i,j].set_xlabel(label_x)

    axs[-1, -1].legend()
    plt.tight_layout()

    if save_path is not None: 
        fig.savefig(os.path.join(save_path, 'filtered_dataset_De.png'))
        print(f'Saved filtered_dataset_De to {save_path}')


    return


def get_colors_from_map(inp_vector):
    n_colors = len(inp_vector)
    cmap1, cmap2, cmap3 = plt.cm.gist_yarg, plt.cm.Blues, plt.cm.RdPu
    values = np.linspace(0.2,0.8,n_colors)
    colors1, colors2, colors3 = {}, {}, {}
    for v, key in zip(values, inp_vector.keys()):
        colors1[key], colors2[key], colors3[key] = cmap1(v), cmap2(v), cmap3(v)
    return colors1, colors2, colors3




################### Plotting filtered data for given cross-section and subfunctions ###################


def plot_filtered_crosssection(eps, geom, save_path = None, dir = 0, rho_sublayer = True):
    '''
    eps         (dict)          8 values for each generalised strain, for different instances '0'...'n' --> draw several cs
    geom        (list)          three values: t, rho, CC
    save_path   (str)           location where image is saved
    dir         (int)           direction in which stresses / strains are plotted
    rho_sublayer(bool)          if true: samples only with rho in 4 layers (not rho in 8 layers)

    Plots a cross-section for a given eps (generalised eps, list of 8 values) and given 
    geometry by re-calculating the layer strains and stresses.
    
    Fixed values: nu = 0, num_layers = 20, constant values for steel


    '''

    # 0 - Preparations for calculating layer strains and stresses
    num_layers = 20
    samplerutils, t_extended = prepare_layer_eps_sig_calc(geom, num_layers)
    
    # 1 - Calculate layer strains and stresses
    eps_sig_plot_data = get_eps_and_sig_layers(eps, t_extended, num_layers, samplerutils, dir, rho_sublayer)

    # 3 - Plot both
    fig, axs = plt.subplots(1,2, figsize = [10,5])
    axs[0] = plot_eps_layer(eps_sig_plot_data, axs)
    axs[1] = plot_sig_layer(eps_sig_plot_data, axs)
    axs[0].legend()

    if save_path is not None: 
        fig.savefig(os.path.join(save_path, 'cross_section_filtered.png'))
        print(f'Saved cross_section_filtered to {save_path}')

    return


def prepare_layer_eps_sig_calc(geom, num_layers):
    mat_dict = get_constant_values()
    analytical_sampler = Sampler_utils_vb(E1 = None, nu1=0, E2=None, nu2=None, mat_dict = mat_dict)
    t_extended = analytical_sampler.extend_material_parameters(np.array(geom).reshape(1,3))
    t1 = t_extended[:,0].reshape(-1,1,1)
    t2 = np.zeros_like(t1).reshape(-1,1,1)
    other = t_extended[:,1:].reshape(-1,t_extended.shape[1]-1,1)
    samplerutils = SamplerUtils(t1, t2, nl=num_layers, mat=3, nel=t_extended.shape[0], E1=None, nu1=0, E2=None, nu2=None, other = other, mat_dict = mat_dict)
    return samplerutils, t_extended


def get_constant_values():
    fsy  = 435                                          # [MPa]
    fsu = 470                                           # [MPa]
    Es = 205e3                                          # [MPa]
    Esh = 8e3                                           # [MPa]
    D = 16                                              # [mm]
    Dmax = 16                                           # [mm]
    s = 200                                             # [mm]                        
    dict_CC.update({'fsy': fsy, 'fsu': fsu, 'Es': Es, 'Esh': Esh, 'D': D, 'Dmax': Dmax, 's': s})
    mat_dict = dict_CC
    return mat_dict


def get_eps_and_sig_layers(eps, t_extended, num_layers, samplerutils, dir, rho_sublayer):
    mat_eps_sig_layer = {}
    for key in eps.keys():
        eh = np.array(eps[key]).reshape((-1, 1, 1, 8))                                                          # shape = (1, 1, 1, 8)
        e0 = np.zeros((t_extended.shape[0], 20, 1, 1, 5), dtype=np.float32)                                # shape = (1, 20, 1, 1, 5)
        [e,ex,ey,gxy,e1,e3,th] = samplerutils.find_e(e0,eh,1)
        s = samplerutils.find_s(e,1, count = [0,0,0], rho_sublayer = rho_sublayer)
        t_vec, s_vec = [], []
        t = t_extended[:,0]
        for i in range(num_layers):
            t_vec.append(t/(num_layers*2)+(t/(num_layers))*i)
            st = s[0][i][0][0][0]
            s_vec.append([st.sx.real,st.sy.real,st.txy.real,st.txz.real,st.tyz.real])

        mat_eps_sig_layer[key] = {
            'z': np.array(t_vec)[:,0],
            'eps_layer': e[0,:,0,0,dir], 
            'sig_layer': np.array(s_vec)[:,dir], 
        }

    return mat_eps_sig_layer


def plot_eps_layer(eps_plot_data, axs):
    colors, _, _ = get_colors_from_map(eps_plot_data)
    for key in eps_plot_data.keys():
        axs[0].plot(eps_plot_data[key]['eps_layer'], eps_plot_data[key]['z'], linestyle = 'solid', linewidth = 2, color = colors[key],
                    marker='o', markersize = 3, label = key)
        axs[0].plot(np.zeros_like(eps_plot_data[key]['z']), eps_plot_data[key]['z'], linestyle = '--', color = colors[key])
        axs[0].set_xlabel('eps_x [-]')
        axs[0].set_ylabel('t [mm]')

        if key == next(iter(eps_plot_data)):
            axs[0].annotate(f"max={eps_plot_data[key]['eps_layer'][-1]*1e3:.3f} ‰",
                        (eps_plot_data[key]['eps_layer'][-1], eps_plot_data[key]['z'][-1]),
                        xytext=(5, 5), textcoords='offset points',
                        color='black', fontsize=9, arrowprops=dict(arrowstyle="->", color="black"))
            
            axs[0].annotate(f"min={eps_plot_data[key]['eps_layer'][0]*1e3:.3f} ‰",
                            (eps_plot_data[key]['eps_layer'][0], eps_plot_data[key]['z'][0]),
                            xytext=(5, -10), textcoords='offset points',
                            color='black', fontsize=9, arrowprops=dict(arrowstyle="->", color="black"))

    return axs[0]

def plot_sig_layer(sig_plot_data, axs):
    colors, _, _ = get_colors_from_map(sig_plot_data)
    for key in sig_plot_data.keys():
        axs[1].plot(sig_plot_data[key]['sig_layer'], sig_plot_data[key]['z'], linestyle = 'solid', linewidth = 2, color = colors[key], 
                    marker='o', markersize = 2)
        axs[1].plot(np.zeros_like(sig_plot_data[key]['z']), sig_plot_data[key]['z'], linestyle = '--', color = colors[key])
        axs[1].set_xlabel('sig_x [N/mm]')
        axs[1].set_ylabel('t [mm]')

        if key == next(iter(sig_plot_data)):
            axs[1].annotate(f"max={sig_plot_data[key]['sig_layer'][-1]:.2f} N/mm$^2$",
                        (sig_plot_data[key]['sig_layer'][-1], sig_plot_data[key]['z'][-1]),
                        xytext=(5, 5), textcoords='offset points',
                        color='black', fontsize=9, arrowprops=dict(arrowstyle="->", color="black"))
            
            axs[1].annotate(f"min={sig_plot_data[key]['sig_layer'][0]:.2f} N/mm$^2$",
                            (sig_plot_data[key]['sig_layer'][0], sig_plot_data[key]['z'][0]),
                            xytext=(5, -10), textcoords='offset points',
                            color='black', fontsize=9, arrowprops=dict(arrowstyle="->", color="black"))


    return axs[1]


################### Filtering the data according to predefined criteria ###################


def filter_dataset(data_path, save_new_data = False):
    '''
    Filters dataset according to predefined filter criteria in terms of epsilon.
    
    :param data_path:       (str)   path to data that should be filtered
    :param save_new_data:   (bool)  should the new dataset be saved? 
    '''

    points = {}

    # extract data from data_path
    path_extended = os.path.join('04_Training\\data\\', data_path)
    data_sig = read_data(path_extended, id = 'sig')
    data_eps = read_data(path_extended, id = 'eps')
    data_t = read_data(path_extended, id = 't')
    data_De = read_data(path_extended, id = 'De')
    points['sig'] = data_sig
    points['eps'] = data_eps
    points['t'] = data_t
    points['De'] = data_De

    # create mask according to filter criteria:
    mask0 = abs(points['eps'][:,0:2]) < 0.5e-6
    mask1 = abs(points['eps'][:,2]) < 0.5e-7
    mask = mask0.any(axis=1) | mask1
    print(f'Amount of points before masking: {points["eps"].shape[0]}')
    print(f'Amount of points after masking: {np.sum(~mask)}')
    print(f'Amount of points in mask (to be removed): {np.sum(mask)}')
    print(f'Ratio: masked/total {np.sum(~mask)/points["eps"].shape[0]*100:.2f}\%')

    # filter points:
    filtered_points = {}
    filtered_points['eps'] = points['eps'][~mask]
    filtered_points['sig'] = points['sig'][~mask]
    filtered_points['t'] = points['t'][~mask]
    filtered_points['De'] = points['De'][~mask]


    # Save points if requested:
    from datetime import datetime
    import pickle
    if save_new_data:
        folder_name = f"data_{datetime.now().strftime('%Y%m%d_%H%M')}_fake"
        save_data_path = os.path.join('04_Training\\data\\', folder_name)
        os.makedirs(save_data_path, exist_ok=True)


        with open(os.path.join(save_data_path, 'new_data_t.pkl'), 'wb') as fp:
                pickle.dump(filtered_points['t'].astype(np.float32), fp)
        with open(os.path.join(save_data_path, 'new_data_eps.pkl'), 'wb') as fp:
                pickle.dump(filtered_points['eps'].astype(np.float32), fp)
        with open(os.path.join(save_data_path, 'new_data_sig.pkl'), 'wb') as fp:
                pickle.dump(filtered_points['sig'].astype(np.float32), fp)
        with open(os.path.join(save_data_path, 'new_data_De.pkl'), 'wb') as fp:
                pickle.dump(filtered_points['De'].astype(np.float32), fp)  
        print('Data saved to ', save_data_path)
        
        print('data shapes:')
        print('t: ', filtered_points['t'].shape)
        print('eps: ', filtered_points['eps'].shape)
        print('sig: ', filtered_points['sig'].shape)
        print('De: ', filtered_points['De'].shape)

    return