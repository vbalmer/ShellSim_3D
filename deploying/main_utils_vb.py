# main_utils.py

# (c) vb, 24.3.2025 
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from matplotlib.cm import viridis
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

# from data_work_depl import multiple_diagonal_plots, plots_mike_dataset, transf_units, multiple_diagonal_plots_D, transform_data, calculate_errors
from deployment_prediction import add_zerovals_stats

import shutil
import wandb

################################## Plotting functions ##################################

def plot_convergence(i, mat_un_thn, numit, mat_e_r, r, NN = 'NN', e = None):
    global it_steps
    global conve50
    global conve90
    global conve99
    global convun50
    global convun90
    global convun99
    global convunmed
    global convthn50
    global convthn90
    global convthn99
    global convthnmed
    global plt
    global fig
    global ax1
    global ax2
    global ax3

    rele = mat_e_r['rele']
    convrf = mat_e_r['convrf']
    convrm = mat_e_r['convrm']
    
    relthn = mat_un_thn['relthn']
    relun = mat_un_thn['relun']



    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)

    if i == 0:
        it_steps = [2]
        conve50 = [np.percentile(abs(rele),50)]
        conve90 = [np.percentile(abs(rele), 90)]
        conve99 = [np.percentile(abs(rele), 99)]
        convun50 = [np.percentile(abs(relun),50)]
        convun90 = [np.percentile(abs(relun), 90)]
        convun99 = [np.percentile(abs(relun), 99)]
        convunmed = [np.median(abs(relun))]
        convthn50 = [np.percentile(abs(relthn),50)]
        convthn90 = [np.percentile(abs(relthn), 90)]
        convthn99 = [np.percentile(abs(relthn), 99)]
        convthnmed = [np.median(abs(relun))]

    else:
        it_steps = np.append(it_steps, i+2)
        conve50 = np.append(conve50, np.percentile(abs(rele),50))
        conve90 = np.append(conve90, np.percentile(abs(rele), 90))
        conve99 = np.append(conve99, np.percentile(abs(rele), 99))
        convun50 = np.append(convun50, np.percentile(abs(relun),50))
        convun90 = np.append(convun90, np.percentile(abs(relun), 90))
        convun99 = np.append(convun99, np.percentile(abs(relun), 99))
        convunmed = np.append(convunmed, np.median(abs(relun)))
        convthn50 = np.append(convthn50, np.percentile(abs(relthn),50))
        convthn90 = np.append(convthn90, np.percentile(abs(relthn), 90))
        convthn99 = np.append(convthn99, np.percentile(abs(relthn), 99))
        convthnmed = np.append(convthnmed, np.median(abs(relthn)))
        
    if  abs(min(min(conve50),0.001)) < 10**12 and max(conve50) > 0 and max(conve50) < 10**12:
        ax1.axis([2, numit+1, min(min(conve50),0.001), max(conve99)])
    ax1.set_yscale("log")
    ax1.set_title("$\delta$" + "$\epsilon$$_{klij}$" + "/" +"$\epsilon$$_{klij}$" )
    ax1.plot(it_steps, conve50,'k')
    ax1.plot(it_steps, conve90, 'b')
    ax1.plot(it_steps, conve99, 'g')
    ax1.legend(["$P$$_{50}$", "$P$$_{90}$","$P$$_{99}$"], loc=3)
    ax1.grid(True)

    if abs(min(min(convun50), 0.001)) < 10 ** 12 and max(convun50) > 0 and max(convun50) < 10 ** 12:
        ax2.axis([2, numit+1, min(min(convun50),0.001), max(convun99)])
    ax2.set_yscale("log")
    ax2.set_title("$\delta$" + "$u$$_{n}$" + "/" +"$u$$_{n}$" )
    ax2.plot(it_steps, convun50,'k')
    ax2.plot(it_steps, convthn50,'k--')
    ax2.plot(it_steps, convun90,'b')
    ax2.plot(it_steps, convunmed, 'm')
    ax2.plot(it_steps, convthn90,'b--')
    ax2.plot(it_steps, convun99,'g')
    ax2.plot(it_steps, convthn99,'g--')
    ax2.plot(it_steps, convthnmed, 'm--')
    ax2.legend(['Displacements','Rotations'],loc = 3)
    ax2.grid(True)

    rrel = r/max(abs(r))
    if abs(min(min(convrf), 0.001)) < 10 ** 12 and max(convrm) > 0 and max(convrm) < 10 ** 12:
        ax3.axis([2, numit+1, min(min(convrf),0.001), max(abs(convrm))])
    ax3.set_yscale("log")
    ax3.set_title("$Residual$")
    ax3.plot(it_steps, convrf, 'r')
    ax3.plot(it_steps, convrm, 'm')
    ax3.legend(["$R$$_{F}$", "$R$$_{M}$"], loc=3)
    ax3.grid(True)


    ax1.set_ylim(1e-15,10)
    ax2.set_ylim(1e-15,10)
    ax3.set_ylim(1e-5,1e8)
    
    plt.subplots_adjust(hspace=0.5)
    # plt.pause(1)
    cwd = os.getcwd()
    path_save_fig = os.path.join(cwd, 'deploying\\plots\\conv_plots')
    fig.savefig(os.path.join(path_save_fig, 'conv_plt_'+str(i)+NN))
    plt.close()

    if i == numit: 
        wandb.log({'conv_plt': plt})

def check_plots_perit(model_path, numit, sh_true, sh_pred, eh, sh_true_0=None, sh_pred_0=None, 
                      D_true=None, D_pred=None, norms_0 = None, sig = True, onlydiag = False):
    '''
    wrapper for saving 
        (1) diagonal plot per number of iteration during deployment
        (2) plot of the dataset plotting eps against sig (within the training data range)
    Note: Always takes the model path from range "_I". This is currently not an issue as all are the same.

    model_path      (str)       path to trained model
    numit           (int)       current number of iteration
    sh_true         (np.arr)    true values of sigma_g
    sh_pred         (np.arr)    predicted values of sigma_g
    D_true          (np.arr)    true values of D, expected shape: (1,8,8)
    D_pred          (np.arr)    predicted values of D, expected shape: (1,8,8)
    eh              (np.arr)    true values of epsilon_g
    sh_true_0       (np.arr)    true values of sigma_g of the first iteration step 
                                (for the axis bounds of the diagonal plots)
    norms_0         (np.arr)    color values for RSE of first iteration step
                                (such that they are always the same for all iterations.)
    sig             (bool)      if true: sig-plots, else: D-plots
    only_diag       (bool)      if true: only plots diagonal plots, not mike plots
    '''

    raise UserWarning('This has not yet been implemented for the THREEDIM case.')
    # TODO!

    if sig: 
        # 1 - diagonal plots sig
        cwd = os.getcwd()
        save_path = os.path.join(cwd, '05_Deploying\\plots\\diagonal_plots')
        path_stats = model_path['data']['_I']
        with open(os.path.join(path_stats, 'mat_data_stats.pkl'),'rb') as handle:
            stats = pickle.load(handle)
        lims_true = np.array([np.min(sh_true_0, axis=0), np.max(sh_true_0, axis=0)])
        lims_pred = np.array([np.min(sh_pred_0, axis=0), np.max(sh_pred_0, axis=0)])
        lims = np.vstack((np.minimum(lims_true[0], lims_pred[0]), np.maximum(lims_true[1], lims_pred[1]))).tolist()
        multiple_diagonal_plots(save_path, sh_true, sh_pred, 'u', stats, color = 'rse', numit = numit, xlim = lims, ylim = lims, norms_ = norms_0)

        # 2 - mike plots
        if not onlydiag:
            with open(os.path.join(path_stats, 'mat_data_np_TrainEvalTest.pkl'),'rb') as handle:
                mat_data_np_TrainEvalTest = pickle.load(handle)
            
            x_all_o = np.concatenate((mat_data_np_TrainEvalTest['X_train'], mat_data_np_TrainEvalTest['X_eval'], mat_data_np_TrainEvalTest['X_test']), axis=0)
            y_all_o = np.concatenate((mat_data_np_TrainEvalTest['y_train'], mat_data_np_TrainEvalTest['y_eval'], mat_data_np_TrainEvalTest['y_test']), axis=0)
            x_all = transf_units(x_all_o, 'eps-t', forward=False)[:,0:8]
            y_all = transf_units(y_all_o, 'sig', forward=False)[:,0:8]
            save_path_2 = os.path.join(cwd, '05_Deploying\\plots\\mike_plots')
            plots_mike_dataset(x_all, eh, eh, y_all, sh_true, sh_pred, save_path_2, 'sim', numit = numit)



    else:
        # 1 - diagonal plots D

        cwd = os.getcwd()
        save_path = os.path.join(cwd, '05_Deploying\\plots\\diagonal_plots_D')
        if D_true[0,0,0] == D_pred[0,0,0]: 
            save_path = os.path.join(cwd, '05_Deploying\\plots\\diagonal_plots_D_true')
        path_stats = model_path['data']['_I']
        with open(os.path.join(path_stats, 'mat_data_stats.pkl'),'rb') as handle:
            stats = pickle.load(handle)

            stats_y = stats['stats_y_train']
            if stats_y['std'].shape[0] <72:
                print(f"Output shape smaller than the expected (Shape: {stats_y['std'].shape[0]}). Expanding statistics dict to include zero-values.")
                if stats_y['std'].shape[0] == 2:
                    stats = add_zerovals_stats(stats, dim = 'ONEDIM-x')
                    print('Please note that the ONEDIM-y case cannot be plotted. A different input variable would need to be defined to cover this case.')
                elif stats_y['std'].shape[0] == 12: 
                    stats = add_zerovals_stats(stats, dim = 'TWODIM')

        lims = None
        norms_0 = None
        multiple_diagonal_plots_D(save_path, D_true, D_pred, 'u', stats, color = 'rse', numit = numit, xlim = lims, ylim = lims, norms_ = norms_0)

        # 2 - mike plots
        if not onlydiag:
            with open(os.path.join(path_stats, 'mat_data_np_TrainEvalTest.pkl'),'rb') as handle:
                mat_data_np_TrainEvalTest = pickle.load(handle)

            save_path_2 = os.path.join(cwd, '05_Deploying\\plots\\mike_plots_D')
            if D_true[0,0,0] == D_pred[0,0,0]: 
                save_path_2 = os.path.join(cwd, '05_Deploying\\plots\\mike_plots_D_true')

            x_all_o = np.concatenate((mat_data_np_TrainEvalTest['X_train'], mat_data_np_TrainEvalTest['X_eval'], mat_data_np_TrainEvalTest['X_test']), axis=0)
            y_all_o = np.concatenate((mat_data_np_TrainEvalTest['y_train'], mat_data_np_TrainEvalTest['y_eval'], mat_data_np_TrainEvalTest['y_test']), axis=0)
            x_all = transf_units(x_all_o, 'eps-t', forward=False)[:,0:8]
            y_all_re = transf_units(y_all_o[:,8:].reshape((-1,8,8)), 'D', forward=False, linel=False)
            y_all = np.concatenate((y_all_re[:,:6,:6].reshape((-1,36)), y_all_re[:,6,6].reshape((-1,1)), y_all_re[:,7,7].reshape((-1,1))), axis = 1)
            
            D_true_re = np.concatenate((D_true[:,:6,:6].reshape((-1, 36)), D_true[:,6,6].reshape((-1, 1)), D_true[:,7,7].reshape((-1,1))), axis = 1)
            D_pred_re = np.concatenate((D_pred[:,:6,:6].reshape((-1, 36)), D_pred[:,6,6].reshape((-1, 1)), D_pred[:,7,7].reshape((-1,1))), axis = 1)
            plots_mike_dataset(x_all, eh, eh, y_all, D_true_re, D_pred_re, save_path_2, tag = 'sim', tag2 = 'D', numit = numit)
    


    return

def eps_sig_plots(i, numit, eh_cum, u_cum, sh_cum, eh_cum_ = None, sh_cum_= None, NN = 'NN', tag = 2):
    '''
    plotting the evolvement of epsilon and sigma throughout the iterations
    i           (int)       current iteration step
    numit       (int)       total amount of iterations
    eh_cum      (np.arr)    cumulative general. strains for every iteration step, shape: (numit, eh.shape)
    sh_cum      (np.arr)    cumulative general. stresses for every iteration step, shape: (numit, sh.shape)  
    eh_cum_     (np.arr)    cumulative general. strains for every iteration step, shape: (numit, eh_.shape)
                            used only in NN-sigma deployment, when calculating stresses via layers
    sh_cum_     (np.arr)    cumulative general. stresses for every iteration step, shape: (numit, sh_.shape)
                            used only in NN-sigma deployment, when calculating stresses via layers
    NN          (str)       either 'NN' or 'NLFEA'
    tag         (int)       integer between 0 and 7, to define which strain / stress shall be plotted (n,m,v, etc.)     
    '''
    raise RuntimeWarning('This function is outdated. Please use all_eps_sig_plots')
    numit_vec = np.arange(0, numit+1, 1)

    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax1.set_xlim([1,numit])
    ax2.set_xlim([1,numit])

    eh_cum_re = eh_cum.reshape(numit, eh_cum.shape[1], -1, 8)
    sh_cum_re = sh_cum.reshape(numit, sh_cum.shape[1], -1, 8)
    if sh_cum_ is not None:
        eh_cum_re_ = eh_cum.reshape(numit, eh_cum_.shape[1], -1, 8)
        sh_cum_re_ = sh_cum.reshape(numit, sh_cum_.shape[1], -1, 8)

    # for the moment, just plotting maximum values of eh, sh over all elements in the FEA

    markers = ['.', '_', '|', 'v']

    for j in range(eh_cum_re.shape[2]):         # iterate over amount of gausspoints (either 1 or 4)
        ax1.plot(numit_vec[:i+1]+1, np.max(eh_cum_re[:i+1,:,j,tag], axis=1), label='eh, gp'+str(j), marker = markers[j])
        ax2.plot(numit_vec[:i+1]+1, np.max(sh_cum_re[:i+1,:,j,tag], axis=1), label='sh, gp'+str(j), marker = markers[j])
        if NN == 'NN':
            ax1.plot(numit_vec[:i+1]+1, np.max(eh_cum_re_[:i+1,:,j,tag], axis=1), '--', label='eh_, gp'+str(j))
            ax2.plot(numit_vec[:i+1]+1, np.max(sh_cum_re_[:i+1,:,j,tag], axis=1), '--', label='sh_, gp'+str(j))
    
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Generalised \n strains')
    ax1.grid(True)
    ax1.legend()

    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Generalised \n stresses')
    ax2.grid(True)
    ax2.legend()

    
    
    # Saving plot
    cwd = os.getcwd()
    path_save_fig = os.path.join(cwd, 'plots\\eps_sig_plots_it')
    fig.savefig(os.path.join(path_save_fig, 'eps_sig_'+str(i)+NN))
    print('saved sig-eps-it plots')
    plt.close()

    return

def all_eps_sig_plots(i, numit, eh_cum, u_cum, sh_cum, eh_cum_ = None, sh_cum_= None, eh_cum_NLFEA = None, sh_cum_NLFEA = None, 
                      NN = 'NN', tag = 'max', final = None):
    '''
    same as eps_sig_plots but for all 8 variables
    '''
    fig, axes = plt.subplots(3, 2, figsize=(12,8))
    plt.suptitle(NN+'\n (only maximum values for all stresses)')

    numit_vec = np.arange(0, numit+2, 1)

    eh_cum_re = eh_cum.reshape(numit+1, eh_cum.shape[1], -1, 8)
    sh_cum_re = sh_cum.reshape(numit+1, sh_cum.shape[1], -1, 8)
    if sh_cum_ is not None:
        eh_cum_re_ = eh_cum_.reshape(numit+1, eh_cum_.shape[1], -1, 8)
        sh_cum_re_ = sh_cum_.reshape(numit+1, sh_cum_.shape[1], -1, 8)
    if sh_cum_NLFEA is not None: 
        eh_cum_re_NLFEA = eh_cum_NLFEA.reshape(numit+1, eh_cum_NLFEA.shape[1], -1, 8)
        sh_cum_re_NLFEA = sh_cum_NLFEA.reshape(numit+1, sh_cum_NLFEA.shape[1], -1, 8)

    # add one row of zeros, to get 9 axes for the 3 plots
    eh_cum_re = np.concatenate((eh_cum_re, np.zeros((*eh_cum_re.shape[:-1], 1))), axis=-1)
    sh_cum_re = np.concatenate((sh_cum_re, np.zeros((*sh_cum_re.shape[:-1], 1))), axis=-1)
    if sh_cum_ is not None: 
        eh_cum_re_ = np.concatenate((eh_cum_re_, np.zeros((*eh_cum_re_.shape[:-1], 1))), axis=-1)
        sh_cum_re_ = np.concatenate((sh_cum_re_, np.zeros((*sh_cum_re_.shape[:-1], 1))), axis=-1)
    if sh_cum_NLFEA is not None: 
        eh_cum_re_NLFEA = np.concatenate((eh_cum_re_NLFEA, np.zeros((*eh_cum_re_NLFEA.shape[:-1], 1))), axis=-1)
        sh_cum_re_NLFEA = np.concatenate((sh_cum_re_NLFEA, np.zeros((*sh_cum_re_NLFEA.shape[:-1], 1))), axis=-1)
    
    markers = ['.', '_', '|', 'v']
    colors = ['red', 'green', 'blue']
    colors_ = ['lightcoral','lightgreen', 'lightblue']
    # labels_e = ['eps_x', 'eps_y', 'eps_xy', 'chi_x', 'chi_y', 'chi_xy', 'gam_x', 'gam_y', 'xx']
    labels_e = ['$\{\hat{\epsilon}, \hat{\sigma}\}_x$', '$\{\hat{\epsilon}, \hat{\sigma}\}_y$', '$\{\hat{\epsilon}, \hat{\sigma}\}_{xy}$', 
                ' ', ' ', ' ',
                ' ',  ' ', ' ']
    labels_s = ['nx', 'ny', 'nxy', 'mx', 'my', 'mxy', 'vx', 'vy', 'xx']
    labels_glob_s = ['Normal forces $n$ [N/mm]', 'Bending moments $m$ [Nmm/mm]', 'Shear forces $v$ [N/mm]']
    labels_glob_e = ['Normal strains $\epsilon$ [-]', 'Curvatures $\chi$ [mrad]', 'Shear strains $\gamma$ [-]']

    if tag == 'max': 
        # use max values of array for plotting
        y_e = np.max(eh_cum_re[:i,:,:,:], axis=1)
        y_s = np.max(sh_cum_re[:i,:,:,:], axis=1)
        if NN == 'NN':
            y_e_ = np.max(eh_cum_re_[:i,:,:,:], axis=1)
            y_s_ = np.max(sh_cum_re_[:i,:,:,:], axis=1)
        if sh_cum_NLFEA is not None:
            y_e_NLFEA = np.max(eh_cum_re_NLFEA[:i,:,:,:], axis=1)
            y_s_NLFEA = np.max(sh_cum_re_NLFEA[:i,:,:,:], axis=1)
    else: 
        # use value given with 'tag' for plotting location
        y_e = eh_cum_re[:i,tag,:,:]
        y_s = sh_cum_re[:i,tag,:,:]
        if NN == 'NN': 
            y_e_ = eh_cum_re_[:i,tag,:,:]
            y_s_ = sh_cum_re_[:i,tag,:,:]
        if sh_cum_NLFEA is not None:
            y_e_NLFEA = eh_cum_re_NLFEA[:i,tag,:,:]
            y_s_NLFEA = sh_cum_re_NLFEA[:i,tag,:,:]

    for k in range(3):
        for j in range(eh_cum_re.shape[2]):         # iterate over amount of gausspoints (either 1 or 4)
            for l in range(3*k,3*k+3):

                if k == 2 and l == (3*k+2):
                    # don't need the last row of shear.
                    pass
                else: 
                    axes[k,0].plot(numit_vec[:i], y_e[:,j,l], 
                                '--', label=labels_e[l] + ' NN' if j == 0 else None,
                                marker = markers[j], color = colors[l-3*k])
                    axes[k,1].plot(numit_vec[:i], y_s[:,j,l], 
                                '--', label= labels_s[l]+ ' NN' if j == 0 else None, 
                                marker = markers[j], color = colors[l-3*k])
                    if NN == 'NN':
                        axes[k,0].plot(numit_vec[:i], y_e_[:,j,l],
                                        ':', label=labels_e[l]+' check' if j==0 else None, 
                                        color = colors_[l-3*k])
                        axes[k,1].plot(numit_vec[:i], y_s_[:,j,l],
                                        ':', label= labels_s[l]+ ' check' if j == 0 else None,
                                        color = colors_[l-3*k])
                    if sh_cum_NLFEA is not None: 
                        axes[k,0].plot(numit_vec[:i], y_e_NLFEA[:,j,l],
                                        '-', label=labels_e[l]+' NLFEA' if j==0 else None, 
                                        color = colors[l-3*k])
                        axes[k,1].plot(numit_vec[:i], y_s_NLFEA[:,j,l],
                                        '-', label= labels_s[l]+ 'NLFEA' if j == 0 else None,
                                        color = colors[l-3*k])

        
        axes[k,0].set_xlabel('Iterations')
        # axes[k,0].legend(loc = 'upper left', bbox_to_anchor=(1,1))
        axes[k,0].grid(True)
        axes[k,0].set_ylabel(labels_glob_e[k])

        axes[k,1].set_xlabel('Iterations')
        # axes[k,1].legend(loc = 'upper left', bbox_to_anchor=(1,1))
        axes[k,1].grid(True)
        axes[k,1].set_ylabel(labels_glob_s[k])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.85, 0.5))

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to fit legend

    cwd = os.getcwd()
    if final == None:
        path_save_fig = os.path.join(cwd, '05_Deploying\\plots\\eps_sig_plots_it')
        fig.savefig(os.path.join(path_save_fig, 'eps_sig_'+str(i)+NN))
    else: 
        path_save_fig = os.path.join(cwd, '05_Deploying\\data_out\\'+final+'\\eps_sig_plots_it')
        fig.savefig(os.path.join(path_save_fig, 'eps_sig_final')+NN)

    print('saved sig-eps-it plots')
    plt.close()

    return

def all_u_fi_plots(i, numit, u_cum, fi_cum, u_NLFEA = None, fi_NLFEA = None, NN = 'NN', tag = 'max'):
    '''same as all_eps_sig_plots but for u'''
    fig, axes = plt.subplots(3,1, figsize = (12,8))
    plt.suptitle(NN)

    numit_vec = np.arange(0, numit+2, 1)

    # no reshaping required. u and fi given at nodes.
    # u shape: (726,1) = (121x6,1) = (11x11x6,1)
    # have always [ux, uy, uz, thx, thy, thz] for every node, after eachother.
    # in the current scenario only uz is relevant --> look at this plot first, later expand to all 6 deform / angles
    
    uz = u_cum[:,2::6,0]
    fz = fi_cum[:,2::6,0]

    # plotting uz, fz:
    if tag == 'max':
        axes[0].plot(numit_vec[:i+1]+1, np.max(uz[:i+1,:], axis=1), '--', label='uz')
        axes[1].plot(numit_vec[:i+1]+1, np.max(fz[:i+1,:], axis=1), '--', label='fz')
    

    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('uz')
    axes[0].grid(True)
    axes[0].legend()
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel('fz')
    axes[1].grid(True)
    axes[1].legend()
    
    # Saving plot
    cwd = os.getcwd()
    path_save_fig = os.path.join(cwd, '05_Deploying\\plots\\u_fi_plots_it')
    fig.savefig(os.path.join(path_save_fig, 'u_fz_'+str(i)+NN))
    print('saved u-fz-it plots')
    plt.close()

    return

def all_De_plots(k, numit, eh_cum = None, eh_cum_ = None, eh_cum_NLFEA= None, De_cum=None, De_cum_ = None, De_cum_NLFEA=None,
                 tag = 'max'):
    '''
    eh_cum          (np.array)      Strain calculated with NLFEA (in first iteration), then follows from prediction of D, calculation of u
    eh_cum_         (np.array)      Strain calculated with NLFEA (in first iteration), then follows from calculation with D_, u_
    De_cum          (np.array)      Stiffness predicted with NN, expected shape = (7, 100, 2, 2, 8, 8)
    De_cum_         (np.array)      Stiffness calculated with NLFEA but based on inputs with NN iteration, expected shape = (7, 100, 2, 2, 8, 8)
    De_cum_NLFEA    (np.array)      Stiffness calculated with NLFEA, expected shape = (7, 100, 2, 2, 8, 8)
    numit           (int)           Total number of iterations
    k               (int)           Current number of iteration
    
    '''

    blocks = [
    (0, 0),  # Top-left
    (0, 3),  # Top-right
    (3, 0),  # Bottom-left
    (3, 3)   # Bottom-right
    ]

    block_names = ['D_m [N/mm]', 'D_mb [N]', 'D_bm [N]', 'D_b [Nmm]']
    strain_names = ['Membrane strain', 'Curvature', 'Shear strain']

    # Reshaping array:
    # Checking only for one gauss point and for the maximum value of all elements
    eh_cum_re = np.concatenate((np.max(eh_cum[:k,:,0,0,:8], axis = 1), np.zeros((k,1))), axis=1)
    eh_cum_re_ = np.concatenate((np.max(eh_cum_[:k,:,0,0,:8], axis = 1), np.zeros((k,1))),axis=1)
    if eh_cum_NLFEA is not None: 
        eh_cum_re_NLFEA = np.concatenate((np.max(eh_cum_NLFEA[:k,:,0,0,:8], axis = 1), np.zeros((k,1))),axis=1)
    if De_cum is not None:
        De_cum_66 = np.max(De_cum[:k,:,0,0,:6,:6], axis = 1)
        De_cum_s = np.concatenate((np.max(De_cum[:k,:,0,0,6,6], axis=1).reshape((-1,1,1)), np.max(De_cum[:k,:,0,0,7,7], axis=1).reshape((-1,1,1))), axis = 2)
        De_cum_66_ = np.max(De_cum_[:k,:,0,0,:6,:6], axis = 1)
        De_cum_s_ = np.concatenate((np.max(De_cum_[:k,:,0,0,6,6], axis=1).reshape((-1,1,1)), np.max(De_cum_[:k,:,0,0,7,7], axis=1).reshape((-1,1,1))), axis = 2)
    if De_cum_NLFEA is not None: 
        De_cum_66_NLFEA = np.max(De_cum_NLFEA[:k,:,0,0,:6,:6], axis = 1)
        De_cum_s_NLFEA = np.concatenate((np.max(De_cum_NLFEA[:k,:,0,0,6,6], axis=1).reshape((-1,1,1)), np.max(De_cum_NLFEA[:k,:,0,0,7,7], axis=1).reshape((-1,1,1))), axis = 2)

    # Start plotting
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(3, 3, height_ratios=[1, 1, 0.6], hspace=0.5)

    axes = [
        # axes for epsilon
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[2, 0]),
        
        # axes for D 
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[2, 2]),
    ]

    x = np.arange(numit)
    colors = [viridis(i / 8) for i in range(9)]

    if De_cum is not None: 
        # Strains (inputs to NN)
        for idx in range(3): 
            ax = axes[idx]
            color_idx = 0
            y = eh_cum_re[:k,idx:idx+3]
            y_ = eh_cum_re_[:k, idx:idx+3]
            if De_cum_NLFEA is not None:
                y_NLFEA = eh_cum_re_NLFEA[:k,idx:idx+3]
            for m in range(3):
                if De_cum_NLFEA is not None:
                    ax.plot(x[:k],y_NLFEA[:,m], color = colors[2*m], label = f"{m}")
                ax.plot(x[:k], y[:,m], color = colors[2*m], linestyle = 'dashed', marker = '.', markevery = (0.05, 0.1))
                ax.plot(x[:k], y_[:,m], color = colors[2*m], linestyle = 'dotted', marker = 'x', markevery=0.1)

            ax.set_title(strain_names[idx])
            ax.set_xlabel('Iteration step')
            ax.set_ylabel(strain_names[idx])
            ax.legend(loc="upper right", fontsize=8)


        # Stiffness terms (6x6)
        for idx, (i, j) in enumerate(blocks):
            ax = axes[idx+3]
            block = De_cum_66[:k, i:i+3, j:j+3]  # shape (7, 3, 3)
            block_ = De_cum_66_[:k, i:i+3, j:j+3]
            if De_cum_NLFEA is not None:
                block_NLFEA = De_cum_66_NLFEA[:k, i:i+3, j:j+3]
    
            color_idx = 0
            for m in range(3):
                for n in range(3):
                    y = block[:, m, n]
                    y_ = block_[:,m,n]
                    ax.plot(x[:k], y, color=colors[color_idx],linestyle = 'dashed', marker = '.', markevery = (0.05, 0.1))
                    ax.plot(x[:k], y_, color=colors[color_idx], linestyle = 'dotted', marker = 'x', markevery=0.1)
                    if De_cum_NLFEA is not None: 
                        y_NLFEA = block_NLFEA[:,m,n]
                        ax.plot(x[:k], y_NLFEA, color=colors[color_idx], label=f"({i+m},{j+n})")
                    color_idx += 1
            ax.set_title(f" Stiffness matrix {block_names[idx]}")
            ax.set_xlabel("Iteration step")
            ax.set_ylabel("Stiffness")
            ax.legend(loc="upper right", fontsize=8)

        # Shear stiffness
        ax_extra = axes[7]
        for col in range(2):
            y = De_cum_s[:k, 0, col]
            y_ = De_cum_s_[:k,0,col]
            ax_extra.plot(x[:k], y, color=colors[col*3], linestyle = 'dashed', marker = '.', markevery = (0.05, 0.1))
            ax_extra.plot(x[:k], y_, color=colors[col*3], linestyle = 'dotted', marker = 'x', markevery=0.1)
            if De_cum_NLFEA is not None: 
                y_NLFEA = De_cum_s_NLFEA[:k,0,col]
                ax_extra.plot(x[:k], y_NLFEA, color = colors[col*3], label=f"D_s({col+1},{col+1})")

        ax_extra.set_title("Stiffness matrix D_s [N/mm]")
        ax_extra.set_xlabel("Iteration step")
        ax_extra.set_ylabel("Stiffness")
        ax_extra.legend(loc="upper right", fontsize=10)

        axes[8].axis('off')
        at_ = AnchoredText('continuous line = NLFEA' + '\n'+
                       'dashed line = NN' + '\n'+
                       'dotted line = mixture',
                        prop=dict(size=10), frameon=True, loc='center')
        at_.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
        axes[8].add_artist(at_)

    elif De_cum_NLFEA is not None: 
        # Strains (inputs to NN)
        for idx in range(3): 
            ax = axes[idx]
            color_idx = 0
            y_NLFEA = eh_cum_re_NLFEA[:k,idx:idx+3]
            for m in range(3):
                ax.plot(x[:k],y_NLFEA[:,m], color = colors[2*m], label = f"{m}")

        ax.set_title(strain_names[idx])
        ax.set_xlabel('Iteration step')
        ax.set_ylabel(strain_names[idx])
        ax.legend(loc="upper right", fontsize=8)
        

        # Stiffnesses
        for idx, (i, j) in enumerate(blocks):
            ax = axes[idx+3]
            block_NLFEA = De_cum_66_NLFEA[:k, i:i+3, j:j+3]
    
            color_idx = 0
            for m in range(3):
                for n in range(3):
                    y_NLFEA = block_NLFEA[:,m,n]
                    ax.plot(x[:k], y_NLFEA, color=colors[color_idx], label=f"({i+m},{j+n})")
                    color_idx += 1
            ax.set_title(f" Stiffness matrix {block_names[idx]}")
            ax.set_xlabel("Iteration step")
            ax.set_ylabel("Stiffness")
            ax.legend(loc="upper right", fontsize=8)

        ax_extra = axes[7]
        for col in range(2):
            y_NLFEA = De_cum_s_NLFEA[:k,0,col]
            ax_extra.plot(x[:k], y_NLFEA, color = colors[col*3], label=f"D_s({col+1},{col+1})")

        ax_extra.set_title("Stiffness matrix D_s [N/mm]")
        ax_extra.set_xlabel("Iteration step")
        ax_extra.set_ylabel("Stiffness")
        ax_extra.legend(loc="upper right", fontsize=10)

        axes[8].axis('off')
        at_ = AnchoredText('continuous line = NLFEA',
                        prop=dict(size=10), frameon=True, loc='center')
        at_.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
        axes[8].add_artist(at_)


    else: 
        raise UserWarning('At least one not-None value needs to be given for plotting.')

    
    cwd = os.getcwd()
    path_save_fig = os.path.join(cwd, '05_Deploying\\plots\\De_plots_it')
    if De_cum is not None and De_cum_NLFEA is not None:
        fig.savefig(os.path.join(path_save_fig, 'De_'+str(k)+'NN vs NLFEA'))
    if De_cum is not None: 
        fig.savefig(os.path.join(path_save_fig, 'De_'+str(k)+'NN'))
    elif De_cum_NLFEA is not None: 
        fig.savefig(os.path.join(path_save_fig, 'De_'+str(k)+'NLFEA'))

    print('saved De_it plots')

    plt.close()

    return

def imshow_plots(coord, ms, L, sh_cum_NN = None, eh_cum_NN = None, sh_cum_NLFEA = None, eh_cum_NLFEA = None):
    num_elem = int(L/ms)
    matrix_aux = np.zeros((num_elem,num_elem))
    if sh_cum_NN is not None: 
        matrix_aux_sh_NN = np.zeros((num_elem,num_elem))
    if eh_cum_NN is not None: 
        matrix_aux_eh_NN = np.zeros((num_elem,num_elem))
    if sh_cum_NLFEA is not None: 
        matrix_aux_sh_NLFEA = np.zeros((num_elem,num_elem))
    if eh_cum_NLFEA is not None: 
        matrix_aux_eh_NFEA = np.zeros((num_elem, num_elem))

    # Creating matrix
    for i in range(num_elem**2): 
        index_new_x = int(np.round((coord['c'][0][i][0]-ms/2)/ms,0))
        index_new_y = int(np.round((coord['c'][0][i][1]-ms/2)/ms,0))
        matrix_aux[index_new_x, index_new_y] = i

    # Plotting
    fig, ax = plt.subplots()
    cax = ax.imshow(matrix_aux, cmap = 'viridis', interpolation = None)
    plt.colorbar(cax)

    for i in range(matrix_aux.shape[0]):
        for j in range(matrix_aux.shape[1]):
            ax.text(j, i, f"{matrix_aux[i, j]:.0f}", ha='center', va='center', color='white', fontsize=10)

    cwd = os.getcwd()
    path_save_fig = os.path.join(cwd, 'plots')
    fig.savefig(os.path.join(path_save_fig, 'element_numbers'))
    print('saved element numbers plots')
    
    return



################################## Auxiliary functions for convergence values ##################################

def un_thn(u):
    un = np.array([])
    thn = np.array([])
    count = 0
    for j in range(len(u)):
        if count < 2.5:
            un = np.append(un,u[j],axis=0)
        else:
            thn = np.append(thn,u[j],axis=0)
        count += 1
        if count > 5.5:
            count = 0
    return un,thn

def convergence_values_un_thn(u, unold, thnold):
    '''- [un ,thn]: node displacements and rotations separated"
       - maxun: maximum absolute value of displacement for which relative changes are tracked. "
                Defined in order not to track changes in (close to) zero values"
       - diffun: un(i) - un(i-1), difference in displacement between iteration steps, for displ. > maxun"
       - relun: relative difference: diffun./un(i) at locations where abs(un(i)) > maxun"
    '''

    [un, thn] = un_thn(u)
    diffun = np.zeros_like(un)
    relun = np.zeros_like(un)
    maxun = np.max(abs(un)) / 1000
    diffun[np.where(abs(un) > maxun)] = np.ndarray.flatten(unold[np.where(abs(un) > maxun)]) - np.ndarray.flatten(
        un[np.where(abs(un) > maxun)])
    relun[np.where(abs(un) > maxun)] = np.divide(diffun[np.where(abs(un) > maxun)],
                                                np.ndarray.flatten(un[np.where(abs(un) > maxun)]))
    diffun[np.where(abs(un) < maxun)] = 0
    relun[np.where(abs(un) < maxun)] = 0


    diffthn = np.zeros_like(un)
    relthn = np.zeros_like(un)
    maxthn = np.max(abs(thn)) / 1000
    diffthn[np.where(abs(thn) > maxthn)] = np.ndarray.flatten(thnold[np.where(abs(thn) > maxthn)]) - np.ndarray.flatten(
        thn[np.where(abs(thn) > maxthn)])
    relthn[np.where(abs(thn) > maxthn)] = np.divide(diffthn[np.where(abs(thn) > maxthn)],
                                                    np.ndarray.flatten(thn[np.where(abs(thn) > maxthn)]))
    diffthn[np.where(abs(thn) < maxthn)] = 0
    relthn[np.where(abs(thn) < maxthn)] = 0

    convun50_ = np.percentile(abs(relun),50)
    convun90_ = np.percentile(abs(relun), 90)
    convun99_ = np.percentile(abs(relun), 99)
    convthn50_ = np.percentile(abs(relthn),50)
    convthn90_ = np.percentile(abs(relthn), 90)
    convthn99_ = np.percentile(abs(relthn), 99)
    convun_med = np.median(abs(relun))
    convthn_med = np.median(abs(relthn))



    mat_convergence = {
        'un': un, 
        'thn': thn,
        'diffun': diffun,
        'relun': relun,
        'diffthn': diffthn,
        'relthn': relthn,
        'convun50': convun50_,
        'convun90': convun90_,
        'convun99': convun99_,
        'convthn50': convthn50_,
        'convthn90': convthn90_,
        'convthn99': convthn99_,
        'convun_med': convun_med,
        'convthn_med': convthn_med
    }

    return mat_convergence

def convergence_values_eps_r(e_conv, eold, r, mat_convergence_eps_r, i):
    '''
    - maxe: maximum absolute value of strain for which relative changes are tracked. 
            Defined in order not to track changes in (close to) zero values"
    - diffe: e(i) - e(i-1), difference in strain between iteration steps, for strains > maxe
    - rele: relative difference: diffe./e(i) at locations where abs(e(i)) > maxe
    '''

    e_conv[e_conv<-99999] = 0
    maxe = np.max(abs(e_conv))/1000
    diffe = np.ndarray.flatten(eold[np.where(abs(e_conv)>maxe)])-np.ndarray.flatten(e_conv[np.where(abs(e_conv)>maxe)])
    rele = np.divide(diffe,np.ndarray.flatten(e_conv[np.where(abs(e_conv)>maxe)]))

    

    # Residuals 
    # Convergence Control: Plot Iteration Step and Display Sum of Residual Forces

    if i ==0: 
        convrf = sum(abs(r[0::6]))+sum(abs(r[1::6]))+sum(abs(r[2::6]))
        convrm = sum(abs(r[3::6])) + sum(abs(r[4::6])) + sum(abs(r[5::6]))   
    else: 
        convrm_ = mat_convergence_eps_r['convrm']
        convrf_ = mat_convergence_eps_r['convrf']
        convrf = np.append(convrf_, sum(abs(r[0::6]))+sum(abs(r[1::6]))+sum(abs(r[2::6])))
        convrm = np.append(convrm_, sum(abs(r[3::6])) + sum(abs(r[4::6])) + sum(abs(r[5::6])))

    conve50_ = np.percentile(abs(rele),50)
    conve90_ = np.percentile(abs(rele), 90)
    conve99_ = np.percentile(abs(rele), 99)

    mat_convergence ={
        'e': e_conv,
        'diffe': diffe,
        'rele': rele,
        'convrf': convrf,
        'convrm': convrm,
        # 'convrf_min': min(min(convrf)),
        # 'convrm_min': min(min(convrm)),
        'conve50': conve50_,
        'conve90': conve90_,
        'conve99': conve99_,
    }

    return mat_convergence




################################## Other auxiliary functions ##################################

def scatter_vb(path, input, input_= None, numit=0, save_path = None, tag = 'eps-eps',
                scatter_all = False, data_type = 'test', filter = None, range_ = None, errors = None):
    '''
    plot input variables of entire dataset (train, eval, test) against 
    path            (str)       to model number (and the entire dataset)
    input           (arr)       input into NN (for given iteration numit)
    input_          (dict)      additional data to plot (e.g. for )
    numit           (int)       current iteration number (start with 0)
    save_path       (str)       location to save figure to
    tag             (str)       'eps-eps': plots input against input; 
                                'eps-D': plots input against all variables in input_

    scatter_all     (bool)      True: create scatter plots for all 64 plots
                                False: creates histograms on diagonal (--> only use if x_data = y_data)
    data_type       (str)       can be train or test to display the training or testing data in the background
    filter          (arr)       filter the data based on given geometrical parameters (in input, parameters 8:11)
    range           (arr)       cutoff the range of the epsilon values given by this array
    errors          (bool)      if errors: also calculate the errors per row between predicted and calculated stiffness and 
                                color it according to colorscale of diagonal scatter plots (one colorbar per row)
    '''
    
    raise UserWarning('This has not yet been implemented for the THREEDIM case.')
    # TODO!
    
    if tag == 'eps-eps':
        label_x = np.array([r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$',
                            r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$',
                            r'$\gamma_x$', r'$\gamma_y$', r'$t$', r'$\rho$', r'$CC$'
                            ])
        label_y = label_x
        input_x = input
        input_y = {
            '1': input,
            '2': None
        }

        # collect all training / test data
        with open(os.path.join(path, 'mat_data_np_TrainEvalTest.pkl'),'rb') as handle:
                mat_data_np = pickle.load(handle)
        if data_type == 'test':
            input_all_raw = mat_data_np['X_test'][:,0:11]
        elif data_type == 'train':
            input_all_raw = mat_data_np['X_train'][:,0:11]
        input_all = transf_units(input_all_raw, 'eps-t', forward = False, linel=False)
        input_all_x = input_all
        input_all_y = input_all
        numit_y = numit


    elif tag == 'eps-D':
        numit_y = numit + 1

        label_x = np.array([r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$',
                            r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$',
                            r'$\gamma_x$', r'$\gamma_y$', r'$t$', r'$\rho$', r'$CC$'
                            ])
        label_y = np.array(['$D_{m,11}$', '$D_{m,12}$', '$D_{m,13}$', '$D_{mb,11}$', '$D_{mb,12}$', '$D_{mb,13}$',
                        '$D_{m,21}$', '$D_{m,22}$', '$D_{m,23}$', '$D_{mb,21}$', '$D_{mb,22}$', '$D_{mb,23}$',
                        '$D_{m,31}$', '$D_{m,32}$', '$D_{m,33}$', '$D_{mb,31}$', '$D_{mb,32}$', '$D_{mb,33}$',
                        '$D_{bm,11}$', '$D_{bm,12}$', '$D_{bm,13}$', '$D_{b,11}$', '$D_{b,12}$', '$D_{b,13}$',
                        '$D_{bm,21}$', '$D_{bm,22}$', '$D_{bm,23}$', '$D_{b,21}$', '$D_{b,22}$', '$D_{b,23}$',
                        '$D_{bm,31}$', '$D_{bm,32}$', '$D_{bm,33}$', '$D_{b,31}$', '$D_{b,32}$', '$D_{b,33}$',
                        '$D_{s,11}$', '$D_{s,22}$'
                        ])
        
        input_x = input
        input_y = input_

        # collect all training / test data
        with open(os.path.join(path, 'mat_data_np_TrainEvalTest.pkl'),'rb') as handle:
                mat_data_np = pickle.load(handle)
        if data_type == 'test':
            input_all_raw_x = mat_data_np['X_test'][:,0:11]
            input_all_raw_y = mat_data_np['y_test'][:,8:72].reshape((-1,8,8))
        elif data_type == 'train':
            input_all_raw_x = mat_data_np['X_train'][:,0:11]
            input_all_raw_y = mat_data_np['y_train'][:,8:72].reshape((-1,8,8))
        input_all_x = transf_units(input_all_raw_x, 'eps-t', forward = False, linel=False)
        input_all_y_re = transf_units(input_all_raw_y, 'D', forward = False, linel = False)
        input_all_y = np.concatenate((input_all_y_re[:,:6,:6].reshape((-1,36)), input_all_y_re[:,6,6].reshape((-1,1)), input_all_y_re[:,7,7].reshape((-1,1))), axis = 1)
        print(f'Smallest strain value in proximity of zero: {np.min(abs(input_all_x[:,:8]), axis = 0)}')

    elif tag == 'eps-sig':
        numit_y = numit

        label_x = np.array([r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$',
                            r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$',
                            r'$\gamma_x$', r'$\gamma_y$', r'$t$', r'$\rho$', r'$CC$'
                            ])
        label_y = np.array(['$n_x$', '$n_y$', '$n_{xy}$', 
                            '$m_x$', '$m_y$', '$m_{xy}$', 
                            '$v_x$', '$v_y$'
                            ])
        
        input_x = input
        input_y = input_

        # collect all training / test data
        with open(os.path.join(path, 'mat_data_np_TrainEvalTest.pkl'),'rb') as handle:
                mat_data_np = pickle.load(handle)
        if data_type == 'test':
            input_all_raw_x = mat_data_np['X_test'][:,0:11]
            input_all_raw_y = mat_data_np['y_test'][:,0:8]
        elif data_type == 'train':
            input_all_raw_x = mat_data_np['X_train'][:,0:11]
            input_all_raw_y = mat_data_np['y_train'][:,0:8]
        input_all_x = transf_units(input_all_raw_x, 'eps-t', forward = False, linel=False)
        input_all_y = transf_units(input_all_raw_y, 'sig', forward = False, linel = False)
        print(f'Smallest strain value in proximity of zero: {np.min(abs(input_all_x[:,:8]), axis = 0)}')


    # filtering or specifying a range which shall be plotted in the input space to reduce the size of the test/train dataset
    if range_ is not None: 
        mask = (input_all_x[:,:8] >= range_[0,:]) & (input_all_x[:,:8]<=range_[1,:])
        valid_rows = mask.all(axis = 1)
        if np.sum(valid_rows) == 0:
            print('No points found for this region. Please increase range.\n ' \
            'Plotting all values of input, not just in the given range.')
        else: 
            print('Amount of rows after masking for ranges of epsilon: ', np.sum(valid_rows))
            input_all_x = input_all_x[valid_rows]
            input_all_y = input_all_y[valid_rows]
            print(f'Smallest strain value in proximity of zero after masking range: {np.min(abs(input_all_x[:,:8]), axis = 0)}')
    if filter is not None:
        mask_f = (abs(input_all_x[:,8:11] - filter) < 1e-5)
        valid_rows_f = mask_f.all(axis = 1)
        if np.sum(valid_rows_f) == 0:
            print('No points found for the defined filter. Please change filter.\n ' \
            'Plotting all values of training input, not just of the given filter.')
        else: 
            print('Amount of rows after filtering:', np.sum(valid_rows_f))
            input_all_x = input_all_x[valid_rows_f]
            input_all_y = input_all_y[valid_rows_f]
            print(f'Smallest strain value in proximity of zero after filtering: {np.min(abs(input_all_x[:,:8]), axis = 0)}')
    if errors is not None: 
        if input_y['1'].any() == None or input_y['2'].any() == None: 
            raise Warning('No prediction available, not executing error calculation.')
        real = input_y['2'][numit_y,:,:]
        pred = input_y['1'][numit_y,:,:]
        with open(os.path.join(path, 'mat_data_stats.pkl'),'rb') as handle:
                stats = pickle.load(handle)
        errors_calc = calculate_errors(real, pred, stats, transf='u', id = 'De-NLRC')
        norms = [mcolors.Normalize(vmin=np.min(errors_calc['rse'][:, i]),
                           vmax=np.max(errors_calc['rse'][:, i]))
                            for i in range(pred.shape[1])]
        scatters = []

    # plotting the data
    fig, ax = plt.subplots(input_y['1'].shape[2], input_x.shape[2], figsize=(input_x.shape[2]*2, input_y['1'].shape[2]*2))
    for i in range(input_y['1'].shape[2]):
        for j in range(input_x.shape[2]):
            ax[i, j].tick_params(axis='both', labelsize=6)
            if i != j: 
                if errors == None:
                    ax[i,j].scatter(input_all_x[:,j], input_all_y[:,i], color = 'blue', alpha = 0.1, s = 4, label = 'dataset '+data_type)
                    if input_y['2'] is not None:
                        ax[i,j].scatter(input_x[numit,:,j], input_y['2'][numit_y,:,i], color = 'coral', alpha = 0.1, s=4, label = 'deployment NN_')
                    ax[i,j].scatter(input_x[numit,:,j], input_y['1'][numit_y,:,i], color = 'lightblue', alpha = 0.1, s=4, label = 'deployment NN')
                else:
                    ax[i,j].scatter(input_all_x[:,j], input_all_y[:,i], color = 'blue', 
                                    alpha = 0.1, s = 4, label = 'dataset '+data_type)
                    scatter = ax[i,j].scatter(input_x[numit,:,j], input_y['1'][numit_y,:,i], 
                                            c = errors_calc['rse'][:, i], cmap = 'plasma', norm = norms[i], 
                                            alpha = 0.4, s=4, label = 'deployment NN')
                    if j == 0:
                        scatters.append(scatter)
                    # don't plot the "true deployment values", only the predicted ones with their corresponding error.
            else: 
                if scatter_all:
                    if errors == None:
                        ax[i,j].scatter(input_all_x[:,j], input_all_y[:,i], color = 'blue', alpha = 0.1, s = 4, label = 'dataset '+data_type)
                        if input_y['2'] is not None:
                            ax[i,j].scatter(input_x[numit,:,j], input_y['2'][numit_y,:,i], color = 'coral', alpha = 0.1, s=4, label = 'deployment NN_')
                        ax[i,j].scatter(input_x[numit,:,j], input_y['1'][numit_y,:,i], color = 'lightblue', alpha = 0.1, s=4, label = 'deployment NN')
                    else: 
                        ax[i,j].scatter(input_all_x[:,j], input_all_y[:,i], color = 'blue', 
                                        alpha = 0.1, s = 4, label = 'dataset '+data_type)
                        scatter = ax[i,j].scatter(input_x[numit,:,j], input_y['1'][numit_y,:,i], 
                                                c = errors_calc['rse'][:, i], cmap = 'plasma', norm = norms[i],
                                                alpha = 0.4, s=4, label = 'deployment NN')
                        if j == 0:
                            scatters.append(scatter)
                else:
                    if tag == 'eps-D':
                        raise Warning('use scatter_all = True when deploying with eps-D')
                    ax[i,j].hist(input_all[:,j], color = 'blue', alpha = 0.5, bins = 10)
                    ax[i,j].hist(input[numit,:,j], color = 'lightblue', alpha = 1, bins = 10)
            if i == input_y['1'].shape[2]-1:
                ax[i,j].set_xlabel(label_x[j])
            if j == 0:
                ax[i,j].set_ylabel(label_y[i])
    
    if errors != None:
        for i in range(input_y['1'].shape[2]):
            cbar = fig.colorbar(scatters[i],ax = ax[i,:], orientation = 'vertical', label = 'RSE')

    ax[0,1].legend()
    plt.suptitle(f'Trained model from v_{path[-3:]}')


    if save_path is not None: 
        if filter is not None and range_ is not None: 
            path_save_fig = os.path.join(save_path, 'scatter_matrix_inp_depl_'+tag+str(numit)+'_filter_range.png')
            if errors is not None: 
                path_save_fig = os.path.join(save_path, 'scatter_matrix_inp_depl_'+tag+str(numit)+'_filter_range_errors.png')
        elif filter is not None: 
            path_save_fig = os.path.join(save_path, 'scatter_matrix_inp_depl_'+tag+str(numit)+'_filter.png')
            if errors is not None: 
                path_save_fig = os.path.join(save_path, 'scatter_matrix_inp_depl_'+tag+str(numit)+'_filter_errors.png')
        elif range_ is not None: 
            path_save_fig = os.path.join(save_path, 'scatter_matrix_inp_depl_'+tag+str(numit)+'_range.png')
            if errors is not None: 
                path_save_fig = os.path.join(save_path, 'scatter_matrix_inp_depl_'+tag+str(numit)+'_range_errors.png')
        else:
            path_save_fig = os.path.join(save_path, 'scatter_matrix_inp_depl_'+tag+str(numit)+'.png')
            if errors is not None: 
                path_save_fig = os.path.join(save_path, 'scatter_matrix_inp_depl_'+tag+str(numit)+'_errors.png')
        plt.savefig(path_save_fig)
        print(f'Saved scatter matrix for deployment vs {data_type} data')


    return

def PCA_plot(path, input, numit = 0, save_path = None,
            data_type = 'test', type = 'eps-t', add_input = None):
    
    '''
    carries out pca for given input data, collapses dataset into two dimensions.
    path            (str)       to model number (and the entire dataset)
    input           (arr)       input data (can be input or output data, specified by "type")
                                expected shape: [numit, num_elem, num_cols(sig: 8, eps-t: 11, D: 64)]
    numit           (int)       current iteration number (start with 0)
    data_type       (str)       can be train or test to display the training or testing data in the background
    type            (str)       'eps-t', 'sig' or 'D' to specify the data on which the PCA should be carried out
    '''

    raise UserWarning('This has not yet been implemented for the THREEDIM case.')
    # TODO!

    
    with open(os.path.join(path, 'mat_data_np_TrainEvalTest.pkl'),'rb') as handle:
            mat_data_np = pickle.load(handle)
    with open(os.path.join(path, 'mat_data_stats.pkl'),'rb') as handle:
            mat_data_stats = pickle.load(handle)
    
    
    # 1- Collect the data from dataset and standardise it

    if type == 'eps-t':
        if data_type == 'test':
            input_all_raw = mat_data_np['X_test'][:,0:11]
        elif data_type == 'train':
            input_all_raw = mat_data_np['X_train'][:,0:11]
        input_all = transf_units(input_all_raw, 'eps-t', forward = False, linel=False)

        input_all_t = transform_data(input_all, mat_data_stats, forward = True, type = ['x-std']*11, sc=False)
        input_t = transform_data(input[numit, :,:], mat_data_stats, forward = True, type = ['x-std']*11, sc = False)

        if add_input is not None: 
            print('This is not yet implemented for eps-t. Will plot without consideration of add_inp')

    elif type == 'D':
        if data_type == 'test':
            input_all_raw = mat_data_np['y_test'][:,8:]
        elif data_type == 'train':
            input_all_raw = mat_data_np['y_train'][:,8:]
        input_all = input_all_raw.reshape((-1,64))
        input_ = transf_units(input[numit,:,:].reshape((-1,8,8)), 'D', forward = True, linel = False)
        input_re = input_.reshape((-1,64))

        input_all_t = transform_data(input_all, mat_data_stats, forward = True, type = ['y-st-stitched']*64, sc = False)
        input_t = transform_data(input_re, mat_data_stats, forward = True, type = ['y-st-stitched']*64, sc = False)

        if add_input is not None: 
            add_input_ = transf_units(add_input[numit,:,:].reshape((-1,8,8)), 'D', forward = True, linel = False)
            add_input_re = add_input_.reshape((-1,64))
            add_input_t = transform_data(add_input_re, mat_data_stats, forward = True, type = ['y-st-stitched']*64, sc = False)

    elif type == 'sig':
        raise RuntimeError('This has not yet been implemented for sigma')

    

    # 2 - Apply PCA
    pca = PCA(n_components = 2)
    input_all_t_pca = pca.fit_transform(input_all_t)
    input_t_pca = pca.transform(input_t)
    if add_input is not None: 
        add_input_t_pca = pca.transform(add_input_t)

    # 3 - plot the output of PCA
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(input_all_t_pca[:, 0], input_all_t_pca[:, 1], label=data_type+' data, N = '+str(input_all_t_pca.shape[0]), alpha=0.6)
    plt.scatter(input_t_pca[:, 0], input_t_pca[:, 1], label='deployment data NN, N = '+str(input_t_pca.shape[0]), alpha=0.6)
    if add_input is not None: 
        plt.scatter(add_input_t_pca[:,0], add_input_t_pca[:,1], label='deployment data NLFEA, N = '+str(input_t_pca.shape[0]), alpha=0.6)
    
    # Plot formatting
    plt.xlabel('Principal Component 1 \n [normalised units]')
    plt.ylabel('Principal Component 2 \n [normalised units]')
    plt.title('PCA Projection to 2D for '+type+', numit = '+str(numit))
    plt.legend()
    plt.tight_layout()
    
    if save_path is not None: 
        if add_input is not None: 
            path_save_fig = os.path.join(save_path, 'PCA_depl_comparison '+str(numit)+' '+type+ ' add_data')
        else:
            path_save_fig = os.path.join(save_path, 'PCA_depl_comparison '+str(numit)+' '+type)
        plt.savefig(path_save_fig)
        print('Saved PCA '+ type+' for deployment')

    return


def find_range_depl(input, numit, min_val = 1e-5):
    min_eps = np.min(input[numit,:,0:3])
    max_eps = np.max(input[numit,:,0:3])

    min_chi = np.min(input[numit,:,3:6])
    max_chi = np.max(input[numit,:,3:6])

    min_gam = np.min(input[numit,:,6:8])
    max_gam = np.max(input[numit,:,6:8])

    # check that max is not equal to min:
    if abs(min_eps-max_eps) < 1e-7:
        min_eps = -min_val
        max_eps = min_val
    if abs(min_gam-max_gam) < 1e-7:
        min_gam = -min_val
        max_gam = min_val
    if abs(min_chi-max_chi) < 1e-7:
        min_chi = -min_val
        max_chi = -min_val
    

    range_depl = np.zeros((2,8))
    range_depl[0,0:3] = np.repeat(min_eps, 3)
    range_depl[1,0:3] = np.repeat(max_eps, 3)
    range_depl[0,3:6] = np.repeat(min_chi,3)
    range_depl[1,3:6] = np.repeat(max_chi,3)
    range_depl[0,6:8] = np.repeat(min_gam, 2)
    range_depl[1,6:8] = np.repeat(max_gam, 2)
    
    return range_depl

def copy_files_with_prefix(source_folder, destination_folder, prefix):
    # Ensure destination folder exists
    plots_folder = os.path.join(destination_folder, "plots")
    os.makedirs(plots_folder, exist_ok=True)
    
    # Iterate through files in the source folder
    for filename in os.listdir(source_folder):
        if filename.startswith(prefix):
            src_path = os.path.join(source_folder, filename)
            dest_path = os.path.join(plots_folder, filename)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dest_path)
                print(f"Copied: {filename} → {plots_folder}")
    
    return




def strain_planes(inputs_norm_tot, inputs_NN_tot=None, numit_until = None, save_path = None, selected_point = None):
    '''
    Creates a strain plane plot to show how strains evolve over time (i.e. over iterations) during the deployment
    For the moment just creates one strain plane (the one in y-direction, can be adjusted later.)


    inputs_norm_tot     (np.array)      epsilon values used as input parameters, including geometrical values
                                        expected shape: (numit,num_elem*go, 11)
    inputs_NN_tot       (np.array)      epsilon values used as input parameters (the ones that are derived from predicted stiffness), including geometrical values
                                        expected shape: (numit,num_elem*go, 11)
    numit_until         (int)           number of iteration until which the strain planes shall be plotted
    save_path           (str)           path to save location, if None: not saved
    selected_point      (int)           to select a point other than the max chi_y value in 0th iteration
                                        (e.g. when investigating a different deployment scenario than pure bending)
    
    '''
    # select point of interest for strain plane
    if selected_point == None: 
        max_points = np.array([np.argmax(inputs_norm_tot[:,:,3], axis = 1),    # chi_x
                                np.argmax(inputs_norm_tot[:,:,4], axis = 1),   # chi_y
                                np.argmax(inputs_norm_tot[:,:,5], axis = 1)])  # chi_xy
        selected_point = max_points[:,6]                              # to choose the number of iteration acc. to which the max. chi is selected


    strain_labels = ['Strains in x-direction [1/mm]', 'Strains in y-direction [1/mm]', 'Strains in xy-direction [1/mm]']


    # extract the relevant parameters from input_norm_tot: eps and chi
    t = np.array([inputs_norm_tot[:,0,8], 
                  inputs_norm_tot[:,0,8], 
                  inputs_norm_tot[:,0,8]])
    eps_i = np.array([inputs_norm_tot[:,selected_point[0],0], 
                       inputs_norm_tot[:,selected_point[1],1], 
                       inputs_norm_tot[:,selected_point[2],2]])
    chi_i = np.array([inputs_norm_tot[:,selected_point[0],3], 
                      inputs_norm_tot[:,selected_point[1],4],
                      inputs_norm_tot[:,selected_point[2],5]])
    if inputs_NN_tot is not None:
        eps_i_NN = np.array([inputs_NN_tot[:,selected_point[0],0], 
                       inputs_NN_tot[:,selected_point[1],1], 
                       inputs_NN_tot[:,selected_point[2],2]])
        chi_i_NN = np.array([inputs_NN_tot[:,selected_point[0],3], 
                        inputs_NN_tot[:,selected_point[1],4],
                        inputs_NN_tot[:,selected_point[2],5]])
        
    if numit_until is not None:
        amt_it = numit_until
    else:
        amt_it = eps_i.shape[1]

    # Calculate x_star (compression height) and strains in top, bottom chord
    x_star = eps_i / np.tan(chi_i)
    if x_star.any() < 1e-8:
        x1 = (t/2) * np.tan(chi_i)
        x2 = (t/2) * np.tan(chi_i)
    else: 
        x1 = (eps_i/x_star) * (t/2 + x_star)
        x2 = (eps_i/x_star) * (t/2 - x_star)

    if inputs_NN_tot is not None:
        x_star_NN = eps_i_NN / np.tan(chi_i_NN)
        x1_NN, x2_NN, x3_NN = np.zeros_like(x_star_NN), np.zeros_like(x_star_NN), np.zeros_like(x_star_NN)
        for j in range(3): 
            for i in range(amt_it):
                if abs(x_star_NN[j,i]) < 1:
                    x_star_NN[j,i] = 0
                    x1_NN[j,i] = (t[j,i]/2) * np.tan(chi_i_NN[j,i])
                    x2_NN[j,i] = (t[j,i]/2) * np.tan(chi_i_NN[j,i])
                    x3_NN[j,i] = 0
                else:
                    x1_NN[j,i] = (eps_i_NN[j,i]/x_star_NN[j,i]) * (t[j,i]/2 + x_star_NN[j,i])
                    x2_NN[j,i] = (eps_i_NN[j,i]/x_star_NN[j,i]) * (t[j,i]/2 - x_star_NN[j,i])
                    x3_NN[j,i] = np.zeros_like(x1_NN[j,i])


    # Combine the calculated values to an array for plotting
    x_values = np.array([x1, -x2])
    y_values = np.array([np.zeros_like(t), t]) 
    if inputs_NN_tot is not None:
        pt_x_lower = np.array([x1_NN, np.zeros_like(x1_NN), x3_NN, x1_NN])
        pt_y_lower = np.array([np.zeros_like(x1_NN), np.zeros_like(x1_NN), t/2+x_star_NN, np.zeros_like(x1_NN)])
        pt_x_upper = np.array([-x2_NN, np.zeros_like(x1_NN), x3_NN, -x2_NN])
        pt_y_upper = np.array([t, t, t/2+x_star_NN, t])

    # plot the strain plane for all numits
    colors = plt.cm.get_cmap('viridis', eps_i.shape[1])(np.arange(eps_i.shape[1]))
    fig, ax = plt.subplots(1,3, figsize=(18,6))
    

    for j in range(3):
        for i in range(amt_it):
            ax[j].plot(x_values[:,j,i], y_values[:,j,i], color = colors[i])     # strain plane
            # ax[j].plot(np.tile(eps_i[j,:], (2,1))[:,i], y_values[:,j,i], color = colors[i], linestyle = '--') # epsilon at middle of CS
            ax[j].plot(eps_i[j,i], t[j,i]/2, color = colors[i], marker='o', markersize = '5')
            if inputs_NN_tot is not None:
                # draw triangles
                ax[j].fill(pt_x_lower[:,j,i], pt_y_lower[:,j,i], color = colors[i], alpha = 0.4)
                ax[j].fill(pt_x_upper[:,j,i], pt_y_upper[:,j,i], color = colors[i], alpha = 0.4)
                ax[j].plot(eps_i_NN[j,i], t[j,i]/2, color = colors[i], marker='x', markersize = '10')
        ax[j].set_ylim([y_values[0,j,0], y_values[1,j,0]])
        ax[j].set_xlabel(strain_labels[j])
    ax[0].set_ylabel('Height in cross-section [mm]')

    legend_handles = []
    for l in range(eps_i.shape[1]):
        pt = Line2D([0], [0], marker='o', color=colors[l], markerfacecolor=colors[l], markersize=3, alpha=1)
        legend_handles.append(pt)
    plt.legend(legend_handles, list(np.arange(eps_i.shape[1])), loc = 'upper right', title='Iteration Number')

    if save_path is not None: 
        if inputs_NN_tot is not None: 
            path_save_fig = os.path.join(save_path, 'strain_plane'+'_all_NN.png')
        else:
            path_save_fig = os.path.join(save_path, 'strain_plane'+'_all_NLFEA.png')
        plt.savefig(path_save_fig)
        print(f'Saved strain plane plot for all directions.')


    return

def cond_distr(x_filter, x_index, y_index, filter_range, data_path, save_path = None):
    '''
    Create a conditional distribution of two variables x_label and y_label.
    x_filter        (list)          list of columns that should be in filter_range in eps
    y_index         (int)           index of y to be plotted (e.g. 0 -> sig_x and D_m11)
    filter_range    (float)         allowed range for the other parameters (outside of the selected x_label and y_label)
    data_path       (str)           path to train, eval, test data
    
    '''

    with open(os.path.join(data_path, 'mat_data_np_TrainEvalTest.pkl'),'rb') as handle:
            mat_data_np = pickle.load(handle)

    eps_all = np.concatenate((mat_data_np['X_train'], mat_data_np['X_eval'], mat_data_np['X_test']), axis = 0)
    sig_all = np.concatenate((mat_data_np['y_train'][:,0:8], mat_data_np['y_eval'][:,0:8], mat_data_np['y_test'][:,0:8]), axis = 0)
    D_all = np.concatenate((mat_data_np['y_train'][:,8:], mat_data_np['y_eval'][:,8:], mat_data_np['y_test'][:,8:]), axis = 0).reshape((-1,8,8))
    

    mask = np.all(np.abs(eps_all[:,x_filter]) < filter_range, axis =1)

    print('Amount of datapoints before filtering:', eps_all.shape[0])
    eps_filtered = eps_all[mask]
    print('Amount of datapoints after filtering:', eps_filtered.shape[0])

    sig_filtered = sig_all[mask]
    D_filtered = D_all[mask]
    print(f'Amount of sig (D) points after filtering (should coincide with previous number) {sig_filtered.shape[0]} ({D_filtered.shape[0]})')

    fig, ax1 = plt.subplots(1,1, figsize = (6,6))
    ax1.scatter(eps_filtered[:,x_index], sig_filtered[:,y_index], s= 5, color = 'coral', alpha = 0.3, label = 'Stress')
    ax1.plot([-filter_range,-filter_range], [np.min(sig_filtered[:,y_index]), np.max(sig_filtered[:,y_index])], color = 'black', linestyle='--')
    ax1.plot([filter_range,filter_range], [np.min(sig_filtered[:,y_index]), np.max(sig_filtered[:,y_index])], color = 'black', linestyle='--')


    # ax1.plot([-filter_range,filter_range], [np.min(sig_filtered[:,y_index]), np.min(sig_filtered[:,y_index])], color = 'black')
    # ax1.plot([-filter_range,filter_range], [np.max(sig_filtered[:,y_index]), ], color = 'black')
    ax2 = ax1.twinx()
    ax2.scatter(eps_filtered[:,x_index], D_filtered[:,y_index, y_index], s = 5, color = 'blue', alpha = 0.3, label = 'Stiffness')
    ax1.set_xlabel(f'Generalised Strain eps_{x_index}')
    ax1.set_ylabel(f'Generalised Stress sig_{y_index}')
    ax2.set_ylabel(f'Stiffness D_{y_index}_{y_index}')
    ax1.legend()


    if save_path is not None:
        path_save_fig = os.path.join(save_path, 'cond_distr_'+str(x_index)+'_'+str(y_index))
        plt.savefig(path_save_fig)
        print(f'Saved conditional distribution plot for eps_{x_index}, sig_{y_index}')

    return


'''
from NN_call import *


def differentiability_vis(model_path, geom, epnum, index = 0, n = 100, save_path = None):
    model_path          (str)       path to NN model and associated datasets
    geom                (list)      geometrical features for which to filter
    epnum               (int)       epoch number of trained model
    index               (int)       index for which to sort. set to 0 for eps_x and Dm11
    n                   (int)       amount of values to make predictions. only every tenth value will be plotted for the training data.
    save_path           (str)       path where figure is saved       
    

    # extract data from training dataset
    with open(os.path.join(model_path, 'mat_data_np_TrainEvalTest.pkl'),'rb') as handle:
        mat_data_np = pickle.load(handle)

    all_data = np.concatenate((mat_data_np['X_train'], mat_data_np['y_train']), axis = 1)
    all_data_sorted = all_data[all_data[:,index].argsort()]
    mask = np.all(abs(all_data_sorted[:,8:11] - geom) < 1e-9, axis = 1)
    filtered_data = all_data_sorted[mask]
    print('Amount of filtered data available', filtered_data.shape) 

    # Plotting:
    # 1 - use the eps_x and all associated eps_generalised to predict D_m11 in the same range and plot them as a line.
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    # extract the X values:
    x_values = filtered_data[:n,0:11]
    model_path_trained_model = model_path+'\\best_trained_model__'+epnum+'.pt'
    mat_NN = predict_sig_D(x_values, model_path, model_path_trained_model, 'train', transf_type = 'st-stitched', predict = 'D', sc=False)
    y_values = mat_NN['D_pred']
    mat_NN_ = predict_sig_D(x_values, model_path, model_path_trained_model, 'train', transf_type = 'st-stitched', predict = 'sig', sc=False)
    y_values_ = mat_NN_['sig_h']

    ax.plot(x_values[:n:10,0], y_values[:n:10,0,0], '-', color = 'coral', label = 'NN: D')

    ax2 = ax.twinx()
    ax2.plot(x_values[:n:10,0], y_values_[:n:10,0], '--', color = 'coral', label = 'NN: sig')
    ax2.set_ylabel('Stress n_x [MN/cm]')

    # 2 - plot eps_x and D_m11 from the filtered data as scatter (crosses), every 50th element in the first n values.
    ax.scatter(filtered_data[:n:50, index], filtered_data[:n:50,index+11+8], marker = 'x', color = 'blue', label = 'NLFEA')


    ax.set_xlabel('Strains eps_x [-]')
    ax.set_ylabel('Stiffness Dm11 [MN/cm]')
    
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')



    if save_path is not None: 
        path_save_fig = os.path.join(save_path, 'Differentiability_plot.png')
        plt.savefig(path_save_fig)
        print(f'Saved Diff\'bility plot.')


    return'''




################################## Storing intermediate solutions ################################## 

def store_intermediate_solutions(numit, NN_hybrid, sh, sh_, eh, eh_, u, u_, De_tot, De_tot_):
    """
    Creates arrays for intermediate solution saving and saves first solution to 0th row in array.
    
    Args: 
        numit       (int)           current iteration number for given load level
        NN_hybrid   (bool-dict)     boolean indicating which values are predicted by NN
        sh, sh_     (np.arr)        stress arrays (standard, and non-NN calculation)
        eh, eh_     (np.arr)        strain arrays (standard, and non-NN calculation)
        u, u_       (np.arr)        displacement arrays (standard and non-NN calculation)
        De_tot, De_tot_ (np.arr)    stiffness matrices (standard and non-NN calculation)
        

    Returns: 
        aggregated arrays of each of the above listed resulting values.

    """

    sh_cum = np.zeros((numit+1, *sh.shape))
    eh_cum = np.zeros((numit+1, *eh.shape))
    u_cum = np.zeros((numit+1, *u.shape))
    fi_cum = np.zeros((numit+1, *u.shape))
    De_cum = np.zeros((numit+1, *De_tot.shape))
    if NN_hybrid['predict_sig'] and not NN_hybrid['predict_D']: 
        sh_cum_ = np.zeros((numit+1, *sh_.shape))
        eh_cum_ = np.zeros((numit+1, *eh_.shape))
        fi_cum_ = np.zeros((numit+1, *u.shape))
        De_cum_ = None
        u_cum_ = None
    elif not NN_hybrid['predict_sig'] and not NN_hybrid['predict_D']:
        eh_cum_ = None
        sh_cum_ = None
        u_cum_ = None
        fi_cum_ = None
        De_cum_ = None
    elif not NN_hybrid['predict_sig'] and NN_hybrid['predict_D']:
        u_cum_ = np.zeros((numit+1, *u.shape))
        fi_cum_ = np.zeros((numit+1, *u.shape))
        De_cum_ = np.zeros((numit+1, *De_tot_.shape))
        eh_cum_ = np.zeros((numit+1, *eh.shape))
        sh_cum_ = None
    elif NN_hybrid['predict_sig'] and NN_hybrid['predict_D']:
        sh_cum_ = np.zeros((numit+1, *sh_.shape))
        eh_cum_ = np.zeros((numit+1, *eh_.shape))
        u_cum_ = np.zeros((numit+1, *u.shape))
        fi_cum_ = np.zeros((numit+1, *u.shape))
        De_cum_ = np.zeros((numit+1, *De_tot_.shape))

    sh_cum[0,:,:,:,:] = sh
    eh_cum[0,:,:,:,:] = eh
    u_cum[0,:,:,] = u
    De_cum[0,:,:,:,:,:] = De_tot
    if NN_hybrid['predict_sig'] and not NN_hybrid['predict_D']: 
        sh_cum_[0,:,:,:,:] = sh_
        eh_cum_[0,:,:,:,:] = eh_
    if not NN_hybrid['predict_sig'] and NN_hybrid['predict_D']:
        u_cum_[0,:,:,] = u_
        De_cum_[0,:,:,:,:,:] = De_tot_
        eh_cum_[0,:,:,:,:] = eh_
    if NN_hybrid['predict_sig'] and NN_hybrid['predict_D']:
        sh_cum_[0,:,:,:,:] = sh_
        eh_cum_[0,:,:,:,:] = eh_
        u_cum_[0,:,:,] = u_
        De_cum_[0,:,:,:,:,:] = De_tot_

    return sh_cum, sh_cum_, eh_cum, eh_cum_, u_cum, u_cum_, De_cum, De_cum_, fi_cum, fi_cum_



