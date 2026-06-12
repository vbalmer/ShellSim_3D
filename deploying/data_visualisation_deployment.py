# Plotting some test plots for dataset
import pickle
import os
import numpy as np
import pandas as pd

from main_utils_vb import scatter_vb, PCA_plot, all_eps_sig_plots, imshow_plots, all_De_plots, find_range_depl, check_plots_perit, copy_files_with_prefix, strain_planes, cond_distr
from Plot_NoDash import main_plt_comp, main_plt, contour_perit
from load_paths_plots import *

##################################################
# Definition of input parameters
##################################################

PREDICT_SIG = True
PREDICT_D = True
PERM = False
PERM1 = False

INP_PLOTS = False
D_PLOTS = False
SIG_PLOTS = False
U_PLOTS = False
LOAD_PATH = True


PCA_PLOTS = False
DIAG_PLOTS = False
SCATTER_PLOTS = False
SCATTER_PLOTS_2D = False
STRAIN_PLANE = False
CONTOUR_PLOTS = False
CONTOUR_PERIT = False
ALLIT_PLOTS = False
COND_PLOT = False


NUMIT = 0
COPY = True
# GEOM = np.array([250, 0.015, 2])

path = os.path.join(os.getcwd(), 'training\\logs\\v_37')        #have not tested whether this path works as intended yet.
path_depl = 'deploying\\data_out\\data_20260612_1731_casexx\\160000'
save_path = 'deploying\\plots\\deployment_visualisation'


##################################################
# Reading in the data
##################################################
if (not LOAD_PATH):
        if PREDICT_D and PREDICT_SIG:
                with open(os.path.join(path_depl, 'mat_res_NN.pkl'),'rb') as handle:
                        mat_res_NN = pickle.load(handle) 
        elif PREDICT_D: 
                with open(os.path.join(path_depl, 'mat_res_NN_D.pkl'),'rb') as handle:
                        mat_res_NN = pickle.load(handle)
        elif PREDICT_SIG: 
                with open(os.path.join(path_depl, 'mat_res_NN_sig.pkl'),'rb') as handle:
                        mat_res_NN = pickle.load(handle)
        if PERM: 
                with open(os.path.join(path_depl, 'mat_res_norm_perm.pkl'), 'rb') as handle:
                        mat_res_perm = pickle.load(handle)
        if PERM1: 
                with open(os.path.join(path_depl, 'mat_res_norm_perm1.pkl'), 'rb') as handle:
                        mat_res_perm = pickle.load(handle)
        with open(os.path.join(path_depl, 'mat_res_norm.pkl'), 'rb') as handle:
                mat_res_norm = pickle.load(handle)

        if 'rho_x' not in mat_res_norm:
                GEOM = np.array([mat_res_norm['t_1'][0], mat_res_norm['rho'][0],mat_res_norm['CC'][0]])
        else:
                if isinstance(mat_res_norm['t_1'], np.int32):
                        GEOM = np.array([mat_res_norm['t_1'], mat_res_norm['rho_x'], mat_res_norm['rho_y'],mat_res_norm['CC'][0]])
                else:
                        GEOM = np.array([mat_res_norm['t_1'][0], mat_res_norm['rho_x'][0], mat_res_norm['rho_y'][0],mat_res_norm['CC'][0]])


        # print steel stresses

        if 'ssx' in mat_res_norm:
                ssx = mat_res_norm['ssx'][0]
                ssy = mat_res_norm['ssy'][0]
                print('ssx_max: ' , np.max(ssx), 'ssy_max: ', np.max(ssy))
                print('ssx_min: ' , np.min(ssx), 'ssy_min: ', np.min(ssy))

################################################
# Plotting
################################################

############## strains: eps-t #################

if INP_PLOTS:
        inputs = mat_res_NN['eh_cum'][0]
        inputs_norm = mat_res_norm['eh_cum'][0]
        inputs_re = inputs.reshape((inputs.shape[0], -1, 8))
        inputs_norm_re = inputs_norm.reshape((inputs_norm.shape[0],-1,8))

        geom = GEOM
        geom_re = np.tile(geom, (inputs_re.shape[0], inputs_re.shape[1], 1))

        inputs_tot = np.concatenate((inputs_re, geom_re), axis = 2)
        inputs_norm_tot = np.concatenate((inputs_norm_re, geom_re), axis = 2)

        if SCATTER_PLOTS:
                # 1 - basic plot
                scatter_vb(path, input = inputs_tot, numit = NUMIT, save_path=save_path, 
                        scatter_all=True, filter = None, range_ = None)

                # 2a - filtered plot
                filter_test_data = GEOM
                scatter_vb(path, input = inputs_tot, numit = NUMIT, save_path=save_path, 
                        scatter_all=True, data_type = 'train', filter = filter_test_data, range_ = None)

                # 2b - zoomed-in plot
                min = np.array([[-0.0005]*3+[-1e-5]*3+[-0.2e-3]*2])
                max = np.array([[0.0005]*3+[1e-5]*3+[0.2e-3]*2])
                range_train_data = np.concatenate((min, max), axis = 1).reshape((2,8))
                range_depl = find_range_depl(inputs_tot, numit = NUMIT)
                print('Deployment range', range_depl)

                scatter_vb(path, input = inputs_tot, numit = NUMIT, save_path = save_path, 
                        scatter_all=True, data_type='train', filter = None, range_ = range_depl)

                scatter_vb(path, input = inputs_tot, numit = NUMIT, save_path = save_path, 
                        scatter_all=True, data_type='train', filter = filter_test_data, range_ = range_depl)
                
                # 2c - copy figures to deployment folder
                if COPY: 
                        copy_files_with_prefix(save_path, path_depl, 'scatter_matrix_inp_depl_eps-eps'+str(NUMIT))
                
        

        # 3 - PCA plot for inputs
        if PCA_PLOTS:
                for i in range(7):
                        PCA_plot(path, input = inputs_tot, numit = i, save_path=save_path,
                                data_type = 'test', type = 'eps-t')
                        

        # 4 - Strain plane cross-section
        if STRAIN_PLANE: 
                strain_planes(inputs_norm_tot, inputs_tot, numit_until=7, save_path=save_path)
                strain_planes(inputs_norm_tot, numit_until=7, save_path=save_path)

                if COPY: 
                        copy_files_with_prefix(save_path, path_depl, 'strain_plane_all')



        if CONTOUR_PERIT:
                idx = [2] 
                contour_perit(mat_res_norm, mat_res_NN, mat_res_perm = None,
                              numit = 9, idx = idx, save_path=save_path, same = False, tag = 'eps')
                if COPY: 
                        copy_files_with_prefix(save_path, path_depl, 'contour_perit_eps'+str(idx))


############## stiffness #################

if D_PLOTS:
        # 0 - rearranging data
        # input_D = mat_res_NN['De_cum'][0]
        # input_D_ = mat_res_NN['De_cum_'][0]
        # input_D_norm = mat_res_norm['De_cum'][0]
        # input_eh = mat_res_NN['eh_cum'][0]
        input_D = mat_res_NN['De_cum']
        input_D_ = mat_res_NN['De_cum_']
        input_D_norm = mat_res_norm['De_cum']
        input_eh = mat_res_NN['eh_cum']
        if PERM or PERM1: 
                input_D_perm = mat_res_perm['De_cum'][0]
                eh_cum_perm = mat_res_perm['eh_cum'][0]

        if input_D_ is not None:
                input_D_re = input_D.reshape((input_D.shape[0],-1,8,8))
                input_D_re_ = input_D_re.reshape((input_D.shape[0], -1,64))
                input_D__re = input_D_.reshape((input_D_.shape[0],-1,8,8))
                input_D__re_ = input_D__re.reshape((input_D_.shape[0], -1,64))
                input_D_norm_re = input_D_norm.reshape((input_D_norm.shape[0], -1, 8, 8))
                input_D_norm_re_ = input_D_norm_re.reshape((input_D_norm.shape[0], -1, 64)) 


        # 1 - PCA plot for outputs of stiffness
        if PCA_PLOTS:
                PCA_plot(path, input = input_D_re_, numit = NUMIT, save_path = save_path, 
                        data_type = 'test', type = 'D')

                for i in range(7):
                        PCA_plot(path, input = input_D_re_, numit = i, save_path = save_path, 
                                data_type = 'test', type = 'D', add_input = input_D_norm_re_)
                

        # 2 - diagonal plots for stiffness
        if DIAG_PLOTS:
                model_path = {
                        'data':{'_I': path},
                        'model':{'_I': path}
                }

                # Diagaonal plot for NN-NLFEA hybrid
                check_plots_perit(model_path, numit = NUMIT, sh_true = None, sh_pred = None, eh = input_eh[NUMIT,:,:,:,:].reshape((-1,8)), 
                                D_true = input_D_[NUMIT+1,:,:,:,:,:].reshape((-1,8,8)), D_pred = input_D[NUMIT+1,:,:,:,:,:].reshape((-1,8,8)), 
                                sig = False, onlydiag = True)
                
                if PERM or PERM1: 
                        check_plots_perit(model_path, numit = NUMIT, sh_true = None, sh_pred = None, eh = input_eh[NUMIT,:,:,:,:].reshape((-1,8)), 
                                D_true = input_D_norm[NUMIT+1,:,:,:,:,:].reshape((-1,8,8)), D_pred = input_D_perm[NUMIT+1,:,:,:,:,:].reshape((-1,8,8)), 
                                sig = False, onlydiag = True)
                        print("Only Permuted Diagonal Plots will be copied to new folder. Please rename them such that they aren't overwritten.")



                # # Diagonal plot just for NLFEA (ground truth)
                # check_plots_perit(model_path, numit = NUMIT, sh_true = None, sh_pred = None, eh = input_eh[NUMIT,:,:,:,:].reshape((-1,8)),
                #                   D_true = input_D_norm[NUMIT+1,:,:,:,:,:].reshape((-1,8,8)), D_pred = input_D_norm[NUMIT+1,:,:,:,:,:].reshape((-1,8,8)), 
                #                   sig = False)
                # for i in range(10):
                #         check_plots_perit(model_path, numit = i, sh_true = None, sh_pred = None, eh = input_eh[i,:,:,:,:].reshape((-1,8)), 
                #                         D_true = input_D_[i+1,:,:,:,:,:].reshape((-1,8,8)), D_pred = input_D[i+1,:,:,:,:,:].reshape((-1,8,8)), 
                #                         sig = False, onlydiag = True)


                if COPY:
                        source_path_0 = os.path.join(os.getcwd(), '05_Deploying\\plots\\diagonal_plots_D')
                        copy_files_with_prefix(source_path_0, path_depl, 'diagonal_match_'+str(NUMIT)+'D_nonzero_u')
                        # for i in range(10):
                        #         copy_files_with_prefix(source_path_0, path_depl, 'diagonal_match_'+str(i)+'D_nonzero_u')
                        # source_path_1 = os.path.join(os.getcwd(), '05_Deploying\\plots\\diagonal_plots_D_true')
                        # copy_files_with_prefix(source_path_1, path_depl, 'diagonal_match_'+str(NUMIT)+'D_nonzero_u')


        # 3 - convergence plots stiffness
        if ALLIT_PLOTS: 
                eh_cum_NN = mat_res_NN['eh_cum'][0]
                eh_cum_NN_ = mat_res_NN['eh_cum_'][0]
                eh_cum_NLFEA = mat_res_norm['eh_cum'][0]
                
                De_cum_NN_ = mat_res_NN['De_cum_'][0]
                if De_cum_NN_ is None:
                        De_cum_NN = None
                else: 
                        De_cum_NN = mat_res_NN['De_cum'][0]
                De_cum_NLFEA = mat_res_norm['De_cum'][0]
        
                # just plotting for the last iteration (numit = 6)
                all_De_plots(10, 10, eh_cum_NN, eh_cum_NN_, eh_cum_NLFEA, 
                            De_cum_NN, De_cum_NN_, De_cum_NLFEA, tag = 'max')
                all_De_plots(4, 10, eh_cum_NN, eh_cum_NN_, eh_cum_NLFEA, 
                            De_cum_NN, De_cum_NN_, De_cum_NLFEA, tag = 'max')
                
                if PERM or PERM1:
                        all_De_plots(10, 10, eh_cum_perm, eh_cum_NN_, eh_cum_NLFEA, 
                            De_cum_NN, De_cum_NN_, De_cum_NLFEA, tag = 'max') 

                
                if COPY: 
                        source_path_allit = os.path.join(os.getcwd(), '05_Deploying\\plots\\De_plots_it')
                        copy_files_with_prefix(source_path_allit, path_depl, 'De_10')
                        copy_files_with_prefix(source_path_allit, path_depl, 'De_4')
                        
        
        # 4a - scatter_vb (mike plot with zoom-in and filter function)
        if SCATTER_PLOTS:
                inputs = mat_res_NN['eh_cum'][0]
                inputs_re = inputs.reshape((inputs.shape[0], -1, 8))
                geom = GEOM
                geom_re = np.tile(geom, (inputs_re.shape[0], inputs_re.shape[1], 1))
                inputs_tot = np.concatenate((inputs_re, geom_re), axis = 2)

                input_dict = {
                        '1': np.concatenate((input_D_re[:,:,:6,:6].reshape((input_D_re.shape[0],-1,36)), input_D_re[:,:,6,6].reshape((input_D_re.shape[0],-1,1)), 
                                        input_D_re[:,:,7,7].reshape((input_D_re.shape[0],-1,1))), axis = 2),
                        '2': np.concatenate((input_D__re[:,:,:6,:6].reshape((input_D__re.shape[0],-1,36)), input_D__re[:,:,6,6].reshape((input_D__re.shape[0],-1,1)), 
                                        input_D__re[:,:,7,7].reshape((input_D__re.shape[0],-1,1))), axis = 2),
                }

                scatter_vb(path, input = inputs_tot, input_ = input_dict, numit = NUMIT, save_path = save_path, tag='eps-D',
                scatter_all=True, data_type='train', filter = None, range_ = None)


                # 4b - filtered scatter plot
                filter_test_data = GEOM
                scatter_vb(path, input = inputs_tot, input_ = input_dict, numit = NUMIT, save_path = save_path, tag='eps-D',
                        scatter_all=True, data_type='train', filter = filter_test_data, range_ = None)

                # 4c - zoomed-in scatter plot
                min = np.array([[-0.0005]*3+[-1e-5]*3+[-0.2e-3]*2])
                max = np.array([[0.0005]*3+[1e-5]*3+[0.2e-3]*2])
                range_train_data = np.concatenate((min, max), axis = 1).reshape((2,8))
                range_depl = find_range_depl(inputs_tot, numit = NUMIT)
                print('Deployment range', range_depl)
                scatter_vb(path, input = inputs_tot, input_ = input_dict, numit = NUMIT, save_path = save_path, tag='eps-D',
                        scatter_all=True, data_type='train', filter = None, range_ = range_depl)
                scatter_vb(path, input = inputs_tot, input_ = input_dict, numit = NUMIT, save_path = save_path, tag='eps-D',
                        scatter_all=True, data_type='train', filter = filter_test_data, range_ = range_depl)
                
                
                
                scatter_vb(path, input = inputs_tot, input_ = input_dict, numit = NUMIT, save_path = save_path, tag='eps-D',
                        scatter_all=True, data_type='train', filter = GEOM, range_ = None, errors = True)

                if COPY: 
                        copy_files_with_prefix(save_path, path_depl, 'scatter_matrix_inp_depl_eps-D'+str(NUMIT))



        if COND_PLOT:
                cond_distr([1,2,3,4,5,6,7], 0, 0, 5e-5, path, save_path)


        if CONTOUR_PERIT:
                idx = [2, 2]
                contour_perit(mat_res_norm, mat_res_NN, numit = 9, idx = idx, same = True, save_path=save_path, tag = 'D')
                # for just plotting NLFEA solution:
                # contour_perit(mat_res_norm, mat_res_NN = None, same = False, numit = 7, tag = 'D', idx = idx, save_path=save_path)

                if COPY: 
                        copy_files_with_prefix(save_path, path_depl, 'contour_perit_D'+str(idx))

############### stresses ######################

if SIG_PLOTS:
        # eh_cum_NN = mat_res_NN['eh_cum'][0]
        # eh_cum_NN_ = mat_res_NN['eh_cum_'][0]
        # eh_cum_NLFEA = mat_res_norm['eh_cum'][0]
        # sh_cum_NN = mat_res_NN['sh_cum'][0]
        # sh_cum_NN_ = mat_res_NN['sh_cum_'][0]
        # sh_cum_NLFEA = mat_res_norm['sh_cum'][0]
        eh_cum_NN = mat_res_NN['eh_cum']
        eh_cum_NN_ = mat_res_NN['eh_cum_']
        eh_cum_NLFEA = mat_res_norm['eh_cum']
        sh_cum_NN = mat_res_NN['sh_cum']
        sh_cum_NN_ = mat_res_NN['sh_cum_']
        sh_cum_NLFEA = mat_res_norm['sh_cum']

        inputs = mat_res_NN['eh_cum'][0]
        inputs_re = inputs.reshape((inputs.shape[0], -1, 8))
        geom = GEOM
        geom_re = np.tile(geom, (inputs_re.shape[0], inputs_re.shape[1], 1))
        inputs_tot = np.concatenate((inputs_re, geom_re), axis = 2)

        input_dict = {
                '1': sh_cum_NN.reshape((sh_cum_NN.shape[0],-1,8)),
                # '2': None,
                '2': sh_cum_NLFEA.reshape((sh_cum_NN.shape[0],-1,8)),
        }

        if SCATTER_PLOTS:
                scatter_vb(path, input = inputs_tot, input_ = input_dict, numit = NUMIT, save_path = save_path, tag='eps-sig',
                        scatter_all=True, data_type='train', filter = None, range_ = None)
                
                filter_test_data = GEOM
                scatter_vb(path, input = inputs_tot, input_ = input_dict, numit = NUMIT, save_path = save_path, tag='eps-sig',
                        scatter_all=True, data_type='train', filter = filter_test_data, range_ = None)
                
                range_depl = find_range_depl(inputs_tot, numit = NUMIT)
                scatter_vb(path, input = inputs_tot, input_ = input_dict, numit = NUMIT, save_path = save_path, tag='eps-sig',
                        scatter_all=True, data_type='train', filter = None, range_ = range_depl*3)

                scatter_vb(path, input = inputs_tot, input_ = input_dict, numit = NUMIT, save_path = save_path, tag='eps-sig',
                        scatter_all=True, data_type='train', filter = filter_test_data, range_ = range_depl*3)

                if COPY: 
                        source_path_scatter = os.path.join(os.getcwd(), 'deploying\\plots\\visualisation_deployment')
                        copy_files_with_prefix(source_path_scatter, path_depl, 'scatter_matrix_inp_depl_eps-sig'+str(NUMIT))

        if CONTOUR_PLOTS:
                with open(os.path.join(path, 'stats.pkl'),'rb') as handle:
                        mat_data_stats = pickle.load(handle)	

                # mat_res_raw0 = pd.DataFrame.from_dict(mat_res_norm)
                # mat_res_raw0_NN = pd.DataFrame.from_dict(mat_res_NN)
                # mat_res_raw0_ = mat_res_raw0.loc[0,:]
                # mat_res_raw0_NN_ = mat_res_raw0_NN.loc[0,:]
                mat_res_raw0_ = mat_res_norm
                mat_res_raw0_NN_ = mat_res_NN

                # comp_list_p = ['Nx', 'Ny', 'Nxy', 'Mx', 'My', 'Mxy', 'Qx', 'Qy']
                # main_plt_comp(mat_res_raw0_, mat_res_raw0_NN_, 'sig', save_path, type_err = 'nrse', 
                #               stats = mat_data_stats, comp_list = comp_list_p, paper=True)

                comp_list_sig = ['My']                        # comp_list_sig = ['Ny', 'My',  'Qy']
                comp_list_eps = ['chiy']                      # comp_list_eps = ['epsy',  'chiy', 'gamy']
                comp_list_u = ['ux', 'uy', 'uz']
                main_plt_comp(mat_res_raw0_, mat_res_raw0_NN_, 'sig', save_path, type_err = 'rrse', 
                              stats = mat_data_stats, comp_list = comp_list_sig)               
                main_plt_comp(mat_res_raw0_, mat_res_raw0_NN_, 'eps', save_path, type_err = 'rrse', 
                        stats = mat_data_stats, comp_list = comp_list_eps)    
                main_plt_comp(mat_res_raw0_, mat_res_raw0_NN_, 'u', save_path, type_err = 'rel', 
                        stats = mat_data_stats, comp_list = comp_list_u)               
                main_plt(mat_res_raw0_, 'sig', save_path)
                main_plt(mat_res_raw0_NN_, 'sig', save_path, nn = True)
                main_plt(mat_res_raw0_, 'eps', save_path)
                main_plt(mat_res_raw0_NN_, 'eps', save_path, nn = True)
                main_plt(mat_res_raw0_, 'u', save_path)
                main_plt(mat_res_raw0_NN_, 'u', save_path, nn = True)

                
                if COPY: 
                        copy_files_with_prefix(save_path, path_depl, 'comparison_'+ str(comp_list_sig) +'.png')
                        copy_files_with_prefix(save_path, path_depl, 'comparison_'+ str(comp_list_eps) +'.png')
                        copy_files_with_prefix(save_path, path_depl, 'comparison_'+ str(comp_list_u) +'.png')
                        copy_files_with_prefix(save_path, path_depl, 'single_sig')
                        copy_files_with_prefix(save_path, path_depl, 'single_eps')
                        copy_files_with_prefix(save_path, path_depl, 'single_u')

        if DIAG_PLOTS:
                model_path = {
                        'data':{'_I': path},
                        'model':{'_I': path}
                }

                # Diagaonal plot for NN-NLFEA hybrid for sigma
                sh_true_0 = sh_cum_NN_[NUMIT,:,:,:,:].reshape((-1,8))
                sh_pred_0 = sh_cum_NN[NUMIT,:,:,:,:].reshape((-1,8))
                # norms_0 = check_plots_perit(model_path, -1, sh_cum_NN_[0,:,:,:,:].reshape((-1,8)),
                #                             sh_cum_NN[0,:,:,:,:].reshape((-1,8)), eh_cum_NN[0,:,:,:,:].reshape((-1,8)),
                #                             sh_true_0, sh_pred_0, sig = True)
                check_plots_perit(model_path, numit = NUMIT, sh_true = sh_cum_NN_[NUMIT,:,:,:,:].reshape((-1,8)), 
                                  sh_pred = sh_cum_NN[NUMIT,:,:,:,:].reshape((-1,8)), eh = eh_cum_NN[NUMIT,:,:,:,:].reshape((-1,8)), 
                                  sh_true_0 = sh_true_0, sh_pred_0 = sh_pred_0, norms_0 = None, D_true = None, D_pred = None, 
                                  sig = True, onlydiag = True)

                if COPY:
                        source_path_0 = os.path.join(os.getcwd(), '05_Deploying\\plots\\diagonal_plots')
                        copy_files_with_prefix(source_path_0, path_depl, 'diagonal_match_'+str(NUMIT)+'original_units_newlim.png')
                        # for i in range(7):
                        #         copy_files_with_prefix(source_path_0, path_depl, 'diagonal_match_'+str(i)+'D_nonzero_u')
                        # source_path_1 = os.path.join(os.getcwd(), '05_Deploying\\plots\\diagonal_plots_D_true')
                        # copy_files_with_prefix(source_path_1, path_depl, 'diagonal_match_'+str(NUMIT)+'D_nonzero_u')


        if ALLIT_PLOTS:
                number = NUMIT
                all_eps_sig_plots(number, 6, eh_cum = eh_cum_NN, u_cum = None, sh_cum = sh_cum_NN, eh_cum_ = eh_cum_NN_, sh_cum_ = sh_cum_NN_,
                                  eh_cum_NLFEA=eh_cum_NLFEA, sh_cum_NLFEA=sh_cum_NLFEA, NN = 'NLFEA', tag = 'max')
                all_eps_sig_plots(number, 6, eh_cum = eh_cum_NN, u_cum = None, sh_cum = sh_cum_NN, eh_cum_ = eh_cum_NN_, sh_cum_ = sh_cum_NN_, 
                                  eh_cum_NLFEA = None, sh_cum_NLFEA = None, NN = 'NN', tag = 'max')
        
                if COPY: 
                        source_path_allit = os.path.join(os.getcwd(), '05_Deploying\\plots\\eps_sig_plots_it')
                        copy_files_with_prefix(source_path_allit, path_depl, 'eps_sig_'+str(number))
        
        
        if CONTOUR_PERIT:
                idx = [2] 
                contour_perit(mat_res_norm, mat_res_NN, mat_res_perm = None, 
                              numit = 9, idx = idx, save_path=save_path, same = False, tag = 'sig')
                if COPY: 
                        copy_files_with_prefix(save_path, path_depl, 'contour_perit_sig'+str(idx))

################ displacements ##################

if U_PLOTS:
        if CONTOUR_PERIT:
                idx = [1]
                contour_perit(mat_res_norm, mat_res_NN, mat_res_perm = None, 
                              numit = 9, idx = idx, save_path = save_path, 
                              diff = False, same = False, tag = 'u')

                if COPY: 
                        copy_files_with_prefix(save_path, path_depl, 'contour_perit_u'+str(idx)) 

################ load-paths ##################

if LOAD_PATH: 

        # path_depl = {
        #        '2D-1': {
        #                 '$\\rho_y$ = 1\%': 'data_20260119_0803_casexx',
        #                 '$\\rho_y$ = 0.75\%': 'data_20260120_1817_casexx',
        #                 '$\\rho_y$ = 1.5\%': 'data_20260120_1637_casexx',
        #         },
        #        '2D-2': {
        #                 '$\\rho_y$ = 1\%': 'data_20260120_0836_casexx',
        #                 '$\\rho_y$ = 0.75\%': 'data_20260121_1038_casexx',
        #                 '$\\rho_y$ = 1.5\%': 'data_20260120_1945_casexx',
        #           },
        #         '2D-3': {
        #                 '$\\rho_y$ = 1\%': 'data_20260120_1008_casexx',
        #                 '$\\rho_y$ = 0.75\%': 'data_20260121_1422_casexx',
        #                 '$\\rho_y$ = 1.5\%': 'data_20260120_1447_casexx',
        #         },
        #         '2D-4': {
        #                 '$\\rho_y$ = 1\%': 'data_20260120_1037_casexx',
        #                 '$\\rho_y$ = 0.75\%': 'data_20260121_1251_casexx',
        #                 '$\\rho_y$ = 1.5\%': 'data_20260121_1131_casexx',
        #         },
        #         '2D-5': {
        #                 '$\\rho_y$ = 1\%': 'data_20260120_1210_casexx',
        #                 '$\\rho_y$ = 0.75\%': 'data_20260121_1356_casexx',
        #                 '$\\rho_y$ = 1.5\%': 'data_20260121_1549_casexx',
        #         },
        # }


        # plot_load_path_wrapper(path_depl, case_study = '2D-1C', until_load_level = [1,18], 
        #                        save_path = save_path, type = 'eps', thresh = 5)
        
        plot_load_path_wrapper('deploying\\data_out\\data_20260612_1631_casexx', case_study = '2D-4', until_load_level = [0,8], 
                               save_path = save_path, type = 'u', thresh = 10)

        
        # plot_load_path_wrapper('05_Deploying\\data_out\\data_20260116_1510_casexx', case_study = '2D-3', until_load_level = [0,9], 
        #                        save_path = save_path, type = 'u', thresh = 10)

        # diagonal_loadpath_plot(path_depl, save_path = save_path, type = 'eps', thresh = 5)