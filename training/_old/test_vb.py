import pickle
import numpy as np
import torch
import os
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from lightning.pytorch import seed_everything
seed_everything(42)

from data_work import *
from FFNN_class_light import *
from call_light import *
from test_utils import *

#########################################################################################################
## 1 - Testing (45°-plots for sigma)
#########################################################################################################

# load hyperparams, data
path = os.path.join(os.getcwd(), '04_Training')
path_plots = os.path.join(path, 'plots')
# add_path = '_simple_logs\\v_339'            # for lin.el. RC material
# add_path = '_simple_logs\\v_243'            # for glass
add_path = '_simple_logs\\v_490'            # for RC, D-matrix
# add_path = '_simple_logs\\v_221'            # for RC, only sig
# add_path = '_simple_logs\\v_320'              # for RC, sig + D (!)
# add_path = '_simple_logs\\v_279'              # for RC, MoE
# add_path = '_simple_logs\\v_264'              # for inverse
# add_path = '_simple_logs\\v_304'
data_model = load_data(path, only_test = True, add_path = add_path)
LINEL = False                                                           # only set false for nonlin RC model, else true
PURE_D = False                                                          # for only predicting D, not sigma
INV = False                                                             # to predict epsilon from sigma
SCALE = False
DOUBLE_NORM = False
TWODIM = True
ONEDIM = False
GEOM_SIZE = 4
onedim_nonzero = 0
REDUCE_DATASIZE = False

# load model from checkpoint:
inp = data_model['inp'] 
print(inp)
model_test_dict = test_model_instance(inp, path, v_num='490', epoch='_247')         # for lin.el. RC material
# model_test_dict = test_model_instance(inp, path, v_num='243', epoch='_19996')         # for glass
# model_test_dict = test_model_instance(inp, path, v_num='221', epoch='_19929')         # for nonlin RC, sig
# model_test_dict = test_model_instance(inp, path, v_num='220', epoch='_19882')         # for nonlin RC, D-matrix
# model_test_dict = test_model_instance(inp, path, v_num='320', epoch='_29929')         # for nonlin RC, sig+D
# model_test_dict = test_model_instance(inp, path, v_num='279', epoch=['MoE_131', 'exp1_29991', 'exp2_29972', 'exp3_29993'])         # for nonlin RC, MoE for sig
# model_test_dict = test_model_instance(inp, path, v_num='264', epoch='_39992')         # for inverse
# model_test_dict = test_model_instance(inp, path, v_num='304', epoch='_9992') 

# in "epoch" also add identifier: _9273 or _exp1_341 etc.; for MoE: use a list ['MoE_578', 'exp1_1217', 'exp2_4899', 'exp3_4835']

# set different type of transformations depending on sobolev loss
if LINEL or PURE_D: 
    transf_type_ = 'std'
elif inp['Sobolev']:
    transf_type_ = 'st-stitched'
else: 
    transf_type_ = 'st-stitched'
# Create predictions and transform to numpy
if not TWODIM and not ONEDIM: 
    plot_data = make_prediction(inp, model_test_dict, data_model, transf_type = transf_type_, sc= SCALE, dn = DOUBLE_NORM)
    stats = data_model['mat_data_stats']
elif TWODIM: 
    plot_data_orig = make_prediction(inp, model_test_dict, data_model, transf_type = transf_type_, sc= SCALE, dn = DOUBLE_NORM)
    plot_data = {}
    for key in plot_data_orig.keys():
        stats = data_model['mat_data_stats']
        plot_data[key] = np.concatenate((plot_data_orig[key][:,:3], 1e-10*np.ones((plot_data_orig[key].shape[0], 5))),1)
elif ONEDIM: 
    plot_data_orig = make_prediction(inp, model_test_dict, data_model, transf_type = transf_type_, sc= SCALE, dn = DOUBLE_NORM)
    plot_data = {}
    stats = {}
    for key in plot_data_orig.keys():
        plot_data[key] = 1e-10*np.ones((plot_data_orig[key].shape[0], 8))
        plot_data[key][:,onedim_nonzero] = plot_data_orig[key][:,0]
    for key in ['stats_y_train', 'stats_y_test']:
        stats[key] = {}
        for subkey in data_model['mat_data_stats'][key].keys():
            a = data_model['mat_data_stats'][key][subkey][0]
            b = data_model['mat_data_stats'][key][subkey][1]
            stats[key][subkey] = 1e-10*np.ones((72,))
            stats[key][subkey][onedim_nonzero] = a
            stats[key][subkey][onedim_nonzero*8+onedim_nonzero+8] = b


if ('MoE-split' not in inp or not inp['MoE-split']) and not PURE_D and not INV:
    # Create plots in normalised version
    # multiple_diagonal_plots(path_plots, plot_data['all_test_labels_t'], plot_data['all_predictions_t'], 't', stats, plot_data['all_train_labels_t'], plot_data['all_predictions_t_train'])
    multiple_diagonal_plots(path_plots, plot_data['all_test_labels_t'], plot_data['all_predictions_t'], 't', stats, 'rse', None, None)

    # Create plots in original scale
    multiple_diagonal_plots(path_plots, plot_data['all_test_labels'], plot_data['all_predictions'], 'o', stats, 'rse')

    # Create plots in scale for simulation (N, mm for improved mechanical understanding / interpretability)
    plot_data_label_u = transf_units(plot_data['all_test_labels'], 'sig', forward = False)
    plot_data_pred_u = transf_units(plot_data['all_predictions'], 'sig', forward = False)
    multiple_diagonal_plots(path_plots, plot_data_label_u, plot_data_pred_u, 'u', stats, 'rse')

    # if not LINEL:
    #     # Create diagonal plots for smaller ranges: 
    #     lims = [[-2000, -2000, -600, -150000, -150000, -80000, -300, -300],
    #              [ 500,   500,  600,  150000,  150000,  80000,  300,  300]]
    #     multiple_diagonal_plots(path_plots, plot_data_label_u, plot_data_pred_u, 'u', stats, 'rse', xlim = lims, ylim = lims)

    # Create plots à la Mike: 
    if not TWODIM and not ONEDIM:
        plots_mike(data_model['mat_data_np_TrainEvalTest']['X_test'], plot_data['all_predictions'], 
                    plot_data['all_test_labels'], path_plots)    



elif 'MoE-split' in inp or inp['MoE-split']:
    raise UserWarning('This MoE model is currently not implemented')
    multiple_diagonal_plots_wrapper(path_plots, plot_data, stats, color='rse')

elif INV: 
    multiple_diagonal_plots(path_plots, plot_data['all_test_labels_t'], plot_data['all_predictions_t'], 't-inv', stats, 'rse')
    multiple_diagonal_plots(path_plots, plot_data['all_test_labels'], plot_data['all_predictions'], 'o-inv', stats, 'rse')
    plot_data_label_u = transf_units(plot_data['all_test_labels'], 'eps', forward = False)
    plot_data_pred_u = transf_units(plot_data['all_predictions'], 'eps', forward = False)
    multiple_diagonal_plots(path_plots, plot_data_label_u, plot_data_pred_u, 'u-inv', stats, 'rse')


#########################################################################################################
# 2 Testing - D-matrix
#########################################################################################################

# Define model to be used
data_model['eval_model'] = model_test_dict['standard']

if 'cVAE' in inp and inp['cVAE']: 
    # instead of D-matrix, do an inverse prediction (from the labels to the features)
    plot_data_inv = make_inv_prediction(inp, model_test_dict, data_model, 'std', num_samples=2, sc = SCALE, dn = DOUBLE_NORM)
    multiple_diagonal_plots(path_plots, plot_data_inv['all_test_features_t'][:,:8], plot_data_inv['all_pred_features_t'][:,:8], 't-inv', stats, 'rse')
elif INV:
    pass
elif not inp['Sobolev']:
    pass
elif inp['MoE-split']:
    print('Note: Derivation of MoE not implemented.')
else: 
    if not PURE_D and not TWODIM and not ONEDIM:
        # make prediction of D (via derivatives)
        plot_data_d = predict_D(data_model, transf_type=transf_type_,sc=SCALE, dn=DOUBLE_NORM)
    elif PURE_D: 
        # make prediction of D (directly with model output)
        plot_data_d = predict_pure_D(data_model, inp, model_test_dict, transf_type=transf_type_, sc=SCALE, dn=DOUBLE_NORM)
    elif TWODIM: 
        if REDUCE_DATASIZE: 
            data_model_adjusted = data_model.copy()
            for key in data_model_adjusted['mat_data_TrainEvalTest'].keys():
                data_model_adjusted['mat_data_TrainEvalTest'][key] = data_model['mat_data_TrainEvalTest'][key][:100000]
            for key in data_model_adjusted['mat_data_np_TrainEvalTest'].keys():
                data_model_adjusted['mat_data_np_TrainEvalTest'][key] = data_model['mat_data_np_TrainEvalTest'][key][:100000]
            plot_data_d_orig = predict_D(data_model_adjusted, transf_type=transf_type_,sc=SCALE, dn=DOUBLE_NORM)
            print('Only calculating predictions for first 100k datapoints to save time. ' \
                  'If you would like to calculate it for all points, use REDUCE_DATASIZE = False.')
        else: 
            plot_data_d_orig = predict_D(data_model, transf_type=transf_type_,sc=SCALE, dn=DOUBLE_NORM)
        # make sure that the output is transformed into a 8x8 shape (such that it can be plotted). Rest is just zeros (i.e. 1e-10)
        plot_data_d = {}
        stats_ext = {
            'stats_y_train': {},
            'stats_y_test': {}
        }
        for key in plot_data_d_orig.keys(): 
            plot_data_d[key] = np.concatenate((
                np.concatenate((plot_data_d_orig[key], 1e-10*np.ones((plot_data_d_orig[key].shape[0],3,5))),2),
                1e-10*np.ones((plot_data_d_orig[key].shape[0],5,8))
            ),1)
        for key in stats['stats_y_train'].keys():
            stats_ext['stats_y_train'][key] = np.concatenate((
                np.concatenate((stats['stats_y_train'][key][:3], 1e-10*np.ones((5, ))),0),
                np.concatenate((
                    np.concatenate((stats['stats_y_train'][key][3:].reshape((3,3)), 1e-10*np.ones((3,5))),1),
                    1e-10*np.ones((5,8))
                ),0).reshape((64,))
            ),0)
            stats_ext['stats_y_test'][key] = np.concatenate((
                np.concatenate((stats['stats_y_test'][key][:3], 1e-10*np.ones((5, ))),0),
                np.concatenate((
                    np.concatenate((stats['stats_y_test'][key][3:].reshape((3,3)), 1e-10*np.ones((3,5))),1),
                    1e-10*np.ones((5,8))
                ),0).reshape((64,))
            ),0)
        stats = stats_ext
    elif ONEDIM: 
        plot_data_d_orig = predict_D(data_model, transf_type=transf_type_,sc=SCALE, dn=DOUBLE_NORM)
        # make sure that the output is transformed into a 8x8 shape (such that it can be plotted). Rest is just zeros (i.e. 1e-10)
        plot_data_d = {}
        for key in plot_data_d_orig.keys(): 
            plot_data_d[key] = 1e-10*np.ones((plot_data_d_orig[key].shape[0],8,8))
            plot_data_d[key][:,onedim_nonzero, onedim_nonzero] = plot_data_d_orig[key][:,0,0]

    # diagonal plot for nonzero values of D 
    if LINEL:
        multiple_diagonal_plots_Dnz(path_plots, plot_data_d['D_sim'], plot_data_d['D_pred'], 'o', stats, 'rse') # for lin.el. / glass
    else:
        multiple_diagonal_plots_D(path_plots, plot_data_d['D_sim'], plot_data_d['D_pred'], 'o', stats, 'rse') # for RC

    # diagonal plot for nonzero values of D (in correct units [N, mm])
    plot_data_d_sim_u = transf_units(plot_data_d['D_sim'], 'D', forward = False, linel = LINEL)
    plot_data_d_pred_u = transf_units(plot_data_d['D_pred'], 'D', forward = False, linel = LINEL)
    if LINEL:
        multiple_diagonal_plots_Dnz(path_plots, plot_data_d_sim_u, plot_data_d_pred_u, 'u', stats, 'rse') # for lin.el. / glass
    else:
        multiple_diagonal_plots_D(path_plots, plot_data_d_sim_u, plot_data_d_pred_u, 'u', stats, 'rse') #for RC
        # Create diagonal plots with limits: 
        # lims: D_mdiag, D_melse, D_mbdiag, D_mbelse, D_bdiag, D_belse, D_s
        lims = [[0e6, -4e5, -1.5e8, -8e7,   0e10, -2e9, 2.5e6],
                [9e6,  4e5,  1.5e8,  8e7, 3.5e10,  2e9, 3e6]]
        multiple_diagonal_plots_D(path_plots, plot_data_d_sim_u, plot_data_d_pred_u, 'u', stats, 'rse', xlim = lims, ylim = lims)

    if not TWODIM and not ONEDIM: 
        mike_plot_data_x = transf_units(data_model['mat_data_np_TrainEvalTest']['X_test'], 'eps-t', forward = False)
        mike_plot_data_D_pred = np.concatenate((plot_data_d_pred_u[:,:6,:6].reshape((-1,36)), plot_data_d_pred_u[:,6,6].reshape((-1,1)), plot_data_d_pred_u[:,7,7].reshape((-1,1))), axis = 1)
        mike_plot_data_D_sim = np.concatenate((plot_data_d_sim_u[:,:6,:6].reshape((-1,36)), plot_data_d_sim_u[:,6,6].reshape((-1,1)), plot_data_d_sim_u[:,7,7].reshape((-1,1))), axis = 1)
        plots_mike(mike_plot_data_x, mike_plot_data_D_pred,
                mike_plot_data_D_sim, path_plots, tag = 'D')

# close wandb
wandb.finish()


# save images to same folder as the training files

save_folder = True
filenames = ['diagonal_match_D_nonzero_o.png', 'diagonal_match_D_nonzero_u.png', 'diagonal_match_'+'D_nonzero_u_newlim.png', 'testo.png', 'testo-D.png',
             'diagonal_match_original.png', 'diagonal_match_original_units.png', 'diagonal_match_transformed.png', 'diagonal_match_original_units_newlim.png',]
            # 'new_data\\trained_model.pt', 'new_data\\trained_model_best.pt']

if PURE_D:
    filenames = ['diagonal_match_D_nonzero_o.png', 'diagonal_match_D_nonzero_u.png', 'testo-D.png']
elif INV: 
    filenames =  ['diagonal_match_original.png', 'diagonal_match_original_units.png', 'diagonal_match_transformed.png']
elif 'cVAE' in inp and inp['cVAE'] or not inp['Sobolev']: 
    filenames = ['testo.png', 'diagonal_match_original.png', 'diagonal_match_original_units.png', 'diagonal_match_transformed.png']


if save_folder and not inp['MoE-split']:
    src_folder = os.path.join(os.getcwd(), '04_Training\\plots')
    dest_folder = os.path.join(os.getcwd(), '04_Training\\new_data\\'+add_path)
    copy_files_to_plots_folder(src_folder, dest_folder, filenames)
elif save_folder and inp['MoE-split']:
    src_folder = os.path.join(os.getcwd(), '04_Training\\plots\\MoE')
    dest_folder = os.path.join(os.getcwd(), '04_Training\\new_data\\'+add_path)
    copy_all_files(src_folder, dest_folder)
