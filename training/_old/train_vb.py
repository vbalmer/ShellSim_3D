import pickle
import numpy as np
import torch
import os
import wandb

from sklearn.model_selection import train_test_split
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from lightning.pytorch import seed_everything
seed_everything(42)

from data_work import *
from FFNN_class_light import *
from train_utils import *


wandb.login()


#################################################################################################################
## 0 - Read in the data
#################################################################################################################

path = os.getcwd()
# NAME =  '04_Training\data\data_20241212_1605_fake'                # Reinforced concrete linear elastic
# NAME = '04_Training\data\data_20250207_1122_fake'                   # Glass
# NAME = '04_Training\data\data_20250311_1701_fake'                   # RC nonlinear
# NAME = '04_Training\data\data_20250328_1443_fake'                   # RC nonlinear, "pimped with data in smaller range"
# NAME = '04_Training\data\data_20250608_1838_fake'                   # RC nonlinear, manufactured data
NAME = '04_Training\data\data_20260105_1639_fake'                     # RC nonlinear, one dimension
# NAME = '04_Training\data\data_20250508_2352_fake_David'
# NAME = '04_Training\data\experimental_data_David'
SAVE_FOLDER = True
SWEEP = False
LINEL = False                                                       # only set false for RC nonlinear, otw true
PURE_D = False                                                      # predict only D directly
INV = False                                                          # to predict eps from sigma
SOBOLEV = True
SCALE = False
DOUBLE_NORM = False
LOG_NORM = False
TWODIM = True
ONEDIM = False
ZEROS = False
GEOM_SIZE = 4



path_data = os.path.join(path, NAME)

# Get sig, eps data
new_data_eps_np = read_data(path_data, 'eps')           # if augmented data set: use 'eps_add', etc. else: use 'eps'
new_data_sig_np_ = read_data(path_data, 'sig')           # if augmented data set: use 'sig_add', etc. else: use 'sig'
amt_data_points = new_data_sig_np_.shape[0]

# Adjust sig to units required for NN training: [MN, cm] instead of [N, mm] (240813_UpdateMeeting_1, ~slide 9)
# will be reversed after prediction of D, sig
new_data_sig_np = transf_units(new_data_sig_np_, 'sig', forward = True)
new_data_inp = new_data_eps_np
new_data_out = new_data_sig_np


# Additional input variables (here: t)
new_data_t_np = read_data(path_data, 't')
new_data_inp = np.concatenate((new_data_eps_np, new_data_t_np),1)
new_data_inp = new_data_inp[:,0:12]                         # only for RC
new_data_inp= transf_units(new_data_inp, 'eps-t', forward = True)
# Additional variable in labels: D, used for sobolev loss if indicated, and for checking in deployment
new_data_De_np = read_data(path_data, 'De')
new_data_De_np = transf_units(new_data_De_np.reshape((amt_data_points, 8, 8)), 'D', forward = True, linel=LINEL)     #change units
new_data_De_np = new_data_De_np.reshape((amt_data_points, 64))
new_data_out = np.concatenate((new_data_sig_np, new_data_De_np),1)

if PURE_D:
    new_data_De_np_ = new_data_De_np.reshape((-1,8,8))
    new_data_out = np.concatenate((new_data_De_np_[:,:6,:6].reshape((-1,36)), new_data_De_np_[:,6,6].reshape((-1,1)), new_data_De_np_[:,7,7].reshape((-1,1)),), axis = 1)
if INV: 
    new_data_inp = transf_units(np.concatenate((new_data_sig_np_, new_data_t_np[:,0:3]),1), 'sig-t', forward = True)
    new_data_out = transf_units(new_data_eps_np, 'eps', forward = True)
    # alternatively: 
    # new_data_inp = new_data_sig_np
    # new_data_out = transf_units(np.concatenate((new_data_eps_np, new_data_t_np[:,0:3]),1), 'eps-t', forward = True)
if TWODIM: 
    new_data_inp = np.concatenate((new_data_eps_np[:,:3], new_data_t_np[:,0:GEOM_SIZE]),1)
    new_data_De_np_ = new_data_De_np.reshape((-1,8,8))
    new_data_out = np.concatenate((new_data_sig_np[:,:3], new_data_De_np_[:,:3,:3].reshape((-1,9))),1)
if ONEDIM: 
    data_col_nonzero = 0
    new_data_inp = np.concatenate((new_data_eps_np[:,data_col_nonzero:data_col_nonzero+1], new_data_t_np[:,0:3]),1)
    new_data_out = np.concatenate((new_data_sig_np[:,data_col_nonzero:data_col_nonzero+1], (new_data_De_np.reshape((-1,8,8))[:,data_col_nonzero, data_col_nonzero]).reshape((-1,1))),1)
if ZEROS: 
    data_col_nonzero = 0
    new_data_inp = np.concatenate((1e-8*np.ones((amt_data_points, 8)), new_data_t_np[:,0:3]), 1)
    new_data_inp[:, data_col_nonzero] = new_data_eps_np[:,data_col_nonzero]
    new_data_out = 1e-8*np.ones((amt_data_points,72))
    new_data_out[:,data_col_nonzero] = new_data_sig_np[:,data_col_nonzero]
    new_data_out[:,data_col_nonzero*8+data_col_nonzero+8] = new_data_De_np[:,data_col_nonzero*8+data_col_nonzero]

# Select amount of bins
nbins = 50

print('Total amount of data points:', amt_data_points)
print('Input shape: ', new_data_inp.shape[1])
print('Output shape: ', new_data_out.shape[1])


#################################################################################################################
# 1 Train-Eval-Test Split
# for train-eval-test = 0.7 - 0.2 - 0.1
#################################################################################################################

# Split into aux and test
X_aux, X_test, y_aux, y_test = train_test_split(new_data_inp,new_data_out, test_size=0.1, random_state=42)

# Split into train and eval
X_train, X_eval, y_train, y_eval = train_test_split(X_aux,y_aux, test_size=0.222, random_state=42)

# Save split data
save_path = os.path.join(path, '04_Training\\new_data')
save_data(X_train, X_eval, X_test, y_train, y_eval, y_test, save_path)
path = os.path.join(os.getcwd(), '04_Training')
path_plots = os.path.join(path, 'plots')

# Plot to check
n_max_hist = 8
if TWODIM:
    n_max_hist = 3
elif ONEDIM: 
    print('Please comment out the histograms, they cannot be shown for one dimension')
# histogram(X_train[:,0:n_max_hist], y_train[:,0:n_max_hist], amt_data_points, nbins, 'eps', path_plots)
# histogram(X_train[:,0:n_max_hist], y_train[:,0:n_max_hist], amt_data_points, nbins, 'sig', path_plots)
# histogram(X_train[:,n_max_hist:], y_train[:,0:n_max_hist], amt_data_points, nbins, 't2', path_plots)
# histogram(X_train[:,n_max_hist:], y_train[:,n_max_hist:].reshape((-1,n_max_hist,n_max_hist)), amt_data_points, nbins, 'De', path_plots)

# histogram(X_eval[:,0:8], y_eval[:,0:8], amt_data_points, nbins, 'eps', path_plots)
# histogram(X_eval[:,0:8], y_eval[:,0:8], amt_data_points, nbins, 'sig', path_plots)
# histogram(X_eval[:,8], y_eval[:,0:8], amt_data_points, nbins, 't', path_plots)
# histogram(X_test[:,0:8], y_test[:,0:8], amt_data_points, nbins, 'eps', path_plots)
# histogram(X_test[:,0:8], y_test[:,0:8], amt_data_points, nbins, 'sig', path_plots)
# histogram(X_test[:,8], y_test[:,0:8], amt_data_points, nbins, 't', path_plots)

# plots_mike_dataset(X_train, X_test, y_train, y_test, path_plots, tag = 'test')
# plots_mike_dataset(X_train, X_eval, y_train, y_eval, path_plots, tag = 'eval')
# plot_nathalie(X_train[:,0:8], X_test[:, 0:8], path_plots, tag = 'eps')
# plot_nathalie(y_train[:,0:8], data_in_test=None, save_path=path_plots, tag = 'sig')


#################################################################################################################
## 2 - Normalisation
################################################################################################################

stats_X_train = statistics(X_train)
stats_y_train = statistics(y_train)
stats_X_eval = statistics(X_eval)
stats_y_eval = statistics(y_eval)
stats_X_test = statistics(X_test)
stats_y_test = statistics(y_test)

mat_data_stats = {'stats_X_train': stats_X_train,
                  'stats_y_train': stats_y_train,
                  'stats_X_test': stats_X_test,
                  'stats_y_test': stats_y_test
                  }

with open(os.path.join(save_path, 'mat_data_stats.pkl'), 'wb') as fp:
        pickle.dump(mat_data_stats, fp)


if LOG_NORM: 
    X_train_t = transform_data(X_train, mat_data_stats, forward=True, type=['x-log']*X_train.shape[1], sc = SCALE, dn = DOUBLE_NORM)
    y_train_t = transform_data(y_train, mat_data_stats, forward=True, type=['y-log']*8+['y-lg-stitched']*64, sc = SCALE, dn = DOUBLE_NORM, 
                               log_add = {'add_data_eps': X_train[:,0:8], 'add_data_sig': y_train[:,0:8]})
    X_eval_t = transform_data(X_eval, mat_data_stats, forward=True, type=['x-log']*X_train.shape[1], sc = SCALE, dn = DOUBLE_NORM)
    y_eval_t = transform_data(y_eval, mat_data_stats, forward=True, type=['y-log']*8+['y-lg-stitched']*64, sc = SCALE, dn = DOUBLE_NORM,
                              log_add = {'add_data_eps': X_eval[:,0:8], 'add_data_sig': y_eval[:,0:8]})
    X_test_t = transform_data(X_test, mat_data_stats, forward=True, type=['x-log']*X_train.shape[1], sc = SCALE, dn = DOUBLE_NORM)
    y_test_t = transform_data(y_test, mat_data_stats, forward=True, type=['y-log']*8+['y-lg-stitched']*64, sc = SCALE, dn = DOUBLE_NORM,
                              log_add = {'add_data_eps': X_test[:,0:8], 'add_data_sig': y_test[:,0:8]})
elif TWODIM and SOBOLEV:
    X_train_t = transform_data(X_train, mat_data_stats, forward=True, type=['x-std']*X_train.shape[1], sc = SCALE, dn = DOUBLE_NORM)
    y_train_t = transform_data(y_train, mat_data_stats, forward=True, type=['y-std']*3+['y-st-stitched']*9, sc = SCALE, dn = DOUBLE_NORM)
    X_eval_t = transform_data(X_eval, mat_data_stats, forward=True, type=['x-std']*X_train.shape[1], sc = SCALE, dn = DOUBLE_NORM)
    y_eval_t = transform_data(y_eval, mat_data_stats, forward=True, type=['y-std']*3+['y-st-stitched']*9, sc = SCALE, dn = DOUBLE_NORM)
    X_test_t = transform_data(X_test, mat_data_stats, forward=True, type=['x-std']*X_train.shape[1], sc = SCALE, dn = DOUBLE_NORM)
    y_test_t = transform_data(y_test, mat_data_stats, forward=True, type=['y-std']*3+['y-st-stitched']*9, sc = SCALE, dn = DOUBLE_NORM)
elif ONEDIM and SOBOLEV: 
    X_train_t = transform_data(X_train, mat_data_stats, forward=True, type=['x-std']*X_train.shape[1], sc = SCALE, dn = DOUBLE_NORM)
    y_train_t = transform_data(y_train, mat_data_stats, forward=True, type=['y-std']*1+['y-st-stitched']*1, sc = SCALE, dn = DOUBLE_NORM)
    X_eval_t = transform_data(X_eval, mat_data_stats, forward=True, type=['x-std']*X_train.shape[1], sc = SCALE, dn = DOUBLE_NORM)
    y_eval_t = transform_data(y_eval, mat_data_stats, forward=True, type=['y-std']*1+['y-st-stitched']*1, sc = SCALE, dn = DOUBLE_NORM)
    X_test_t = transform_data(X_test, mat_data_stats, forward=True, type=['x-std']*X_train.shape[1], sc = SCALE, dn = DOUBLE_NORM)
    y_test_t = transform_data(y_test, mat_data_stats, forward=True, type=['y-std']*1+['y-st-stitched']*1, sc = SCALE, dn = DOUBLE_NORM)
elif SOBOLEV:
    X_train_t = transform_data(X_train, mat_data_stats, forward=True, type=['x-std']*X_train.shape[1], sc = SCALE, dn = DOUBLE_NORM)
    y_train_t = transform_data(y_train, mat_data_stats, forward=True, type=['y-std']*8+['y-st-stitched']*64, sc = SCALE, dn = DOUBLE_NORM)
    X_eval_t = transform_data(X_eval, mat_data_stats, forward=True, type=['x-std']*X_train.shape[1], sc = SCALE, dn = DOUBLE_NORM)
    y_eval_t = transform_data(y_eval, mat_data_stats, forward=True, type=['y-std']*8+['y-st-stitched']*64, sc = SCALE, dn = DOUBLE_NORM)
    X_test_t = transform_data(X_test, mat_data_stats, forward=True, type=['x-std']*X_train.shape[1], sc = SCALE, dn = DOUBLE_NORM)
    y_test_t = transform_data(y_test, mat_data_stats, forward=True, type=['y-std']*8+['y-st-stitched']*64, sc = SCALE, dn = DOUBLE_NORM)
else: 
    X_train_t = transform_data(X_train, mat_data_stats, forward=True, type=['x-std']*X_train.shape[1], sc = SCALE, dn = DOUBLE_NORM)
    y_train_t = transform_data(y_train, mat_data_stats, forward=True, type=['y-std']*y_train.shape[1], sc = SCALE, dn = DOUBLE_NORM)
    X_eval_t = transform_data(X_eval, mat_data_stats, forward=True, type=['x-std']*X_train.shape[1], sc = SCALE, dn = DOUBLE_NORM)
    y_eval_t = transform_data(y_eval, mat_data_stats, forward=True, type=['y-std']*y_train.shape[1], sc = SCALE, dn = DOUBLE_NORM)
    X_test_t = transform_data(X_test, mat_data_stats, forward=True, type=['x-std']*X_train.shape[1], sc = SCALE, dn = DOUBLE_NORM)
    y_test_t = transform_data(y_test, mat_data_stats, forward=True, type=['y-std']*y_train.shape[1], sc = SCALE, dn = DOUBLE_NORM)



# Plot to check
# histogram(X_train_t[:,0:n_max_hist], y_train_t[:,0:n_max_hist], amt_data_points, nbins, 'eps', path_plots)
# histogram(X_train_t[:,0:n_max_hist], y_train_t[:,0:n_max_hist], amt_data_points, nbins, 'sig', path_plots)
# histogram(X_train_t[:,n_max_hist:], y_train_t[:,0:n_max_hist], amt_data_points, nbins, 't2', path_plots)
# histogram(X_train[:,n_max_hist:], y_train_t[:,n_max_hist:].reshape((-1,n_max_hist,n_max_hist)), amt_data_points, nbins, 'De', path_plots)

# histogram(X_train_t[:,8], y_train_t[:,0:8], amt_data_points, nbins, 't', path_plots)
# histogram(X_eval_t[:,0:8], y_eval_t[:,0:8], amt_data_points, nbins, 'eps', path_plots)
# histogram(X_eval_t[:,0:8], y_eval_t[:,0:8], amt_data_points, nbins, 'sig', path_plots)
# histogram(X_eval_t[:,8], y_eval_t[:,0:8], amt_data_points, nbins, 't', path_plots)
# histogram(X_test_t[:,0:8], y_test_t[:,0:8], amt_data_points, nbins, 'eps', path_plots)
# histogram(X_test_t[:,0:8], y_test_t[:,0:8], amt_data_points, nbins, 'sig', path_plots)
# histogram(X_test_t[:,8], y_test_t[:,0:8], amt_data_points, nbins, 't', path_plots)


################################################################################################################
## 3 - Define input parameters
################################################################################################################

# Save input data as torch
loaders, mat_torch = data_to_torch(X_train_t, y_train_t, X_eval_t, y_eval_t, 
                        X_test_t, y_test_t, 
                        save_path, SOBOLEV, batch_size = None, tag = (PURE_D or INV))

# Plot to check (only for sig, eps)
# histogram_torch(train_loader, amt_data_points, nbins, 'eps')
# histogram_torch(train_loader, amt_data_points, nbins, 'sig')
# histogram_torch(val_loader, amt_data_points, nbins, 'eps')
# histogram_torch(val_loader, amt_data_points, nbins, 'sig')
# histogram_torch(test_loader, amt_data_points, nbins, 'eps')
# histogram_torch(test_loader, amt_data_points, nbins, 'sig')

###############################################
# definition of input parameters for model
###############################################

# definition of input parameters for non-sweep run
inp = {
    # Network architecture / characteristics
    'data_name': NAME,
    'scaling': SCALE,                               # defined at the top.
    'input_size': 7,                               # 8 eps + t (thickness) = 9 . For DeepONet use 8 as input size!
    'out_size': 3,                                  # 8 sig, or: 38 D-matrix, or 1: energy, or: 3 for 2D-network, or: 1 for 1D-network
    'hidden_layers': str([512]*5),
    'batch_size': 4096,                             # Can be defined here, if None: runs with single-batch
    'num_epochs': 300,
    'switch_step_percentage': 1,                    # Percentage after which to switch to LBFGS instead of Adam 
    'activation': 'ELU',
    'learning_rate': 0.005,
    'lr_scheduler':'standard',                      # 'standard': the one used by mike; 'plateau': reduceLRonPlateau
    'dropout_rate': 0,
    'BatchNorm': False,
    'kfold': False, 
    'num_samples': 5, 
    'fourier_mapping': False,  
    'loss_type': 'MSELoss',                         # can be 'MSELoss', 'HuberLoss', 'MSLELoss', 'wMSELoss', 'RMSELoss'
    'Split_Loss': False,                            # whether loss should be split into m, b, s
    'w_mbs': [0.1, 0.8, 0.1],                       # weights [wm, wb, ws] (neglected if Split_loss = false)
    'w_Dmbs':[0.4, 0.2, 0.4] ,                      # weights [wDm, wDb, wDs] (neglected if Split_loss = false)

    # Training type
    'simple_m': True,                               # this should always be set to True.                         

    # Network type (Sobolev, Pretrained, DeepONet, MoE)          
    'Sobolev': SOBOLEV,                             # Defined in Step ##3 above, DO NOT CHANGE HERE
    'w_s': str([0.9,0.1]),                          # weights [w1, w2] for 1st, 2nd order loss with sobolev (neglected if sobolev = False), or 'max'
    'w_smooth': None,                               # weight for adding a third loss term (fourth-order), to smoothen 2nd derivative (str([1e-2,None]))
                                                    # if don't want to use, set = None; only use if Sobolev is not False; full = full calculation of derivative
    'only_diag': False,                             # only takes diagonal values of stiffness for loss calculation                        
    'w_diag': None,                                 # weights to add to the losses of diagonal entries of stiffness matrix 
                                                    # e.g. str([1,0.2]): 0.2 is used for off-diag terms in all submatrices, else: None
                                                    # if None: no weight is used, diagonal entries are not changed in magnitude
    'w_range_D': False,                             # weights for stiffness according to their ranges
    'w_nonzero': False,                             # only consider nonzero values for calculating the loss of stiffness D
    'energy': False,                                # if true: uses sobolev-energy loss instead of standard sobolev loss
                                                    # only implemented for NN, not for DeepONet
    'pretrain': None,                               # use the pretrained model, with version (str); if str=None: pretrained model is not used, e.g. str(['v_186', '_39967'])
                                                    # this is only possible w/o split_net
    # 'pretrain': str(['v_303', '_19979']),             
    'hidden_layers_new': str([64]*1),               # For the added layers when using pretrained model
    'BatchNorm_new':False,                          # For the added layers when using pretrained model
    'activation_new':'ELU',                         # For the added layers when using pretrained model
    'DeepONet': False,
    'num_trunk': 3,
    'MoE': False,                                   # MoE without splitting data (random assignment)
    'MoE-split': False,                             # run MoE with splitting data and assigning to experts

    # cVAE variables, if not specified here, variables above are used (e.g. learning rate, activation)
    'cVAE': False,
    'latent_dim': 32,
    'layers_enc': str([64]*10),
    'layers_dec': str([64]*10),
    'w_cVAE': str([1, 1e-4, 1])                     # weights of the different losses: reproduction, KLD, performance
}

# saving inp file for use in testing
with open(os.path.join(save_path, 'inp.pkl'), 'wb') as fp:
        pickle.dump(inp, fp)



# constant values that should be added in the sweep dict and are not part of sweep_config
constant_inp = {
    'data_name': NAME,
    'simple_m': True,
    'lr_scheduler':'standard',
    'num_samples': 5,
    'BatchNorm': False,
    'Sobolev': SOBOLEV,                           
    'kfold': False,
    }


################################################################################################################
## 4 - Define main train function
################################################################################################################
def main_train(config=None, project_name='ShellSim_sweep_v3', save_folder = False):
    
    # logging input parameters for wandb
    with wandb.init(config=config, project=project_name):

        #__________________________________________________
        # 4a - Set hyperparameters, create instance of model
        #__________________________________________________

        wandb.config.update(constant_inp)
        
        inp_dict = wandb.config
        inp = wandb.config
        wandb.config.update({'activation': wandb.config['activation']}, allow_val_change=True)
        wandb.config.update({'loss_type': wandb.config['loss_type']}, allow_val_change=True)

        # create instance of model
        # for customisation of individual nets, adjust this function
        model_dict, inp_dict = model_instance(inp, inp_dict, save_path)


        # print model architecture
        model_print(model_dict, inp)

        #________________________________________________________
        # 4b - Train and evaluate model, save best evaluated model
        #________________________________________________________

        if inp['simple_m']:
        # train with simple pytorch function from train_vb.py
            if not inp['MoE']:
                model = simple_train(inp, model_dict, mat_torch, save_path)
                torch.save(model.state_dict(), os.path.join(save_path, 'last_trained_model.pt'))
                model.eval()
            elif inp['MoE']:
                # Training a MoE directly without splitting data.
                model = simple_train(inp, model_dict, mat_torch, save_path)
                torch.save(model.state_dict(), os.path.join(save_path, 'last_trained_model.pt'))
                model.eval()
                if inp['MoE-split']:
                    raise Warning('This version of the MoE is outdated.')
                    # Split the data for training of MoE
                    thresholds = np.array([200, 1000])             # N/mm
                    mat_torch_exp1, mat_torch_exp2, mat_torch_exp3, mat_torch_MoE = data_split_MoE(mat_torch, stats_y_train, thresholds, type = 'random')
                    
                    # Train the 4 individual models
                    expert1 = simple_train(inp, model_dict, mat_torch_exp1, save_path, no = 'exp1')
                    print('Training of expert 1 finished')
                    expert2 = simple_train(inp, model_dict, mat_torch_exp2, save_path, no = 'exp2')
                    print('Training of expert 2 finished')
                    expert3 = simple_train(inp, model_dict, mat_torch_exp3, save_path, no = 'exp3')
                    print('Training of expert 3 finished')
                    MoE = simple_train(inp, model_dict, mat_torch_MoE, save_path, no = 'MoE')
                    print('Training of MoE finished')
                    expert1.eval()
                    expert2.eval()
                    expert3.eval()
                    MoE.eval()

        else:
            # train with pytorch lightning
            raise RuntimeWarning('The pytorch lightning version of this code is outdated. Please use simple_m = True')
            if not inp['Split_Net_all']:
                # Define logs, callbacks and trainer
                trainer_dict = trainer_instance(inp, n_patience = 100, log_every = 8)
                if not inp['Split_Net']:
                    # Train model
                    print('Training standard model')
                    trainer_dict['standard'].fit(model_dict['standard'], loaders['train'], loaders['val'])
                elif inp['Split_Net']:
                    # Train model
                    print('Training split model')
                    trainer_dict['m'].fit(model_dict['m'], loaders['train_m'], loaders['val_m'])
                    trainer_dict['b'].fit(model_dict['b'], loaders['train_b'], loaders['val_b'])
                    trainer_dict['s'].fit(model_dict['s'], loaders['train_s'], loaders['val_s'])

            elif inp['Split_Net_all']:
                # Multi-GPU (not working in parallel, just sequentially.)
                if torch.cuda.is_available():
                    def train_gpu(i, gpu_no):
                        trainer_dict = trainer_instance(inp, n_patience = 50, log_every = 8, gpu_id = gpu_no, inp_all = inp_dict)
                        with wandb.init(project = 'ShellSim_FFNN_vp', name = 'model'+ str(i) + '_' + str(torch.randn((1,1))), reinit = True):
                            trainer_dict[str(i)].fit(model_dict[str(i)], loaders['train_'+str(i)], loaders['val_'+str(i)])
                        return trainer_dict
                    
                    print('Training split_all model')
                    for i in range(8):
                        gpu_id = i % 4
                        trainer_dict = train_gpu(i, gpu_id)
                
                # CPU
                else: 
                    print('Training split_all model')
                    for i in range(8):
                        trainer_dict = trainer_instance(inp, n_patience = 100, log_every = 8, inp_all=inp_dict)
                        trainer_dict[str(i)].fit(model_dict[str(i)], loaders['train_'+str(i)], loaders['val_'+str(i)])
        
        
        # Copy model and data to separate folder.
        filenames = ['inp.pkl', 'mat_data_np_TrainEvalTest.pkl', 'mat_data_stats.pkl', 'mat_data_TrainEvalTest.pkl',
                    'last_trained_model.pt', 'best_trained_model.pt']

        if save_folder:
            src_folder = os.path.join(os.getcwd(), '04_Training\\new_data')
            base_dest_folder = os.path.join(os.getcwd(), '04_Training\\new_data\\_simple_logs')
            copy_files_with_incremented_version(src_folder, base_dest_folder, filenames)

    return inp


################################################################################################################
## 5 Train normally OR define sweep parameters and run sweep (if desired)
################################################################################################################

if not SWEEP:
    #_______________________________________
    # 5a - Call function to train without sweep
    # _______________________________________
    inp_ = inp
    # print(inp_)
    inp = main_train(config = inp_, project_name = 'ShellSim_FFNN_v1', save_folder = SAVE_FOLDER)

elif SWEEP:
    #________________________________________
    # 5b - Call function to train with sweep
    # ______________________________________
    # Define sweep configuration
    sweep_config = {
        "method":"random",
        "metric": {"goal": "minimize", "name": "best_val_loss"},
        "parameters":{
            "input_size": {"values": [8]},      # needs to be 8 for DeepONet
            "out_size": {"values": [8]},
            "num_epochs": {"values": [30000]},
            "learning_rate": {"values": [0.001]},
            "hidden_layers": {"values": [str([64]*20), str([64]*10), str([128]*10), str([128]*20)]},
            "dropout_rate": {"values": [0, 0.05]},
            "loss_type": {"values": ['MSELoss']},
            "activation": {"values": ['ELU','ReLU']},
            "fourier_mapping": {"values": [True, False]},
            "pretrain": {"values": [None]},
            "w_s": {"values": ['max', str([0.4, 0.6]), str([0.5, 0.5]), str([0.25, 0.75])]},    #, str([0.2, 0.8]), str([0.3, 0.7])]},
            "DeepONet":{"values": [True]},
        }
    }

    # start sweep
    sweep_id = wandb.sweep(sweep = sweep_config, project='ShellSim_sweep_v3')
    wandb.agent(sweep_id, 
                function=main_train,
                count = 1)


wandb.finish()