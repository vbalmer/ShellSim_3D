# vb, 29.03.2026

import os
import glob
from pathlib import Path
import pickle

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.offsetbox import AnchoredText


from architectures import FFNN
from data_utils import get_normalised_data, data_to_torch, transform_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


################################ main test function ################################

def test_NN_model(test_data: dict, stats: dict, save_path: str, version: int, 
                  plot_path = 'training\\plots_test'): 
    """
    Main testing function. inp dict and trained model are collected from saved model path.
    Data and stats are passed directly from train file. 

    Args: 
        test_data  (dict): test_data (from train script)
        stats      (dict): Statistics (from train script). Only train statistics will be used for normalisation
        save_path   (str): location where trained model is saved
        version     (int): version of trained model to look for

    Returns: 
        plots in plot_test folder

    """

    ## 0 - Create test model instance
    inp = get_inp_from_folder(save_path, version)
    test_model = test_model_instance(inp, save_path, version)

    ## 1 - Predict stresses
    plot_data_sig = predict_sig(test_model, inp, test_data, stats, sobolev=inp['Sobolev'])

    ## 2 - Predict stiffnesses
    # plot_data_D = predict_D(test_model, test_data, stats, sobolev=inp['Sobolev'])
    # TODO!



    ## 3 - Create diagonal plots for stresses

    # 3a - Normalised data
    multiple_diagonal_plots(plot_data_sig['labels_norm'], plot_data_sig['pred_norm'], 
                            stats, 'rse', plot_path, transf = 't')

    # 3b - Transformed data (no unit transformation required)
    multiple_diagonal_plots(plot_data_sig['labels'], plot_data_sig['pred'], 
                            stats, 'rse', plot_path, transf = 'u')

    ## 4 - Create diagonal plots for stiffnesses

    # 4a - Normalised data
    # TODO!

    # 4b - Transformed data (no unit transformation required)
    # TODO!



    return


################################ auxiliary functions for testing ################################

def get_inp_from_folder(path:str, version: int) -> dict:
    """
    Read inp file that was used to create model specified in save_path.
    
    """
    full_path = os.path.join(path, 'v_'+str(version))
    with open(os.path.join(full_path, 'inp.pkl'),'rb') as handle:
        inp = pickle.load(handle)

    return inp

def test_model_instance(inp:dict, path:str, version:int) -> FFNN:
    """
    Create instance of test model

    Args: 
        inp     (dict): Hyperparameter configuration
        path     (str): Path to trained model
        version  (int): Version of trained model

    """

    
    model_path = get_full_model_path(path, version)
    
    model_test = FFNN(inp)
    model_test.load_state_dict(torch.load(model_path, map_location = device))
    model_test.to(device)
    model_test.eval()


    return model_test

def get_full_model_path(path:str, version:int):
    """
    Extract correct "best_trained_model_xx.pt" file from given version folder
    
    Args: 
        path        (str):   Path to folder where model can be found.
        version     (int):   Version of trained model.
    Returns: 
        model_path  (str):   Path to trained model.

    """

    # Build path to model
    model_folder = os.path.join(path, f'v_{version}')
    matches = glob.glob(os.path.join(model_folder, "**", "best_trained_model_*.pt"), recursive=True)
    model_path = matches[0]

    # Extract epoch from filename
    stem = Path(model_path).stem  # e.g. "best_trained_model_123"
    epoch = stem.replace("best_trained_model_", "")
    print(f'Testing model v_{version} with best trained model at epoch {epoch}')

    return model_path


def predict_sig(test_model:FFNN, inp:dict, test_data:dict, stats:dict, sobolev:bool):
    """
    Evaluates trained model at given test data set. 
    Returns predictions and labels for normalised [-]  and standard data [N,mm].

    Args: 
        test_model  (FFNN):     Trained model in eval mode
        inp         (dict):     Hyperparameters of trained model
        test_data   (dict):     Contains two np arrs: X_test and y_test
        stats       (dict):     Statistics for dataset. only train stats used for normalisation.
        sobolev     (bool):     if true: includes sobolev-transformation in normalisation.

    Returns: 
        plot_data   (dict):     Containing predictions and labels (each normalised and standard)  
    
    """

    # 1 - normalise data & transfer to torch
    test_data_norm = get_normalised_data(test_data, stats, sobolev)
    test_data_norm_torch = data_to_torch(test_data_norm)

    # 2 - make prediction in normalised coordinates
    pred_norm = get_normalised_sig_pred(inp, test_data_norm_torch, test_model)
    
    # 3 - make predictions in [N,mm] by reverting normalisation
    pred = get_unnormalised_data(pred_norm, stats, sobolev)

    # 4 - collect data into dataset
    plot_data = {
        'pred': pred,
        'pred_norm': pred_norm.cpu().detach().numpy(),
        'labels': test_data['y_test'][:,:6],
        'labels_norm': test_data_norm['y_test_t'][:,:6]
    }

    return plot_data

def get_normalised_sig_pred(inp:dict, data:dict, test_model: FFNN) -> torch.Tensor:
    """
    Get normalised predictions for sigma (from actual call to FFNN).

    Args: 
        inp         (dict): Required for batchsize
        data        (dict): test data in normalised, torch version (X_test_tt, y_test_tt)
        test_model  (FFNN): trained model in eval mode.

    Returns: 
        pred_norm   (torch.Tensor): Normalised predictions for test dataset.
    """

    if inp['batch_size'] is None:
        batch_size_test = data['X_test_tt'].shape[0]
    else: 
        batch_size_test = inp['batch_size']
        
    test_dataset = TensorDataset(data['X_test_tt'].to(device))
    test_loader = DataLoader(test_dataset, batch_size = batch_size_test, shuffle = False)

    pred_norm = []
    for (X_test_tt, ) in test_loader:
        preds = test_model(X_test_tt)
        pred_norm.append(preds)
    pred_norm = torch.cat(pred_norm, dim = 0)

    return pred_norm

def get_unnormalised_data(pred_norm: torch.Tensor, stats: dict, sobolev: bool) -> torch.Tensor:
    """
    Undo normalisation to get predictions in correct units [N, mm]

    Args: 
        pred_norm   (torch.Tensor): Normalised predictions
        stats       (dict):         Statistical values required for unnormalisation (only train used)
        sobolev     (bool):         if true: includes transformation of sobolev data
    
    Returns: 
        pred        (torch.Tensor): Unnormalised predictions (N, mm)

    """

    ydim = pred_norm.shape[1]
    if sobolev:
        norm_type_y = ['y-std']*ydim
    else: 
        norm_type_y = ['y-std']*6+['y_st-stitched']*36

    pred = transform_data(pred_norm.cpu().detach().numpy(), stats, forward = False, type = norm_type_y)


    return pred


def predict_D(test_model: FFNN, inp: dict, test_data: dict, stats: dict, sobolev:bool):
    
    if sobolev: 
        pass
        # TODO

    else: 

        print('This model was not trained with Sobolev losses. Predictions may differ significantly from labels in dataset.')

    return

################################ auxiliary functions for plotting ################################

def multiple_diagonal_plots(Y: np.array, preds: np.array, stats: dict, error_type: str, save_path: str, transf:str):
    """
    Creates diagonal plots to show results of testing. 

    Args: 
        Y       (np.arr):   test labels (shape: ntest, 6)
        preds   (np.arr):   test predictions (shape: ntest, 6) 
        stats     (dict):   statistical values utilised in error calculation (only train values)
        error_type (str):   'rse', 'nrse', ...
        save_path  (str):   location where plot will be stored.
        transf    (dict):   't': plotting normalised data, 'u': plotting original units data

    Returns: 
        diagonal plots as png in save_path
    """


    # Calculate relevant errors
    errors = calculate_errors(Y, preds, stats, transf, id = 'sig')

    # Define figures
    fig, ax = plt.subplots(2, 3, figsize=[15.5, 10], dpi=100)
    fig.subplots_adjust(wspace=0.5)

    # Plot figures
    errors_ = {key: arr.reshape(-1, 2, 3) for key, arr in errors.items()}
    scatters = plot_main_figure(ax, Y.reshape((-1,2,3)), preds.reshape((-1,2,3)), 
                                errors_, error_type)

    # Formatting figures
    figure_formatting(ax, transf, Y.reshape((-1,2,3)), preds.reshape((-1,2,3)))
    add_description_box(ax, errors_)
    add_colorbars(fig, ax, scatters, transf, error_type)
    axs = plt.gca()
    axs.set_aspect('equal', 'box')
    axs.axis('square')

    # Save figure    
    save_diag_figure(save_path, transf)

    return




def figure_formatting(ax, transf:str, Y:np.array, pred:np.array):
    """
    Sets up formatting and labels for axes

    Args: 
        ax      ():         axes of figure
        transf  (str):      type to plot (t: normalised or u: in original units)
        Y       (np.arr):   labels (for limit calculation), shape: (ntest, 2, 3)
        pred    (np.arr):   predictions (for limit calculation), shape (ntest, 2, 3)
    
    """
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": 12,
        })
    
    if transf == 'u':
        plotname = np.array([['$n_x$', '$n_y$', '$n_{xy}$'],
                            ['$m_x$', '$m_y$', '$m_{xy}$'],
                            ['$v_x$', '$v_y$', '$v_y$']])
        plotname_p = np.array([[r'$\tilde{n}_{x}$', r'$\tilde{n}_{y}$', r'$\tilde{n}_{xy}$'],
                            [r'$\tilde{m}_x$', r'$\tilde{m}_y$', r'$\tilde{m}_{xy}$'],
                            [r'$\tilde{v}_x$', r'$\tilde{v}_y$', r'$\tilde{v}_y$']])
        units = np.array([[r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$'],
                          [r'$\rm [Nmm/mm]$', r'$\rm [Nmm/mm]$', r'$\rm [Nmm/mm]$'],  
                          [r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$']])
    elif transf == 't':
        plotname = np.array([[r'$n_{x,norm}$', r'$n_{y,norm}$', r'$n_{xy,norm}$'],
                        [r'$m_{x,norm}$', r'$m_{y,norm}$', r'$m_{xy,norm}$'],
                        [r'$v_{x,norm}$', r'$v_{y,norm}$', r'$v_{y,norm}$']])
        plotname_p = np.array([[r'$\tilde{n}_{x,norm}$', r'$\tilde{n}_{y,norm}$', r'$\tilde{n}_{xy,norm}$'],
                        [r'$\tilde{m}_{x,norm}$', r'$\tilde{m}_{y,norm}$', r'$\tilde{m}_{xy,norm}$'],
                        [r'$\tilde{v}_{x,norm}$', r'$\tilde{v}_{y,norm}$', r'$\tilde{v}_{y,norm}$']])
        units = np.array([['$[-]$', '$[-]$', '$[-]$'],
                          ['$[-]$', '$[-]$', '$[-]$'],  
                          ['$[-]$', '$[-]$', '$[-]$']])

    for i in range(2):
        for j in range(3):
            ax[i,j].set_ylabel(plotname_p[i,j]+' '+ units[i,j])
            ax[i,j].set_xlabel(plotname[i,j]+' '+ units[i,j])
            ax[i,j].set_xlim([np.min([np.min(Y[:,i,j]), np.min(pred[:,i,j])]), 
                              np.max([np.max(Y[:,i,j]), np.max(pred[:,i,j])])])
            ax[i,j].set_ylim([np.min([np.min(Y[:,i,j]), np.min(pred[:,i,j])]), 
                              np.max([np.max(Y[:,i,j]), np.max(pred[:,i,j])])])
            ax[i,j].grid(True, which='major', color='#666666', linestyle='-')
    
    return 

def calculate_errors(Y, predictions, stats, transf, id = 'sig'):
    num_cols = 6
    num_cols_plt = 6

    ### Transform the units of the statistics which relate back to the train set to the units desired in the diagonal plot
    if transf == 't':
        mean_train_ = 0*np.ones((1,num_cols))
        q_5_train_ = -1.645*np.ones((1,num_cols))
        q_95_train_ = 1.645*np.ones((1,num_cols))
    elif transf == 'u':
        # the statistics are already in the units N, mm
        mean_train_ = stats['stats_y_train']['mean'][0:num_cols].reshape((1,num_cols))
        q_5_train_ = stats['stats_y_train']['q_5'][0:num_cols].reshape((1,num_cols))
        q_95_train_ = stats['stats_y_train']['q_95'][0:num_cols].reshape((1,num_cols))

    if id == 'De':
        raise UserWarning('This is an old version of the code that has not yet been checked.')
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
        raise UserWarning('This is an old version of the code that has not yet been checked.')
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

def plot_main_figure(ax, Y, pred, errors, error_type):
    """
    actual scatter plotting

    Args: 
        ax      ():         axes of figure
        Y       (np.arr):   labels, shape: (ntest, 2, 3)
        pred    (np.arr):   predictions, shape: (ntest, 2, 3)
        errors  (np.arr):   errors, shape: (ntest, 2, 3)
        error_type (str):   type of error to plot as colorscale

    """

    norms = [mcolors.Normalize(vmin=np.min(errors[error_type][:, i, :]),
                               vmax=np.max(errors[error_type][:, i, :]))
                                for i in range(2)]
    scatters = []

    for i in range(2):
         for j in range(3):
            scatter = ax[i,j].scatter(Y[:,i,j], pred[:,i,j], marker = 'o', s = 20, 
                                      c = errors[error_type][:, i,j], cmap = 'plasma', linestyle='None', alpha = 0.4, 
                                      norm = norms[i])
            scatters.append(scatter)

            # add diagonal line to show where points should be.
            ax[i,j].plot([np.min([np.min(Y[:,i,j]), np.min(pred[:,i,j])]), np.max([np.max(Y[:,i,j]), np.max(pred[:,i,j])])], 
                         [np.min([np.min(Y[:,i,j]), np.min(pred[:,i,j])]), np.max([np.max(Y[:,i,j]), np.max(pred[:,i,j])])],
                            color='white', linestyle='--', linewidth = 1)


    return scatters
            
def add_description_box(axs, errors):
    """
    Adds error description box to every subplot

    ax      ():         axes of plot
    errors  (np.arr):   errors, shape (ntest, 2, 3)
    """


    for i in range(2):
        for j in range(3):
            at = AnchoredText('$R^2 = ' + np.array2string(errors['r_squared2'][:,i,j][0], precision=2) + '$ \n' +
                            '$RMSE = ' + np.array2string(errors['rmse'][:,i,j][0], precision=2) + '$ \n' +
                            '$rRMSE = ' + np.array2string(errors['rrmse'][:,i,j][0]*100, precision=0) + '\% $ \n' +
                            '$||rRSE||_{\infty} = ' + np.array2string(errors['rrse_max'][:,i,j][0]*100, precision=0) + '\% $ \n'+
                            '$nRMSE = ' + np.array2string(errors['nrmse'][:,i,j][0]*100, precision=0) + '\% $  \n'+
                            '$||nRSE||_{\infty} = ' + np.array2string(errors['nrse_max'][:,i,j][0]*100, precision=0) + '\% $',
                            prop=dict(size=10), frameon=True,loc='upper left')
            at.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
            axs[i,j].add_artist(at)

    return

def add_colorbars(fig, ax, scatters, transf, error_type):
    """
    add colorbars.
    """
    # TODO: add "transf" as variable. RSE doesn't have any unit if trasnf = 't'

    for i in range(2):
        if error_type == 'rse': 
            if i == 0:
                name = 'RSE \: [N/mm]'
            if i == 1:
                name = 'RSE \: [Nmm/mm]'
        elif error_type == 'nrse':
           name = 'nRSE \: [\%]'
        cbar = fig.colorbar(scatters[i*3], ax=ax[i,:], orientation='vertical', label=f'Row {i+1} $'+name+'$')
        cbar.set_label('$'+name+'$')

    return

def save_diag_figure(save_path, transf, filename = 'multi_diag_scatter'):
    filename = 'diagonal_match_'+transf+'.png'
    save_location = os.path.join(save_path, filename)
    plt.savefig(save_location, dpi=100, bbox_inches='tight')
    print(f'Saved {filename} to {save_path}')

    return



