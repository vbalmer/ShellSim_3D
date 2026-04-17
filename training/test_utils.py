# vb, 29.03.2026

import os
import glob
from pathlib import Path
import pickle

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.func import vmap
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
    plot_data_D = predict_D(test_model, inp, test_data, stats, sobolev=inp['Sobolev'])


    ## 3 - Create diagonal plots for stresses

    # 3a - Normalised data
    multiple_diagonal_plots(plot_data_sig['labels_norm'], plot_data_sig['pred_norm'], 
                            stats, 'rse', plot_path, transf = 't')

    # 3b - Transformed data (no unit transformation required)
    multiple_diagonal_plots(plot_data_sig['labels'], plot_data_sig['pred'], 
                            stats, 'rse', plot_path, transf = 'u')

    ## 4 - Create diagonal plots for stiffnesses

    # 4a - Normalised data
    multiple_diagonal_plots_D(plot_data_D['labels_norm'], plot_data_D['pred_norm'],
                              stats, 'rse', plot_path, transf = 't')

    # 4b - Transformed data (no unit transformation required)
    multiple_diagonal_plots_D(plot_data_D['labels'], plot_data_D['pred'],
                              stats, 'rse', plot_path, transf = 'u')



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
    pred = get_unnormalised_data(pred_norm.cpu().detach().numpy(), stats, sobolev)

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

    pred = transform_data(pred_norm, stats, forward = False, type = norm_type_y)


    return pred


def predict_D(test_model: FFNN, inp: dict, test_data: dict, stats: dict, sobolev:bool):
    """
    Predict values of stiffness based on trained model.

    Args:
        test_model  (FFNN): Trained model 
        inp         (dict): Hyperparameters
        test_data   (dict): Features for test data
        stats       (dict): Statistical values of training data set
        sobolev     (bool): True, if test_model was trained including sobolev losses.
    Returns: 
        plot_data   (dict): Containing normalised and original-unit predictions and labels. Shape: ((ntest, 6, 6))
    """

    if not sobolev:
        print('This model was not trained with Sobolev losses. Predictions may differ significantly from labels in dataset.')


    # 1 - normalise data & transfer to torch
    test_data_norm = get_normalised_data(test_data, stats, sobolev)
    test_data_norm_torch = data_to_torch(test_data_norm)

    # 2 - make prediction in normalised coordinates
    pred_norm = get_normalised_D_pred(inp, test_data_norm_torch, test_model)

    # 3 - make predictions in [N,mm] by reverting normalisation
    pred_ext = np.concatenate((np.zeros((pred_norm.shape[0], 6)), pred_norm.reshape((-1, 36))), axis=1)     # for the function below to work correctly need given shape.
    pred = get_unnormalised_data(pred_ext, stats, sobolev)


    # 4 - collect data into dataset
    plot_data = {
        'pred': pred[:,6:].reshape((-1,6,6)),
        'pred_norm': pred_norm,
        'labels': test_data['y_test'][:,6:].reshape((-1, 6, 6)),
        'labels_norm': test_data_norm['y_test_t'][:,6:].reshape((-1,6,6))
    }

    return plot_data

def get_normalised_D_pred(inp:dict, data: dict, test_model: FFNN, batch_size_fallback = 265, fallback = False):
    """
    Get normalised predictions for stiffness (from actual call to FFNN).

    Args: 
        inp         (dict): Required for batchsize
        data        (dict): test data in normalised, torch version (X_test_tt, y_test_tt)
        test_model  (FFNN): trained model in eval mode.
        test_batch_size(int): safeguard batch size in case standard batch size is too large.
        fallback    (bool): auxiliary variable for recursive function

    Returns: 
        pred_norm   (torch.Tensor): Normalised predictions for test dataset.
    """

    if inp['batch_size'] is None:
        batch_size_test = data['X_test_tt'].shape[0]
    else: 
        batch_size_test = inp['batch_size']
        
    test_dataset = TensorDataset(data['X_test_tt'].to(device))
    test_loader = DataLoader(test_dataset, batch_size = batch_size_test, shuffle = False)
    if fallback:
        test_loader = DataLoader(test_loader.dataset, batch_size=batch_size_fallback)
    

    jacobian = []
    model_cpu = test_model.to(torch.device('cpu'))
    for batch_no, (X_test_tt,) in enumerate(test_loader):
        X_test_tt = X_test_tt.cpu()
        try:
            J = vmap(torch.func.jacrev(model_cpu.forward), randomness='different')(X_test_tt)
            jacobian.append(J.detach().numpy())
        except (torch.cuda.OutOfMemoryError, MemoryError):
            print(f'OOM at batch {batch_no}, retrying with batch_size={batch_size_test}')
            return get_normalised_D_pred(inp, data, test_model, fallback=True)

        if batch_no % 20 == 0:
            print(f'Stiffness Prediction Batch {batch_no}/{len(test_loader)}')

    pred_norm = np.concatenate(jacobian, axis=0)


    return pred_norm

################################ auxiliary functions for plotting stresses ################################

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
    for ax_ in ax.ravel():
        ax_.set_aspect('equal', 'box')
        ax_.axis('square')

    # Save figure    
    save_diag_figure(save_path, transf, id = 'sig')

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
    if id == 'sig':
        num_cols = 6
        num_cols_plt = 6
    elif id == 'De-NLRC': 
        idx = 6
        num_cols_plt = 36

    ### Transform the units of the statistics which relate back to the train set to the units desired in the diagonal plot
    if transf == 't':
        # for transformed units.
        if id == 'sig' or id == 'eps':
            mean_train_ = 0*np.ones((1,num_cols))
            q_5_train_ = -1.645*np.ones((1,num_cols))
            q_95_train_ = 1.645*np.ones((1,num_cols))

            mean_train = mean_train_
            q_5_train = q_5_train_
            q_95_train = q_95_train_

        elif id == 'De-NLRC':
            mean_train_ = 0*np.ones((1,idx, idx))
            q_5_train_ = -1.645*np.ones((1,idx, idx))
            q_95_train_ = 1.645*np.ones((1,idx, idx))

            mean_train = mean_train_.reshape((-1, idx*idx))
            q_5_train = q_5_train_.reshape((-1, idx*idx))
            q_95_train = q_95_train_.reshape((-1, idx*idx))

    elif transf == 'u':
        # the statistics are already in the units N, mm
        if id == 'sig' or id == 'eps':
            mean_train_ = stats['stats_y_train']['mean'][0:num_cols].reshape((1,num_cols))
            q_5_train_ = stats['stats_y_train']['q_5'][0:num_cols].reshape((1,num_cols))
            q_95_train_ = stats['stats_y_train']['q_95'][0:num_cols].reshape((1,num_cols))
        
            mean_train = mean_train_
            q_5_train = q_5_train_
            q_95_train = q_95_train_

        if id == 'De-NLRC':
            # for nonlinear version of De (i.e. Dmb is not zero)
            # if predicting sigma with network and D with derivatives
            mean_train_ = stats['stats_y_train']['mean']
            q_5_train_ = stats['stats_y_train']['q_5']
            q_95_train_ = stats['stats_y_train']['q_95']

            mean_train = mean_train_[idx:].reshape((-1,idx*idx))
            q_5_train = q_5_train_[idx:].reshape((-1,idx*idx))
            q_95_train = q_95_train_[idx:].reshape((-1,idx*idx))


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

def plot_main_figure(ax, Y, pred, errors, error_type, norms_ = None):
    """
    actual scatter plotting

    Args: 
        ax      ():         axes of figure
        Y       (np.arr):   labels, shape: (ntest, nrows, ncols)
        pred    (np.arr):   predictions, shape: (ntest, nrows, ncols)
        errors  (np.arr):   errors, shape: (ntest, nrows, ncols)
        error_type (str):   type of error to plot as colorscale
        norms_    (list):   colors for colorbar in stiffness plot. Is none for sig-plot.

    """

    nrows = Y.shape[1]
    ncols = Y.shape[2]

    if norms_ is None:
        norms = [mcolors.Normalize(vmin=np.min(errors[error_type][:, i, :]),
                                vmax=np.max(errors[error_type][:, i, :]))
                                    for i in range(nrows)]
    else: 
        norms = norms_.copy()

    scatters = []

    for i in range(nrows):
         for j in range(ncols):
            scatter = ax[i,j].scatter(Y[:,i,j], pred[:,i,j], marker = 'o', s = 20, 
                                      c = errors[error_type][:, i,j], cmap = 'plasma', linestyle='None', alpha = 0.4, 
                                      norm = norms[i] if norms_ is None else norms[i // 3 * 2 + j // 3])
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
    errors  (np.arr):   errors, shape (ntest, nrows, ncols)
    """

    nrows = errors['r_squared2'].shape[1]
    ncols = errors['r_squared2'].shape[2]

    for i in range(nrows):
        for j in range(ncols):
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

    for i in range(2):
        if error_type == 'rse': 
            if i == 0:
                if transf == 't':
                    name = 'RSE \: [-]'
                elif transf == 'u':
                    name = 'RSE \: [N/mm]'
            if i == 1:
                if transf == 't':
                    name = 'RSE \: [-]'
                elif transf == 'u':
                    name = 'RSE \: [Nmm/mm]'
        elif error_type == 'nrse':
           name = 'nRSE \: [\%]'
        cbar = fig.colorbar(scatters[i*3], ax=ax[i,:], orientation='vertical', label=f'Row {i+1} $'+name+'$')
        cbar.set_label('$'+name+'$')

    return

def save_diag_figure(save_path, transf, filename = 'multi_diag_scatter', id = None):
    filename = 'diagonal_match_'+id+'_'+transf+'.png'
    save_location = os.path.join(save_path, filename)
    plt.savefig(save_location, dpi=100, bbox_inches='tight')
    print(f'Saved {filename} to {save_path}')

    return



################################ auxiliary functions for plotting stiffnesses ################################

def multiple_diagonal_plots_D(D_true: np.array, D_pred: np.array, stats: dict, error_type: str, save_path: str, transf: str):
    """
    Creates diagonal plots to show results of testing for stiffness

    Args: 
        D_true  (np.arr):   test labels (shape: ntest, 6, 6)
        D_pred  (np.arr):   test predictions (shape: ntest, 6, 6) 
        stats     (dict):   statistical values utilised in error calculation (only train values)
        error_type (str):   'rse', 'nrse', ...
        save_path  (str):   location where plot will be stored.
        transf    (dict):   't': plotting normalised data, 'u': plotting original units data

    Returns: 
        diagonal plots as png in save_path
    """

    # Calculate relevant errors
    errors = calculate_errors(D_true.reshape((-1, 36)), D_pred.reshape((-1,36)), 
                              stats, transf, id = 'De-NLRC')

    
    # Define figures
    fig, ax = plt.subplots(6, 6, figsize=[25, 25], dpi=100)
    fig.subplots_adjust(wspace=0.4)

    errors_ = {key: arr.reshape(-1, 6, 6) for key, arr in errors.items()}
    norms = add_colorbars_D(fig, ax, errors_, transf, error_type)

    # Plot figures
    scatters = plot_main_figure(ax, D_true, D_pred, errors_, error_type, norms)

    # Formatting figures
    figure_formatting_D(ax, transf, D_true, D_pred)
    add_description_box(ax, errors_)
    for ax_ in ax.ravel():
        ax_.set_aspect('equal', 'box')
        ax_.axis('square')

    # Save figure    
    save_diag_figure(save_path, transf, id = 'D')

    return



def figure_formatting_D(ax, transf: str, D_true:np.array, D_pred:np.array):
    """
    Sets up formatting and labels for axes

    Args: 
        ax      ():         axes of figure
        transf  (str):      type to plot (t: normalised or u: in original units)
        D_true  (np.arr):   labels (for limit calculation), shape: (ntest, 2, 3)
        D_pred  (np.arr):   predictions (for limit calculation), shape (ntest, 2, 3)
    
    """

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
                        ])
    plotname_p = np.array([[r'$\tilde{D}_{m,11}$', r'$\tilde{D}_{m,12}$', r'$\tilde{D}_{m,13}$', r'$\tilde{D}_{mb,11}$', r'$\tilde{D}_{mb,12}$', r'$\tilde{D}_{mb,13}$'],
                        [r'$\tilde{D}_{m,21}$', r'$\tilde{D}_{m,22}$', r'$\tilde{D}_{m,23}$', r'$\tilde{D}_{mb,21}$', r'$\tilde{D}_{mb,22}$', r'$\tilde{D}_{mb,23}$'],
                        [r'$\tilde{D}_{m,31}$', r'$\tilde{D}_{m,32}$', r'$\tilde{D}_{m,33}$', r'$\tilde{D}_{mb,31}$', r'$\tilde{D}_{mb,32}$', r'$\tilde{D}_{mb,33}$'],
                        [r'$\tilde{D}_{bm,11}$', r'$\tilde{D}_{bm,12}$', r'$\tilde{D}_{bm,13}$', r'$\tilde{D}_{b,11}$', r'$\tilde{D}_{b,12}$', r'$\tilde{D}_{b,13}$'],
                        [r'$\tilde{D}_{bm,21}$', r'$\tilde{D}_{bm,22}$', r'$\tilde{D}_{bm,23}$', r'$\tilde{D}_{b,21}$', r'$\tilde{D}_{b,22}$', r'$\tilde{D}_{b,23}$'],
                        [r'$\tilde{D}_{bm,31}$', r'$\tilde{D}_{bm,32}$', r'$\tilde{D}_{bm,33}$', r'$\tilde{D}_{b,31}$', r'$\tilde{D}_{b,32}$', r'$\tilde{D}_{b,33}$'],
                        ])

    if transf == 't':
        units = np.array([['$[-]$']*6,
                          ['$[-]$']*6,
                          ['$[-]$']*6,
                          ['$[-]$']*6,
                          ['$[-]$']*6,
                          ['$[-]$']*6,                          
                          ])
    elif transf =='u':
        units = np.array([[r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$'],
                          [r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$'],
                          [r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$'],
                          [r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$'],
                          [r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$'],
                          [r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$'],
                          ])

    
    for i in range(6):
        for j in range(6):
            ax[i,j].set_ylabel(plotname_p[i,j]+' '+ units[i,j], labelpad = 2)
            ax[i,j].set_xlabel(plotname[i,j]+' '+ units[i,j], labelpad = 2)
            ax[i,j].set_xlim([np.min([np.min(D_true[:,i,j]), np.min(D_pred[:,i,j])]), 
                              np.max([np.max(D_true[:,i,j]), np.max(D_pred[:,i,j])])])
            ax[i,j].set_ylim([np.min([np.min(D_true[:,i,j]), np.min(D_pred[:,i,j])]), 
                              np.max([np.max(D_true[:,i,j]), np.max(D_pred[:,i,j])])])
            ax[i,j].grid(True, which='major', color='#666666', linestyle='-')

    return

def add_colorbars_D(fig, axs, errors: dict, transf: str, error_type: str):
    """
    Create four colorbars corresponding to the 4 submatrices of D (Dm, Dmb, Dbm and Db)

    fig         ():     Figure
    axs         ():     Set of axes
    errors      (dict): Contains all calculated errors
    transf      (str):  Type of plot (t: transformed units, u: original units)
    error_type  (str):  e.g. 'rse' 
    
    """

    # Define blocks for which colorbars are visualised.
    block_positions = [(0, 0), (0, 3), (3, 0), (3, 3)]
    norms = []

    for bx, by in block_positions:
        block_errors = np.concatenate([
            errors[error_type][:, b0, b1].flatten()
            for b0 in range(bx, bx + 3) for b1 in range(by, by + 3)
            ])
        norms.append(mcolors.Normalize(vmin=np.min(block_errors), vmax=np.max(block_errors)))



    # Define 3x3 blocks for colorbars
    if transf == 'u':
        colorbar_labels = {"rse": ["RSE [N/mm]", "RSE [N]", "RSE [N]", "RSE [Nmm]"], "nrse":  ["nRSE [%]", "nRSE [%]", "nRSE [%]", "nRSE [%]"]}
    else: 
        colorbar_labels = {"rse": ["RSE [-]", "RSE [-]", "RSE [-]", "RSE [-]"], "nrse":  ["nRSE [%]", "nRSE [%]", "nRSE [%]", "nRSE [%]"]}
    colorbar_label = colorbar_labels.get(error_type, "Error Scale")

    for norm, (bx, by), label in zip(norms, block_positions, colorbar_label):
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axs[bx:bx+3, by:by+3], orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label(label, fontsize=12)

    return norms


