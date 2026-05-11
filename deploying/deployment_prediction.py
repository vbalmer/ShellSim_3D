# NN prediction for deployment

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.test_utils import predict_D, predict_sig, get_inp_from_folder, test_model_instance
from training.single_element_utils import get_stats_from_folder


def make_NN_prediction(input_j:np.array, predict:str, model_path: str):
    """
    Make prediction for deployment with NN. Ammends the input_j with y_test dummy variable and returns desired prediction. 
    Relies on same functions as testing /single-element-testing of NN.

    Args:
        input_j     (np.arr)
        predict     (str)           either sig or D (depending on what you want to predict). Function is not made to predict both at once but could be ammended for that.
        model_path  ([str, int])    as in the definition in single_element_test ([model_path, model_version], e.g. ['training\\logs', 33])

    Returns: 
        sig_D_NN    (dict)          dict containing sig or De as array, where always one of the two is "None".
    """
    
    prediction_data = {
            'X_test': input_j,
            'y_test': np.zeros_like(input_j)        # dummy variable (empty).
        }


    inp = get_inp_from_folder(model_path[0], model_path[1])
    test_model = test_model_instance(inp, model_path[0], model_path[1])
    stats = get_stats_from_folder(model_path[0], model_path[1])

    if predict == 'sig':
        sig_NN = predict_sig(test_model, inp, prediction_data, stats, sobolev = inp['Sobolev'])
        De_NN = {'pred': None}
    elif predict == 'D':
        sig_NN = {'pred': None}
        De_NN = predict_D(test_model, inp, prediction_data, stats, sobolev = inp['Sobolev'])

    sig_D_NN = {
                'sig': sig_NN['pred'],
                'D': De_NN['pred'],
            }     

    return sig_D_NN


def add_zerovals_stats(mat_data_stats, dim, small_value = 1e-20):
    '''
    adds zero-values to statistics of NNs that are not the desired shape required for predicting / transforming data
    mat_data_stats  (dict)      if ONEDIM: contains only 1+3 values in x and 1+1 values in y
                                if TWODIM: contains only 3+3 values in x and 3+9 values in y
    dim             (str)       'ONEDIM_x', 'ONEDIM_y', 'TWODIM'
    small_value     (float)     small value that represents zero
    '''

    if dim == 'ONEDIM_x':
        stats_new = {}
        for key in ['stats_y_train', 'stats_y_test']:
            stats_new[key] = {}
            for subkey in mat_data_stats[key].keys():
                a = mat_data_stats[key][subkey][:1]
                b = mat_data_stats[key][subkey][1:]
                stats_new[key][subkey] = np.concatenate((a, small_value*np.ones(7,), b, small_value*np.ones(63,)), axis = 0)
        for key in ['stats_X_train', 'stats_X_test']:
            stats_new[key] = {}
            for subkey in mat_data_stats[key].keys():
                a = mat_data_stats[key][subkey][:1]
                b = mat_data_stats[key][subkey][1:]
                stats_new[key][subkey] = np.concatenate((a, small_value*np.ones(7,), b), axis = 0)
        stats_new
    elif dim == 'ONEDIM_y':
        stats_new = {}
        for key in ['stats_y_train', 'stats_y_test']:
            stats_new[key] = {}
            for subkey in mat_data_stats[key].keys():
                a = mat_data_stats[key][subkey][:1]
                b = mat_data_stats[key][subkey][1:]
                stats_new[key][subkey] = np.concatenate((small_value*np.ones(1,), a, small_value*np.ones(6,), small_value*np.ones(9,), b, small_value*np.ones(54,)), axis = 0)
        for key in ['stats_X_train', 'stats_X_test']:
            stats_new[key] = {}
            for subkey in mat_data_stats[key].keys():
                a = mat_data_stats[key][subkey][:1]
                b = mat_data_stats[key][subkey][1:]
                stats_new[key][subkey] = np.concatenate((small_value*np.ones(1,), a, small_value*np.ones(6,), b), axis = 0)
        stats_new
    elif dim == 'TWODIM':
        stats_new = {}
        for key in ['stats_y_train', 'stats_y_test']:
            stats_new[key] = {}
            for subkey in mat_data_stats[key].keys():
                a = mat_data_stats[key][subkey][:3]
                b = mat_data_stats[key][subkey][3:]
                stats_new[key][subkey] = np.concatenate((a, small_value*np.ones(5,), 
                                                         b[:3], small_value*np.ones(5,),
                                                         b[3:6], small_value*np.ones(5,),
                                                         b[6:], small_value*np.ones(40+5,),), axis = 0)
        for key in ['stats_X_train', 'stats_X_test']:
            stats_new[key] = {}
            for subkey in mat_data_stats[key].keys():
                a = mat_data_stats[key][subkey][:3]
                b = mat_data_stats[key][subkey][3:]
                stats_new[key][subkey] = np.concatenate((a, small_value*np.ones(5,), b), axis = 0)
        stats_new
    elif dim == 'THREEDIM':
        raise UserWarning('This has not yet been implemented for the THREEDIM case.')
        pass

    else: 
        raise UserWarning('Please do not use this function if model_dim = ALL_DIM.')

    return stats_new