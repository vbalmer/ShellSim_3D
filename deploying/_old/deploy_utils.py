import numpy as np
import os
import sys
from Main_vb import main_solver
import time
import pandas as pd
from dict_CC import dict_CC
import wandb


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout



def extend_material_parameters(mat_tot_dict_, mat_dict=dict_CC):
    # Add constant steel parameters
    fsy = 435               # [MPa]
    fsu = 470               # [MPa]
    Es = 205e3              # [MPa]
    Esh = 8e3               # [MPa]
    D = 16                  # [mm]

    mat_tot_dict_.update({'fsy': fsy, 'fsu': fsu, 'Es': Es, 'Esh': Esh, 'D': D})
    mat_tot_dict = mat_tot_dict_

    # Add additional concrete parameters
    index = int(np.where(mat_dict['CC'] == mat_tot_dict['CC'])[0])
    mat_tot_dict['E_1'] = mat_dict['Ec'][index]
    mat_tot_dict['tb0'] = mat_dict['tb0'][index]
    mat_tot_dict['tb1'] = mat_dict['tb1'][index]
    mat_tot_dict['ect'] = mat_dict['ect'][index]
    mat_tot_dict['ec0'] = mat_dict['ec0'][index]
    mat_tot_dict['fcp'] = mat_dict['fcp'][index]
    mat_tot_dict['fct'] = mat_dict['fct'][index]

    return mat_tot_dict


def run_deployment(mat_tot, conv_plt, simple, n_simple, NN_hybrid, path_collection, new_folder_path):
    t0 = time.time()
     

    # run the simulation
    if simple:
        mat_res = [dict() for x in range(n_simple)]  
        for i in range(int(n_simple)):
            # mat = mat_tot.loc[i,:]
            mat = mat_tot
            mat_res[i] = main_solver(mat,conv_plt, NN_hybrid, path_collection, new_folder_path)
            if i>0 and i%10 == 0:
                print('**********************************************************************')
                print('Data points upto row', i, 'are simulated')
                print('time required for first', i,'points:',time.time()-t0, 'secs') 
    else: 
        RuntimeError('Always use simple = True for deployment')

    if new_folder_path is None: 
        save_data(NN_hybrid, mat_res)

    return mat_res


def save_data(NN_hybrid, mat_res):
    # save the data: 
    if NN_hybrid['predict_sig'] and NN_hybrid['predict_D']: 
        fname = 'mat_res_NN.pkl'
    elif NN_hybrid['predict_sig']:
        fname = 'mat_res_NN_sig.pkl'
    elif NN_hybrid['predict_D']:
        fname = 'mat_res_NN_D.pkl'
    elif NN_hybrid['PERM'] is not None:
        fname = 'mat_res_norm_perm.pkl'
    elif 'PERM1' in NN_hybrid and NN_hybrid['PERM1'] is not None: 
        fname = 'mat_res_norm_perm1.pkl'
    else: 
        fname = 'mat_res_norm.pkl'

    mat_res_pd_NN = pd.DataFrame.from_dict(mat_res)
    mat_res_pd_NN.to_pickle(os.path.join('05_Deploying\\data_out',fname))

    return



def get_paths(vnum, epnum, vnumD=None, epnumD=None):
    path_train = '04_Training'

    data_path = {
            # "_": os.path.join(path_train,'new_data\\_simple_logs\\v_80'),
            "_I": os.path.join(path_train,'new_data\\_simple_logs\\v_'+vnum),
            "_II": os.path.join(path_train,'new_data\\_simple_logs\\v_'+vnum),
            "_III": os.path.join(path_train,'new_data\\_simple_logs\\v_'+vnum),
    }
    model_path = {
            # "D": os.path.join(path_train,'logs\\train_log\\version_83_main\\checkpoints\\best_model.ckpt'),
            # "m": os.path.join(path_train,'logs\\train_log\\version_128\\checkpoints\\best_model_m.ckpt'),
            # "b": os.path.join(path_train,'logs\\train_log\\version_128\\checkpoints\\best_model_b.ckpt'),
            # "s": os.path.join(path_train,'logs\\train_log\\version_128\\checkpoints\\best_model_s.ckpt'),
            #"sig": os.path.join(path_train, 'new_data\\_simple_logs\\v_80\\best_trained_model__5138.pt'),
            "sig_I": os.path.join(path_train, 'new_data\\_simple_logs\\v_'+vnum+'\\best_trained_model__'+epnum+'.pt'),
            "sig_II": os.path.join(path_train, 'new_data\\_simple_logs\\v_'+vnum+'\\best_trained_model__'+epnum+'.pt'),
            "sig_III": os.path.join(path_train, 'new_data\\_simple_logs\\v_'+vnum+'\\best_trained_model__'+epnum+'.pt'),
    }


    if vnumD is not None: 
        data_path["D"] = os.path.join(path_train,'new_data\\_simple_logs\\v_'+vnumD)
        model_path["D"] = os.path.join(path_train, 'new_data\\_simple_logs\\v_'+vnumD+'\\best_trained_model__'+epnumD+'.pt')


    path_collection = {
            "model": model_path,
            "data": data_path
    }

    return path_collection