# vb, 25.03.2026


import pickle
import os
import sys
import wandb


import torch


########################################  wrapper functions ########################################

def main_train(data: dict, config = None, project_name = 'ShellSim3D_sweep', save_folder = False):
    """
    Main training function. 

    Args: 
        data            (dict):     training data (X_train, y_train)
        config          (dict):     wandb config dict for sweeps, None in case only single training is carried out.
        project_name    (str):      wandb project name
        save_folder     (str):      folder, where trained model should be stored.

    """

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
        model_dict, inp_dict = model_instance(inp, inp_dict, save_path)

        # print model architecture
        model_print(model_dict, inp)

        #________________________________________________________
        # 4b - Train and evaluate model, save best evaluated model
        #________________________________________________________

        if inp['simple_m']:
        # train with simple pytorch function from train_vb.py
            if not inp['MoE']:
                model = simple_train(inp, model_dict, data, save_path)
                torch.save(model.state_dict(), os.path.join(save_path, 'last_trained_model.pt'))
                model.eval()
            elif inp['MoE']:
                # Training a MoE directly without splitting data.
                model = simple_train(inp, model_dict, data, save_path)
                torch.save(model.state_dict(), os.path.join(save_path, 'last_trained_model.pt'))
                model.eval()

        # Copy model and data to separate folder.
        filenames = ['inp.pkl', 'mat_data_np_TrainEvalTest.pkl', 'mat_data_stats.pkl', 'mat_data_TrainEvalTest.pkl',
                    'last_trained_model.pt', 'best_trained_model.pt']

        if save_folder:
            src_folder = os.path.join(os.getcwd(), '04_Training\\new_data')
            base_dest_folder = os.path.join(os.getcwd(), '04_Training\\new_data\\_simple_logs')
            copy_files_with_incremented_version(src_folder, base_dest_folder, filenames)

    return inp



def training_wrapper(data, inp: dict, save_folder:str, sweep: bool) -> None:
    """
    Wrapper around main train function, depending on whether to include sweep or not

    Args: 
        data        (dict): Containing X_train, y_train in torch, normalised version
        inp         (dict): Hyperparameter dict
        save_folder (bool): If true, save resulting trained models in folder.

    Returns: 
        None.
    
    """
    if not sweep:
        #_______________________________________
        # 5a - Call function to train without sweep
        # _______________________________________
        inp_ = inp
        # print(inp_)
        inp = main_train(data, config = inp_, project_name = 'ShellSim3D', save_folder = save_folder)

    elif sweep:
        raise UserWarning('This part of the code has not yet been debugged in the new version.')
        #________________________________________
        # 5b - Call function to train with sweep
        # ______________________________________
        # Define sweep configuration
        from config_inp import sweep_config

        # start sweep
        sweep_id = wandb.sweep(sweep = sweep_config, project='ShellSim3D_sweep')
        wandb.agent(sweep_id, 
                    function=main_train,
                    count = 1)
        
    return


######################################## training setup functions ########################################


def model_instance():
    # TODO!
    return



def simple_train():
    # TODO!
    return


######################################## auxiliary functions ########################################


def copy_files_with_incremented_version():
    # TODO!
    return


def model_print():
    # TODO!
    return



def save_inp(inp: dict, save_path = 'training\\config'):
    """
    saving inp file for use in testing
    """

    with open(os.path.join(save_path, 'inp.pkl'), 'wb') as fp:
            pickle.dump(inp, fp)
    return
