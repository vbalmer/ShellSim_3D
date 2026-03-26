# vb, 25.03.2026


import pickle
import os
import wandb
import torch
import random
import numpy as np
import re
import glob
import shutil


from torch.utils.data import TensorDataset, DataLoader


from architectures import *
SEED = 42



########################################  wrapper functions ########################################

def main_train(data: dict, save_path: str, config = None, project_name = 'ShellSim3D_sweep', save_folder = False):
    """
    Main training function. 

    Args: 
        data            (dict):     training data (X_train, y_train)
        save_path       (dict):     location where to save the intermediate data
        config          (dict):     wandb config dict for sweeps, None in case only single training is carried out.
        project_name    (str):      wandb project name
        save_folder     (bool):     if True, creates a new folder to store data instead of overwriting existing data in save_path.

    """

    # logging input parameters for wandb
    with wandb.init(config=config, project=project_name):

        #__________________________________________________
        # 1 - Set hyperparameters, create instance of model
        #__________________________________________________
        from config_inp import constant_inp

        wandb.config.update(constant_inp)
        
        inp = wandb.config
        wandb.config.update({'activation': wandb.config['activation']}, allow_val_change=True)
        wandb.config.update({'loss_type': wandb.config['loss_type']}, allow_val_change=True)

        # create instance of model
        model_dict = model_instance(inp)

        # print model architecture
        model_print(model_dict)

        #________________________________________________________
        # 2 - Train and evaluate model, save best evaluated model
        #________________________________________________________

        model = simple_train(inp, model_dict, data, save_path)
        torch.save(model.state_dict(), os.path.join(save_path, 'last_trained_model.pt'))
        model.eval()

        # Copy model and data to separate folder.
        filenames = ['inp.pkl', 'last_trained_model.pt', 'best_trained_model.pt']

        if save_folder:
            src_folder = os.path.join(os.getcwd(), 'training\\config')
            base_dest_folder = os.path.join(os.getcwd(), 'training\\logs')
            copy_files_with_incremented_version(src_folder, base_dest_folder, filenames)

    return inp

def training_wrapper(data:dict, inp: dict, save_path:str, save_folder:str, sweep: bool) -> None:
    """
    Wrapper around main train function, depending on whether to include sweep or not

    Args: 
        data        (dict): Containing X_train, y_train in torch, normalised version
        inp         (dict): Hyperparameter dict
        save_path    (str): location where intermediate data is saved
        save_folder (bool): If true, save resulting trained models in folder.
        sweep       (bool): If true: carries out hyperparameter sweep

    Returns: 
        Trained model saved in save_path.
    
    """
    if not sweep:
        #_______________________________________
        # Call function to train without sweep
        # _______________________________________
        inp_ = inp
        inp = main_train(data, save_path, config = inp_, project_name = 'ShellSim3D', save_folder = save_folder)

    elif sweep:
        #________________________________________
        # Call function to train with sweep
        # ______________________________________
        raise UserWarning('This part of the code has not yet been debugged in the new version.')
    
        # Define sweep configuration
        from config_inp import sweep_config

        # start sweep
        sweep_id = wandb.sweep(sweep = sweep_config, project='ShellSim3D_sweep')
        wandb.agent(sweep_id, 
                    function= lambda: main_train(data = data, save_path = save_path),
                    count = 1)
        
    return


######################################## training setup functions ########################################


def model_instance(inp:dict):
    """
    Creates instance of the model. 
    
    Args:
        inp         (dict):     Includes all hyperparameters
        save_path    (str):     Location where to store intermediate results
    Returns:
        model_dict  (dict):     contains instance of requested model

    """

    model_dict = {}
    model = FFNN(inp)
    # could define some other scenarios here in case I want to use other model architectures - see old script.

    model_dict['standard'] = model


    return model_dict

def simple_train(inp:dict, model_dict:dict, data:dict, save_path:str) -> FFNN:
    """
    Setting up optimiser and model, as well as loaders depending on batch size.
    
    Args:
        inp         (dict):     Hyperparameter configuration
        model_dict  (dict):     instance of model architecture
        data        (dict):     Normalised training and evaluation data (X and y)
        save_path   (str):      Location where to store intermediate results

    Returns:
        model       (FFNN):     Trained model (depending on architecture can be different type)

    """
    
    set_torch_params()

    model = model_dict['standard']

    optimizer = optimizer_setup(inp, data, model)
    
    scheduler = scheduler_setup(inp, optimizer)

    lossFn = loss_setup(inp)
    
    model.to(device)

    MAXEPOCHS = inp['num_epochs']

    if inp['batch_size'] is None:
        batch_size_train = data['X_train_tt'].shape[0]
        batch_size_eval = data['X_eval_tt'].shape[0]
    else: 
        batch_size_train = inp['batch_size']
        batch_size_eval = inp['batch_size']

    train_dataset = TensorDataset(data['X_train_tt'], data['y_train_tt'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    eval_dataset = TensorDataset(data['X_eval_tt'], data['y_eval_tt'])
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size_eval, shuffle = True)
    model = simple_train_aux_(train_loader, eval_loader, model, optimizer, scheduler, lossFn, MAXEPOCHS, save_path, inp)

    return model


def simple_train_aux_(train_loader, eval_loader, model, optimizer, scheduler, lossFn, MAXEPOCHS, save_path, inp):
    """
    Actual training loop.
    
    """
    best_val_loss = float('inf')
    for epoch in range(MAXEPOCHS):
        model.train()

        batch = 0
        epoch_loss = 0
        for X_train_batch, y_train_batch in train_loader:
            X_torch, y_torch = X_train_batch.to(device), y_train_batch.to(device)

            def closure(backward = True):
                optimizer.zero_grad()

                if inp['Sobolev']:
                    custom_crit = CustomLosses(inp, model)
                    loss, loss_logs = custom_crit.Sobolev_CustomLoss(X_torch, y_torch)
                    # loss, loss_logs = custom_crit.comp_SobolevCustom(X_torch, sig_torch)
                else:
                    pred = model(X_torch)
                    loss = lossFn(pred, y_torch[:,0:inp['out_size']])
                    loss_logs = {'train_loss': loss}

                if backward:
                    loss.backward()

                return loss, loss_logs
    
            # Take steps
            optimizer.zero_grad()
            loss, loss_logs = optimizer.step(closure)

            epoch_loss  += loss.item()

            if inp['lr_scheduler'] == 'standard':
                scheduler.step()
            elif inp['lr_scheduler'] == 'plateau':
                pass

            # Logs
            wandb.log(loss_logs)
            wandb.log({'Learning Rate': scheduler.get_last_lr()[0]})
            wandb.log({'epoch': epoch})
            batch = batch+1

            if epoch % 100 == 0 or batch % int(len(train_loader)/2) == 0:
                print(f"Epoch [{epoch}/{MAXEPOCHS}] and Batch [{batch}/{len(train_loader)}] with LR {scheduler.get_last_lr()[0]:.2e}: \tLoss: {loss.item():.4e}")
            if batch == int(len(train_loader))-1: 
                avg_train_loss = epoch_loss / batch
                wandb.log({'Average train loss per epoch': avg_train_loss})


        model, best_val_loss = simple_eval(inp, model, eval_loader, train_loader, lossFn, scheduler, epoch, MAXEPOCHS, best_val_loss, save_path)


    return model

def simple_eval(inp, model, eval_loader, train_loader, lossFn, scheduler, epoch, MAXEPOCHS, best_val_loss, save_path):

    epoch_val_loss = 0
    val_batch = 0

    for X_eval_batch, y_eval_batch in eval_loader:
        X_eval, y_eval = X_eval_batch.to(device), y_eval_batch.to(device)
        model.eval()
        with torch.no_grad():
            sig_torch_val = y_eval
            pred_val = model(X_eval)
            
            if not inp['Sobolev']:
                val_loss = lossFn(pred_val, sig_torch_val[:,0:inp['out_size']])
            elif inp['Sobolev']: 
                custom_crit = CustomLosses(inp, model)
                val_loss, _ = custom_crit.Sobolev_CustomLoss(X_eval, sig_torch_val)
    
            wandb.log({"val_loss": val_loss})
            epoch_val_loss  += val_loss.item()
            val_batch += 1

            if epoch % 100 == 0 or val_batch % int(len(eval_loader)/2) == 0:
                print(f"Epoch [{epoch}/{MAXEPOCHS}] and Batch [{val_batch}/{len(eval_loader)}]: \tValidation Loss: {val_loss.item():.4e}")

        if val_batch == int(len(train_loader))-1: 
            avg_val_loss = epoch_val_loss / val_batch
            wandb.log({'Average evaluation loss per epoch': avg_val_loss})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wandb.log({"best_val_loss": best_val_loss})
            save_best_model(model, epoch, val_loss, save_dir=save_path)

        if inp['lr_scheduler'] == 'standard':
            pass
        elif inp['lr_scheduler'] == 'plateau':
            scheduler.step(val_loss)

    return model, best_val_loss

def optimizer_setup(inp:dict, data:dict, model: FFNN) -> Adam_LBFGS:
    if inp['batch_size'] is None: 
        no_switch_step = int(inp['switch_step_percentage']*inp['num_epochs'])
    else: 
        no_switch_step = int(inp['switch_step_percentage']*inp['num_epochs']*int(data['X_train_tt'].shape[0]/inp['batch_size']))
    optimizer = Adam_LBFGS(model.parameters(), no_switch_step)
    return optimizer

def scheduler_setup(inp:dict, optimizer:Adam_LBFGS):
    if inp['lr_scheduler'] == 'standard':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer.adam, step_size=100, gamma=0.99)
    elif inp['lr_scheduler'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.adam, 'min')

    return scheduler

def loss_setup(inp:dict):

    loss_mapping = {
    'MSELoss': nn.MSELoss,
    'HuberLoss': nn.HuberLoss,
    'MSLELoss': MSLELoss,
    'wMSELoss': wMSELoss,
    'RMSELoss': RMSELoss
    }

    lossFn = loss_mapping[inp['loss_type']]()

    return lossFn

def save_best_model(model, epoch, val_loss, save_dir):
    # save current model
    os.makedirs(save_dir, exist_ok=True)
    model_filename =f'best_trained_model__{epoch}.pt'
    torch.save(model.state_dict(), os.path.join(save_dir, model_filename))
    if epoch > 200:
        print(f'Best model saved at epoch {epoch} with validation loss: {val_loss:.4f}')
    
    # remove old models
    model_pattern = re.compile(fr"best_trained_model__(\d+)\.pt")
    
    for filename in os.listdir(save_dir):
        match = model_pattern.match(filename)
        if match: 
            epoch_ = int(match.group(1))
            if epoch_ < epoch:
                file_path = os.path.join(save_dir, filename)
                os.remove(file_path)
    return



######################################## auxiliary functions ########################################


def copy_files_with_incremented_version(src_folder:str, base_dest_folder:str, files_to_copy: str):
    """
    Copies files from src_folder to a new folder in base_dest_folder with an incremented version number.
    """
    # Get the latest version number in the destination folder and increment it
    latest_version = get_latest_version_folder(base_dest_folder)
    new_version = latest_version + 1
    new_folder_name = f"v_{new_version}"
    new_folder_path = os.path.join(base_dest_folder, new_folder_name)
    
    # Create the new versioned folder
    os.makedirs(new_folder_path, exist_ok=True)
    
    # Copy files from the source folder to the new versioned folder
    for file_name in files_to_copy:
        src_path = os.path.join(src_folder, file_name)
        dest_path = os.path.join(new_folder_path, file_name)   

        if file_name == 'best_trained_model.pt':
            model_file = glob.glob(os.path.join(src_folder, 'best_trained_model_*.pt'))
            for i in range(len(model_file)):
                file_name_new = os.path.basename(model_file[i])
                src_path_new = os.path.join(src_folder, file_name_new)
                dest_path_new = os.path.join(new_folder_path, file_name_new)
                if os.path.exists(src_path_new):
                    shutil.copy2(src_path_new, dest_path_new)
                    os.remove(src_path_new)
                else: 
                    print(f"{file_name} does not exist in the source folder.")

        else: 
            if os.path.exists(src_path):
                shutil.copy2(src_path, dest_path)
            else:
                print(f"{file_name} does not exist in the source folder.")

    
    print(f"Files copied to {new_folder_path}")
    return


def get_latest_version_folder(base_folder):
        """
        Finds the latest version folder in the base folder with the format 'v_num'.
        """
        version_folders = [f for f in os.listdir(base_folder) if re.match(r'v_\d+', f)]
        version_numbers = [int(re.search(r'v_(\d+)', folder).group(1)) for folder in version_folders]
        return max(version_numbers) if version_numbers else 0

def model_print(model_dict:dict) -> None:
    '''
    Prints model architecture for reference in wandb
    '''
    model = model_dict['standard']
    print(model)
    return

def save_inp(inp: dict, save_path = 'training\\config'):
    """
    saving inp file for use in testing
    """

    with open(os.path.join(save_path, 'inp.pkl'), 'wb') as fp:
            pickle.dump(inp, fp)
    return

def set_torch_params():
    """
    Set torch parameters and seed.
    """

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return