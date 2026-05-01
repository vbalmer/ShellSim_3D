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
import math


from torch.utils.data import TensorDataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(os.getpid())


from architectures import *
from test_utils import test_NN_model
SEED = 42



########################################  wrapper functions ########################################

def main_train(data: dict, save_path: str, config = None, project_name = 'ShellSim3D_sweep', save_folder = False, streaming = False):
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

        model = simple_train(inp, model_dict, data, save_path, streaming=streaming)
        torch.save(model.state_dict(), os.path.join(save_path, 'last_trained_model.pt'))
        model.eval()

        # Copy model and data to separate folder.
        filenames = ['inp.pkl', 'stats.pkl', 'last_trained_model.pt', 'best_trained_model.pt', 'test_data.pkl']

        if save_folder:
            # Use save_path (set by train.py from MODEL_DIR / LOGS_DIR env vars on HPC)
            # rather than cwd-relative Windows paths, so this also works on Linux.
            src_folder = save_path
            base_dest_folder = os.environ.get('LOGS_DIR') or os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'logs'
            )
            os.makedirs(base_dest_folder, exist_ok=True)
            copy_files_with_incremented_version(src_folder, base_dest_folder, filenames)

    return inp

def setup_dirs() -> tuple:
    """Resolve DATA_DIR / MODEL_DIR / LOGS_DIR from env vars or local defaults."""
    path_data = os.environ.get('DATA_DIR') or os.path.join('D:\\', 'VeraBalmer\\ShellSim3D')
    _here     = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.environ.get('MODEL_DIR') or os.path.join(_here, 'config')
    LOGS_DIR  = os.environ.get('LOGS_DIR')  or os.path.join(_here, 'logs')
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,  exist_ok=True)
    print(f'[train] DATA_DIR  = {path_data}')
    print(f'[train] MODEL_DIR = {MODEL_DIR}')
    print(f'[train] LOGS_DIR  = {LOGS_DIR}')
    return path_data, MODEL_DIR, LOGS_DIR


def setup_hyperparams(sobolev: bool) -> dict:
    """Load config and apply any env-var overrides (NUM_EPOCHS, BATCH_SIZE, HIDDEN_LAYERS)."""
    from config_inp import inp, constant_inp
    inp['Sobolev']          = sobolev
    constant_inp['Sobolev'] = sobolev
    if os.environ.get('NUM_EPOCHS'):
        inp['num_epochs']    = int(os.environ['NUM_EPOCHS'])
        print(f'[train] NUM_EPOCHS override    -> {inp["num_epochs"]}')
    if os.environ.get('BATCH_SIZE'):
        inp['batch_size']    = int(os.environ['BATCH_SIZE'])
        print(f'[train] BATCH_SIZE override    -> {inp["batch_size"]}')
    if os.environ.get('HIDDEN_LAYERS'):
        inp['hidden_layers'] = os.environ['HIDDEN_LAYERS']
        print(f'[train] HIDDEN_LAYERS override -> {inp["hidden_layers"]}')
    return inp


def resolve_streaming() -> bool:
    """Return the STREAMING flag, with STREAMING env-var taking precedence."""
    if os.environ.get('STREAMING'):
        val = os.environ['STREAMING'].lower() in ('1', 'true', 'yes')
        print(f'[train] STREAMING override     -> {val}')
        return val
    return True   # default: streaming on


def run_streaming_pipeline(path_data: str, MODEL_DIR: str, LOGS_DIR: str,
                            inp: dict, sobolev: bool,
                            save_folder: bool, sweep: bool) -> None:
    """
    Full training pipeline for datasets too large to fit in RAM.
    Mirrors the section structure of the in-memory path in train.py.
    """
    from data_utils import (get_dataset_size, get_streaming_splits,
                            compute_stats_full_dataset, load_test_sample,
                            HDF5StreamingDataset)

    #### 0 - Read data (meta only — no arrays loaded) ####
    n_total = get_dataset_size(path_data)
    print(f'[streaming] Total samples: {n_total / 1e9:.3f} B')

    #### 1 - Train-Eval-Test Split ####
    idx_train, idx_eval, idx_test = get_streaming_splits(n_total)

    #### 2 - Normalisation (full-dataset pass for exact mean/std) ####
    stats = compute_stats_full_dataset(path_data, sobolev)

    #### 3 - Save config + test sample ####
    save_inp(inp,   save_path=MODEL_DIR)
    save_stats(stats, save_path=MODEL_DIR)
    test_data = load_test_sample(path_data, sobolev, idx_test)
    save_test_data(test_data, save_path=MODEL_DIR)

    #### 4 - Train ####
    stream_data = {
        'train': HDF5StreamingDataset(path_data, stats, sobolev, idx_train, shuffle=True),
        'eval':  HDF5StreamingDataset(path_data, stats, sobolev, idx_eval,  shuffle=False),
    }
    training_wrapper(stream_data, inp,
                     save_path=MODEL_DIR,
                     save_folder=save_folder, sweep=sweep,
                     streaming=True)

    #### 5 - Test ####
    test_NN_model(test_data, stats, save_path=LOGS_DIR, version=None)


def training_wrapper(data:dict, inp: dict, save_path:str, save_folder:str, sweep: bool, streaming: bool = False) -> None:
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
        inp = main_train(data, save_path, config = inp_, project_name = 'ShellSim3D', save_folder = save_folder, streaming = streaming)

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

def simple_train(inp:dict, model_dict:dict, data:dict, save_path:str, streaming:bool = False) -> FFNN:
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

    optimizer = optimizer_setup(inp, data, model, streaming=streaming)

    scheduler = scheduler_setup(inp, optimizer)

    lossFn = loss_setup(inp)

    model.to(device)

    MAXEPOCHS = inp['num_epochs']

    if streaming:
        # data is {'train': HDF5StreamingDataset, 'eval': HDF5StreamingDataset}
        if inp['batch_size'] is None:
            raise ValueError('batch_size cannot be None in streaming mode.')
        batch_size_train = inp['batch_size']
        batch_size_eval  = inp['batch_size']
        # Windows uses 'spawn' for multiprocessing: worker processes re-import
        # train.py from scratch, which re-runs the whole script and crashes.
        # num_workers=0 (single-process) avoids this on Windows; Linux uses
        # fork so multi-worker loading works fine there.
        _nw_train = 0 if os.name == 'nt' else 4
        _nw_eval  = 0 if os.name == 'nt' else 2
        train_loader = DataLoader(data['train'], batch_size=batch_size_train,
                                  num_workers=_nw_train, pin_memory=True)
        eval_loader  = DataLoader(data['eval'],  batch_size=batch_size_eval,
                                  num_workers=_nw_eval, pin_memory=True)
        # Cap eval batches per epoch so validation doesn't dominate wall time.
        # Default: evaluate on at most 500 batches (= 500 * batch_size samples).
        max_eval_batches = 500
    else:
        if inp['batch_size'] is None:
            batch_size_train = data['X_train_tt'].shape[0]
            batch_size_eval  = data['X_eval_tt'].shape[0]
        else:
            batch_size_train = inp['batch_size']
            batch_size_eval  = inp['batch_size']
        train_dataset = TensorDataset(data['X_train_tt'], data['y_train_tt'])
        train_loader  = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
        eval_dataset  = TensorDataset(data['X_eval_tt'], data['y_eval_tt'])
        eval_loader   = DataLoader(eval_dataset,  batch_size=batch_size_eval,  shuffle=True)
        max_eval_batches = None     # no cap for in-memory eval

    model = simple_train_aux_(train_loader, eval_loader, model, optimizer, scheduler,
                              lossFn, MAXEPOCHS, save_path, inp,
                              max_eval_batches=max_eval_batches)

    return model


def simple_train_aux_(train_loader, eval_loader, model, optimizer, scheduler, lossFn, MAXEPOCHS, save_path, inp, max_eval_batches=None):
    """
    Actual training loop.

    Args:
        max_eval_batches (int | None): If set, evaluation is capped at this many
            batches per epoch. Useful for streaming datasets where the full eval
            split is too large to iterate every epoch.
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


        model, best_val_loss = simple_eval(inp, model, eval_loader, train_loader, lossFn, scheduler, epoch, MAXEPOCHS, best_val_loss, save_path, max_eval_batches=max_eval_batches)


    return model

def simple_eval(inp, model, eval_loader, train_loader, lossFn, scheduler, epoch, MAXEPOCHS, best_val_loss, save_path, max_eval_batches=None):

    epoch_val_loss = 0
    val_batch = 0

    for X_eval_batch, y_eval_batch in eval_loader:
        if max_eval_batches is not None and val_batch >= max_eval_batches:
            break
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

def optimizer_setup(inp:dict, data:dict, model: FFNN, streaming: bool = False) -> Adam_LBFGS:
    if inp['switch_step_percentage'] == 1:
        # Never switch to LBFGS — stay on Adam for the entire training run.
        no_switch_step = float('inf')
    elif inp['batch_size'] is None:
        no_switch_step = int(inp['switch_step_percentage']*inp['num_epochs'])
    else:
        n_train = len(data['train']) if streaming else data['X_train_tt'].shape[0]
        no_switch_step = int(inp['switch_step_percentage']*inp['num_epochs']*int(n_train/inp['batch_size']))
    optimizer = Adam_LBFGS(inp, model.parameters(), no_switch_step)
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
    'RMSELoss': RMSELoss,
    'RelMSELoss': RelMSELoss,
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

def save_inp(inp: dict, save_path = os.path.join('training', 'config')):
    """
    saving inp file for use in testing / later inference
    """

    with open(os.path.join(save_path, 'inp.pkl'), 'wb') as fp:
            pickle.dump(inp, fp)
    return

def save_stats(stats:dict, save_path = os.path.join('training', 'config')):
    """
    saving stats file for use in testing / later inference 
    """

    with open(os.path.join(save_path, 'stats.pkl'), 'wb') as fp:
            pickle.dump(stats, fp)

    return

def save_test_data(train_eval_test_data, save_path = os.path.join('training', 'config')):
    """
    Save test data in config folder.

    Args: 
        train_eval_test_data    (dict): dict containing train, eval and test data
        save_path               (str):  location where to save the data    

    Returns:
        saves only test data in config folder
    
    """
    test_data = {'X_test': train_eval_test_data['X_test'],
                'y_test': train_eval_test_data['y_test']}
    with open(os.path.join(save_path, 'test_data.pkl'), 'wb') as fp:
            pickle.dump(test_data, fp)

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