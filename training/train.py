# vb, 24.03.2026

import torch
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from lightning.pytorch import seed_everything
seed_everything(42)

from data_utils import *
from train_utils import *
from test_utils import *


SAVE_FOLDER = True
SWEEP = False
SOBOLEV = True
GEOM_SIZE = 0           # [t, rho_x, rho_y, CC]
PLOT_DATA = False


############################ 0 - Read data      ############################
# Note: Leave data in original units (i.e. N, mm)

# 0.0 Fetch data
# DATA_DIR / MODEL_DIR env vars let us override I/O locations on HPC.
# Defaults preserve the original Windows behaviour.
path_data = os.environ.get('DATA_DIR') or os.path.join('D:\\', 'VeraBalmer\\ShellSim3D')
_here = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get('MODEL_DIR') or os.path.join(_here, 'config')
LOGS_DIR = os.environ.get('LOGS_DIR') or os.path.join(_here, 'logs')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
print(f'[train] DATA_DIR = {path_data}')
print(f'[train] MODEL_DIR = {MODEL_DIR}')
print(f'[train] LOGS_DIR = {LOGS_DIR}')
features, labels_concat = get_data(path_data, SOBOLEV)


# 0.1 Add geometrical variability to features if desired [TODO]
features = add_geom_data(path_data, features, GEOM_SIZE)
    

############################ 1 Train-Eval-Test Split ############################

train_eval_test_data = split_data(features, labels_concat, test_size = 0.1, eval_size = 0.2)

plot_split_data(train_eval_test_data, plot = PLOT_DATA)

############################ 2 - Normalisation      ############################

stats = get_stats(train_eval_test_data)

norm_data = get_normalised_data(train_eval_test_data, stats, SOBOLEV)

plot_norm_data(norm_data, plot = PLOT_DATA)

torch_data = data_to_torch(norm_data)


############################ 3 - Hyperparams        ############################

from config_inp import inp, constant_inp
inp['Sobolev'] = SOBOLEV
constant_inp['Sobolev'] = SOBOLEV

# Optional env-var overrides (smoke tests / scaling sweeps). Defaults unchanged.
if os.environ.get('NUM_EPOCHS'):
    inp['num_epochs'] = int(os.environ['NUM_EPOCHS'])
    print(f"[train] NUM_EPOCHS override -> {inp['num_epochs']}")
if os.environ.get('BATCH_SIZE'):
    inp['batch_size'] = int(os.environ['BATCH_SIZE'])
    print(f"[train] BATCH_SIZE override -> {inp['batch_size']}")
if os.environ.get('HIDDEN_LAYERS'):
    # e.g. HIDDEN_LAYERS='[256, 256, 256]' -- parsed by config_inp consumers via ast.literal_eval.
    inp['hidden_layers'] = os.environ['HIDDEN_LAYERS']
    print(f"[train] HIDDEN_LAYERS override -> {inp['hidden_layers']}")

save_inp(inp, save_path = MODEL_DIR)
save_stats(stats, save_path = MODEL_DIR)
save_test_data(train_eval_test_data, save_path = MODEL_DIR)


############################ 4 - Train              ############################


training_wrapper(torch_data,  inp,
                save_path = MODEL_DIR,
                save_folder = SAVE_FOLDER, sweep = SWEEP)


############################ 5 - Test              ############################

test_data = {'X_test': train_eval_test_data['X_test'],
             'y_test': train_eval_test_data['y_test']}

test_NN_model(test_data, stats,
              save_path = LOGS_DIR, version = None)