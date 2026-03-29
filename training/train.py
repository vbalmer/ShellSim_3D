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
TEST_ONLY = True        # if True: only carries out testing. Seeding allows for same train-test-split.
PLOT_DATA = False


############################ 0 - Read data      ############################
# Note: Leave data in original units (i.e. N, mm)

# 0.0 Fetch data
path_data = os.path.join('D:\\', 'VeraBalmer\\ShellSim3D')
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

save_inp(inp)


############################ 4 - Train              ############################

training_wrapper(torch_data,  inp, 
                 save_path = 'training\\config', 
                 save_folder = SAVE_FOLDER, sweep = SWEEP, test_only = TEST_ONLY)


############################ 5 - Test              ############################

test_data = {'X_test': train_eval_test_data['X_test'],
             'y_test': train_eval_test_data['y_test']}

test_NN_model(test_data, stats,
              save_path = 'training\\logs', version = 1)