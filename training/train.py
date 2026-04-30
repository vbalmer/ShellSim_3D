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
SWEEP       = False
SOBOLEV     = True
GEOM_SIZE   = 0           # [t, rho_x, rho_y, CC]
PLOT_DATA   = False
# Set True when dataset is too large to fit in RAM (> ~50 GB).
# All data stays on disk; only one chunk lives in memory at a time.
STREAMING   = True


path_data, MODEL_DIR, LOGS_DIR = setup_dirs()
inp = setup_hyperparams(SOBOLEV)

if STREAMING:
    run_streaming_pipeline(path_data, MODEL_DIR, LOGS_DIR,
                           inp, SOBOLEV, SAVE_FOLDER, SWEEP)

else:
    ############################ 0 - Read data      ############################
    # Note: leave data in original units (N, mm)

    features, labels_concat = get_data(path_data, SOBOLEV)
    features = add_geom_data(path_data, features, GEOM_SIZE)

    ############################ 1 - Train-Eval-Test Split ############################

    train_eval_test_data = split_data(features, labels_concat, test_size=0.1, eval_size=0.2)
    plot_split_data(train_eval_test_data, plot=PLOT_DATA)

    ############################ 2 - Normalisation  ############################

    stats     = get_stats(train_eval_test_data)
    norm_data = get_normalised_data(train_eval_test_data, stats, SOBOLEV)
    plot_norm_data(norm_data, plot=PLOT_DATA)
    torch_data = data_to_torch(norm_data)

    ############################ 3 - Hyperparams    ############################

    save_inp(inp,  save_path=MODEL_DIR)
    save_stats(stats, save_path=MODEL_DIR)
    save_test_data(train_eval_test_data, save_path=MODEL_DIR)

    ############################ 4 - Train          ############################

    training_wrapper(torch_data, inp,
                     save_path=MODEL_DIR,
                     save_folder=SAVE_FOLDER, sweep=SWEEP,
                     streaming=False)

    ############################ 5 - Test           ############################

    test_data = {'X_test': train_eval_test_data['X_test'],
                 'y_test': train_eval_test_data['y_test']}
    test_NN_model(test_data, stats, save_path=LOGS_DIR, version=None)
