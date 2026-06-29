# vb, 06.03.2026
# main sampling file for 3D RC shells

import os
import config
import numpy as cp
import wandb
config.USE_GPU = False

from sampler_utils_RC3D import *
from simulating_sig_vec_RC3D import *


SAMPLE_2D     = False
PLOT_D        = True
SAVE_D        = True
FILTER_DATA   = True
CHUNKED       = False
LOG_WANDB     = os.environ.get('LOG_WANDB', '0').lower() in ('1', 'true', 'yes')
SAMPLING_TYPE = 'combined_log_uniform'   
REMOVE_OUTLIERS = True
CHUNK_SIZE    = 10_000_000


############################ 0 - Get constants ############################

constants, mat_dict = get_constant_sampling_params(sample_2d=SAMPLE_2D)
constants, save_data_dir = setup_sampler_dirs(constants)

config_ = initialise_wandb(constants, mat_dict, LOG_WANDB)

simulatesig = SigSimulator(constants)
cm = 3

save_plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(save_plot_path, exist_ok=True)


############################ 1-4 - Sampling ############################

if SAMPLE_2D:
    eps_l = sample_eps(sampler='uniform', constants=constants)
    permute_and_save(eps_l, constants, save_data_dir, save_batchwise=True)
    eps_g_last = sig_g_last = D_last = None

    raise UserWarning("2D sampling path only calculates strains. Functionality not tested for a while.")

elif CHUNKED:
    eps_g_last, sig_g_last, D_last = run_chunked_sampling(
        constants, mat_dict, save_data_dir, simulatesig, cm,
        sampling_type=SAMPLING_TYPE, chunk_size=CHUNK_SIZE,
        filter_data=FILTER_DATA, save_D=SAVE_D, remove_outliers = REMOVE_OUTLIERS
    )

else:
    
    eps_g = sample_eps(sampler=SAMPLING_TYPE, constants=constants)
    eps_g[:, 2] = eps_g[:, 2] * 2          # eps_xy → gamma_xy
    eps_g[:, 5] = eps_g[:, 5] * 2          # chi_xy → 2*chi_xy

    if FILTER_DATA:
        eps_g = filter_3d_data(eps_g, constants = constants, prefilter = True)

    n_sub = max(1, constants['n_samples_3D'] // 1_000_000)
    sig_g, dh = sig_simulation_batchwise(
        cp.asarray(eps_g), simulatesig, cm, mat_dict, n_batches=n_sub
    )

    eps_g_last, sig_g_last, D_last = eps_g, sig_g, dh

    if REMOVE_OUTLIERS:
        _, _, mask_outliers = find_outlier_d(D_last, eps_g_last, 100)
        eps_g_last = eps_g_last[~mask_outliers]
        sig_g_last = sig_g_last[~mask_outliers]
        D_last = D_last[~mask_outliers]
        print(f'Removed {np.sum(mask_outliers)} outliers from dataset ({np.sum(mask_outliers)/(D_last.shape[0])*100:.3f} %)')

    save_3D_data(eps_g_last, save_data_dir, filename='eps_g')
    save_3D_data(sig_g_last, save_data_dir, filename='sig_g')
    if SAVE_D:
        save_3D_data(D_last, save_data_dir, filename='D')


############################ 5 - Plots (from last chunk) ############################

if PLOT_D and eps_g_last is not None:
    plot_filtered_stiffness(eps_g_last, D_last, 0, save_plot_path, remove_outliers = REMOVE_OUTLIERS)
    imshow_D_filtered(eps_g_last, D_last, 0, save_plot_path)


############################ 6 - Finish ############################

if LOG_WANDB:
    wandb.finish()
