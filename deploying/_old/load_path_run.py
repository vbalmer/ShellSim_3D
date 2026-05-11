### load-path wrapper for run_NN.py
# vb, 10.11.2025

import numpy as np
from datetime import datetime
import os
import wandb

from run_NN import run_deployment_loadpath, save_deployment_loadpath


#################
# Input definition
#################

mat_tot_dict = {                        # no force definition here, as this changes with every iteration
    'L': np.array([6000]),
    'B': np.array([6000]),
    'CC': np.array([1]),
    'E_1': np.array([0]),
    'E_2': np.array([0]),
    'ms': np.array([600]),
    's': np.array([9]),
    't_1': np.array([300]),
    't_2': np.array([0]),
    'nl': np.array([20]),
    'nu_1': np.array([0]),
    'nu_2': np.array([0]),
    'mat': np.array([3]),
    'rho_x': np.array([0.025]),
    'rho_y': np.array([0.0375])
}


inp_run = {
    "mat_tot_dict": mat_tot_dict,
    # "model_no": ["479", "294"],                               # [model version, epoch number]
    "model_no": ["480", "167"],
    # "model_no": ["481", "246"],
    "numit": 9,
    "predict": [True, True]                                     # [predict_sig, predict_D]
}

CONTINUE = True                                                 # if CONTINUE = True: continues iteration from previous load level, 
                                                                # does not start every load level from linear elastic
                                                                # if only calculating one load level, please use Continue = False


# rho_y = 1%
# load_levels = [200, 400, 600, 800, 1000, 1100, 1200, 1300, 1350, 1375, 1400, 1410, 1420]
# load_levels = [-2000, -4000, -6000, -8000, -9000, -10000, -10100, -10200, -10250, -10300]
# load_levels = [200, 400, 600, 800, 900, 910, 920, 930, 940, 950]

# rho_y = 1.5%
# load_levels = [200, 600, 1000, 1200, 1400, 1600, 1650, 1700, 1710, 1720, 1730, 1740]
# load_levels = [200, 600, 1000, 1200, 1400, 1600, 1800, 2000, 2025, 2050, 2060, 2070]
load_levels = [-2000, -4000, -6000, -8000, -9000, -10000, -10100, -10200, -10250, -10300]


# rho_y = 0.75%
# load_levels = [200, 400, 600, 800, 1000, 1100, 1125, 1150, 1175, 1200]
# load_levels = [200, 400, 600, 800, 850, 900, 1000, 1025, 1050, 1075, 1100]
# load_levels = [-2000, -4000, -6000, -8000, -9000, -10000, -10100, -10200, -10250, -10300]


#################
# Run load-deformation path
#################

current_time = datetime.now()
new_folder = current_time.strftime("data_%Y%m%d_%H%M_case" + 'xx')
new_folder_path = os.path.join('05_Deploying\\data_out', new_folder)
os.makedirs(new_folder_path, exist_ok=True)


if CONTINUE: 
    run = wandb.init(
        project="ML-FEA-deployment-v1",
        config = dict(mat_tot_dict),
    )

    NN_hybrid, conv_plt = run_deployment_loadpath(inp_run, load_levels, new_folder_path)

else: 
    for force_i in load_levels: 
        run = wandb.init(
            project="ML-FEA-deployment-v1",
            config = dict(mat_tot_dict),
        )

        NN_hybrid, conv_plt = run_deployment_loadpath(inp_run, force_i)

        # save all files to same folder 
        save_deployment_loadpath(new_folder_path, force_i, NN_hybrid, conv_plt)
        print(f'Finished run {load_levels.index(force_i)+1}/{len(load_levels)}, with force {force_i} N/mm.')


    