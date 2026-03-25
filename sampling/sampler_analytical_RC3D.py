# vb, 06.03.2026
# main sampling file for 3D RC shells


import os
import config
import cupy as cp
config.USE_GPU = True

from sampler_utils_RC3D import *
from simulating_sig_vec_RC3D import *


SAMPLE_2D = False
PLOT_D = True
SAVE_D = True
BATCHWISE = True



############################ 0 - Get constants ############################

constants, mat_dict = get_constant_sampling_params(SAMPLE_2D)
save_data_dir = os.path.join('D:\\', 'VeraBalmer\\ShellSim3D')


############################ 1 - Sampling strains ############################

if SAMPLE_2D:
    # 1.1 Sample 2D layer strains 
    eps_l = sample_eps(sampler = 'uniform', constants = constants)
    eps_l[:,2] = eps_l[:,2]*2   #converting eps_xy to gamma_xy

    # 1.2 Permute and interpolate linearly to determine gen. strains and curvatures 
    #     (batch-wise implementation), save the generalised strains (batch-wise)
    
    permute_and_save(eps_l, constants, save_data_dir)

else:
    # 1.1 - sample 3D generalised strains
    eps_g = sample_eps(sampler = 'uniform_3D_grouped', constants = constants)
    eps_g[:,2] = eps_g[:,2]*2   #converting eps_xy to gamma_xy

    
    # 1.2 - save 3D generalised strains
    save_3D_data(eps_g, save_data_dir, filename = 'eps_g')


############################ 2 - Simulating stresses ############################

simulatesig = SigSimulator(constants)
cm = 3

if not BATCHWISE:

    # 2.1 Find layer strains
    e = simulatesig.find_e_vec(eps_g)

    # 2.2 Find layer stresses
    s = simulatesig.find_s_vec(e, mat_dict, cm_klij = cm)

    # 2.3 Find generalised stresses
    sig_g = simulatesig.find_sh_vec(s, cm_klij = cm)

    # 2.4 Find stiffnesses
    dh = simulatesig.find_dh_vec(s, mat_dict, cm_klij = cm)

    # 2.5 Save generalised stresses
    save_3D_data(sig_g, save_data_dir, filename = 'sig_g')

else: 
    sig_g, dh = sig_simulation_batchwise(cp.asarray(eps_g), simulatesig, cm, mat_dict, n_batches = 1)

    # 2.5 Save generalised stresses
    save_3D_data(sig_g, save_data_dir, filename = 'sig_g')


############################ 3 - Visualising stiffnesses ############################

if PLOT_D:
    save_plot_path = os.path.join(os.getcwd(), "sampling\\plots\\")
    plot_filtered_stiffness(eps_g, dh, 0, save_plot_path)
    imshow_D_filtered(eps_g, dh, 0, save_plot_path)

    # for debugging:
    # imshow_D_all(dh, save_plot_path)
    # imshow_sig_eps_all(sig_g, eps_g, save_plot_path)


if SAVE_D: 
    save_3D_data(dh, save_data_dir, filename = 'D')