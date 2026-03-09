# vb, 06.03.2026
# main sampling file for 3D RC shells


import os
import numpy as np

from sampler_utils_RC3D import *


SAMPLE_2D = False


############################ 0 - Get constants ############################

constants, mat_dict = get_constant_sampling_params(SAMPLE_2D)
save_data_dir = os.path.join('D:\\', 'VeraBalmer\\ShellSim3D')


############################ 1 - Sampling strains ############################

if SAMPLE_2D:
    # 1.1 Sample 2D layer strains 
    eps_l = sample_eps(sampler = 'uniform', constants = constants)


    # 1.2 Permute and interpolate linearly to determine gen. strains and curvatures 
    #     (batch-wise implementation), save the generalised strains (batch-wise)
    
    permute_and_save(eps_l, constants, save_data_dir)

else:
    # 1.1 - sample 3D generalised strains
    eps_g = sample_eps(sampler = 'uniform_3D', constants = constants)
    
    # 1.2 - save 3D generalised strains
    save_3D_eps(eps_g, save_data_dir)

save_data_path = os.path.join(save_data_dir, 'output_eps_g.h5')
plot_3D_eps(save_data_path)


############################ 2 - Simulating stresses ############################

# TODO!
# sig_l_all = get_all_sig_2D(eps_l_all)
