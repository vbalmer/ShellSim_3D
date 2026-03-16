# vb, 06.03.2026
# main sampling file for 3D RC shells


import os
import numpy as np

from sampler_utils_RC3D import *
from simulating_sig_vec_RC3D import *


SAMPLE_2D = False


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
    save_3D_eps(eps_g, save_data_dir)


############################ 2 - Simulating stresses ############################

simulatesig = SigSimulator(constants)

# 2.1 Find layer strains
e = simulatesig.find_e_vec(eps_g)

# 2.2 Find layer stresses
s = simulatesig.find_s_vec(e, mat_dict)

# 2.3 Find generalised stresses
sh = simulatesig.find_sh_vec(s)

# 2.4 Find stiffnesses
dh = simulatesig.find_dh_vec(s)

# 2.5 Save generalised stresses
#TODO!