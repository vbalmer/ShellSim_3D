# vb, 10.03.2026

import os
from sampler_utils_RC3D import *


########################################## visualise strains ##########################################

save_data_dir = os.path.join('D:\\', 'VeraBalmer\\ShellSim3D')
save_data_path = os.path.join(save_data_dir, 'output_eps_g.h5')
plot_3D_eps(save_data_path, n_every = int(250))

# TODO?
# visualisation of stress distributions across the height of the cross section?

########################################## visualise stresses ##########################################