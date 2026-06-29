# vb, 10.03.2026

import os
from sampler_utils_RC3D import *


# ########################################## visualise strains ##########################################

# save_data_dir = os.path.join('C:\\', 'kuy_sampling_strain-stress')
# save_data_path = os.path.join(save_data_dir, 'output_eps_g.h5')
# plot_3D_data(save_data_path, filename = 'scatter_eps_g', n_every = int(1))

# # TODO?
# # visualisation of stress distributions across the height of the cross section?

# ########################################## visualise stresses ##########################################

# save_data_dir = os.path.join('C:\\', 'kuy_sampling_strain-stress')
# save_data_path = os.path.join(save_data_dir, 'output_sig_g.h5')
# plot_3D_data(save_data_path, filename = 'scatter_sig_g', n_every = int(1))


# Ergänzung kuy, 29.06.2026
########################################## read-in data ##########################################
save_data_dir = os.path.join('C:\\', 'kuy_sampling_strain-stress')
save_data_path = os.path.join(save_data_dir, 'output_eps_g.h5')
data_eps_g = read_h5_file(save_data_path,  filename = 'scatter_eps_g', n_every = int(1))

print(type(data_eps_g))      # should be numpy.ndarray
print(data_eps_g.shape)      # e.g. (100000, 6)
print(data_eps_g)        # first rows

save_data_dir = os.path.join('C:\\', 'kuy_sampling_strain-stress')
save_data_path = os.path.join(save_data_dir, 'output_sig_g.h5')
data_sig_g = read_h5_file(save_data_path,  filename = 'scatter_sig_g', n_every = int(1))

print(type(data_sig_g))      # should be numpy.ndarray
print(data_sig_g.shape)      # e.g. (100000, 6)
print(data_sig_g[:5])        # first rows
