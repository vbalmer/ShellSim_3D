# checking storage specs
import os
import shutil

n_tot = 4e9
eps_l_all_bytes = n_tot*3*20*4          # float32 = 4 bytes
eps_g_bytes = n_tot*6*4

dh_bytes = n_tot*6*6*4

# Calculate required disk space
total_gb = (eps_l_all_bytes + eps_g_bytes) / 1e9
print(f'Estimated file size eps_l_all with n_tot = {n_tot/1e9:.1f}*1e9: {eps_l_all_bytes/1e9:.1f} GB')
print(f'Estimated file size eps_g with n_tot = {n_tot/1e9:.1f}*1e9: {eps_g_bytes/1e9:.1f} GB')
print(f'Estimated file size dh with n_tot = {n_tot/1e9:.1f}*1e9: {dh_bytes/1e9:.1f} GB')
print(f'Combined file size wiht n_tot = {n_tot/1e9:.1f}*1e9: {total_gb:.1f} GB')

# Check available disk space
save_dir = os.path.join(os.getcwd(),'sampling\\data')
free_gb = shutil.disk_usage(save_dir).free / 1e9
print(f'Available disk space in {save_dir}: {free_gb:.1f} GB')

data_dir = os.path.join('D:\\', 'VeraBalmer\\ShellSim3D')
free_gb_data = shutil.disk_usage(data_dir).free / 1e9
print(f'Available disk space in {data_dir}: {free_gb_data:.1f} GB')