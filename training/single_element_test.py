# vb, 16.04.2026

from single_element_utils import *


# geom = [300, 0.025, 0.025, 1]
geom = []
model_path = 'training\\logs'
model_version = 12
model_version_comp = 15

save_path = 'training\\plots_single_el'


min_, max_ = -3e-3, 5e-3
# min_, max_ = -0.02e-3, 0.033e-3

single_element_test([0], geom = geom, model_path = [model_path, model_version], 
                    min_ = min_, max_ = max_, save_path = save_path, plot_LFEA = False,
                    multirow = False, NN_comp = [model_path, model_version_comp], all_cols = False, test_points = False)