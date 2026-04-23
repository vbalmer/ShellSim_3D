# vb, 16.04.2026

from single_element_utils import *


# geom = [300, 0.025, 0.025, 1]
geom = []
model_path = 'training\\logs'
model_version = 6
model_version_comp = 9
save_path = 'training\\plots_single_el'

single_element_test([0], geom = geom, model_path = [model_path, model_version], save_path = save_path,
                    multirow = False, NN_comp = [model_path, model_version_comp], all_cols = False, train_points = False)