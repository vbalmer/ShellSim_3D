# vb, 16.04.2026

from single_element_utils import *


# geom = [300, 0.025, 0.025, 1]
geom = []
model_path = 'training\\logs'
model_version = 1
save_path = 'training\\plots_single_el'

single_element_test([3], geom = geom, model_path = [model_path, model_version], save_path = save_path,
                    multirow = False, NN_comp = None, all_cols = False, train_points = False)