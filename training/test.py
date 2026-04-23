# vb, 21.04.2026
# main test file
# only works for version > 4 (before, no test data saved.)

from test_utils import *


VERSION = 10                 

############################ 5 - Test              ############################

test_data = get_testdata_from_folder('training\\logs', version = VERSION)
stats = get_stats_from_folder('training\\logs', version = VERSION)

test_NN_model(test_data, stats,
              save_path = 'training\\logs', version = VERSION)