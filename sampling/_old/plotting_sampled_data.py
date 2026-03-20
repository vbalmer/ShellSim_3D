import os
import numpy as np

from data_work import plots_mike_dataset
from plotting_sampled_data_utils import *
from data_work import plotting_sampled_scatter_2D, find_corresponding_eps


mat_data_names = {                                          #sorted in descending order (of epsilon ranges) --> it will be plotted like that
    # '2': '04_Training\\data\\data_20250328_1421_fake',
    # '3': '04_Training\\data\\data_20250428_1150_fake',
    # '4': '04_Training\\data\\data_20250428_1350_fake', 
    # '5': '04_Training\\data\\data_20250428_1438_fake',
    # '1': '04_Training\\data\\data_20250328_1431_fake',
    # '8': '04_Training\\data\\data_20250506_1902_fake',
    # '9': '04_Training\\data\\data_20250507_1826_fake',
    # '7': '04_Training\\data\\data_20250505_1217_fake',
    #'10': '04_Training\\data\\data_20250508_0916_fake',
    # '6': '04_Training\\data\\data_20250430_1501_fake',
    # '11': '04_Training\data\data_20250512_1000_fake',
    # '12': '04_Training\data\data_20250512_1009_fake',
    # '13': '04_Training\data\data_20250512_1035_fake',
    # '14': '04_Training\data\data_20250512_1728_fake', 
    # '15': '04_Training\data\data_20250512_1744_fake',
    # '16': '04_Training\data\data_20250515_1617_fake'
    # '17': '04_Training\\data\\data_20250515_2018_fake',
    # '18': '04_Training\\data\\data_20250516_0934_fake',
    # '19': '04_Training\data\data_20250516_1154_fake',
    # '20': '04_Training\data\data_20250516_1435_fake',
    # '22': '04_Training\data\data_20250521_1224_fake',
    # '23': '04_Training\data\data_20250522_1829_casexx',
    # '24': '04_Training\data\data_20250526_1019_casexx',
    # '25': '04_Training\data\data_20250526_2110_fake',
    # '26': '04_Training\data\data_20250527_1144_casexx'
    # '27': '04_Training\data\data_20250603_2149_fake',
    # '29': '04_Training\data\data_20250608_1748_fake',
    # '30': '04_Training\data\data_20250608_1832_fake',
    # '28': '04_Training\data\data_20250605_1144_fake',
    # 'xx': '04_Training\data\data_20250608_1838_fake',
    # '32': '04_Training\data\data_20250815_1055_fake',
    # '33': '04_Training\data\data_20250815_1103_fake', 
    # '34': '04_Training\data\data_20250821_1516_fake',
    # '35': '04_Training\data\data_20250825_1731_fake',
    # '36': '04_Training\data\data_20250827_1100_fake',
    # '37': '04_Training\data\data_20250825_1628_fake',
    # '38': '04_Training\data\data_20250825_1654_fake',
    # '39': '04_Training\data\data_20250826_1705_fake',
    # '40': '04_Training\data\data_20250827_1513_fake',
    # '41': '04_Training\data\data_20250827_1830_fake',
    # '42': '04_Training\data\data_20250827_1935_fake',
    # '43': '04_Training\data\data_20250828_1218_fake',
    # '44': '04_Training\data\data_20250828_1422_fake',
    # '45': '04_Training\data\data_20250828_1510_fake'
    # '46': '04_Training\data\data_20250828_1607_fake',
    # '49': '04_Training\data\data_20250828_1800_fake',
    # '50': '04_Training\data\data_20250901_1632_fake',
    # '51': '04_Training\data\data_20250901_1702_fake',
    # '52': '04_Training\data\data_20250901_1815_fake',
    # '53': '04_Training\data\data_20250901_1751_fake'
    # '57': '04_Training\data\data_20250903_1059_fake',
    # '54': '04_Training\data\data_20250902_1039_fake',
    # '55': '04_Training\data\data_20250903_0924_fake',
    # 'xx': '04_Training\data\data_20250924_1054_fake',
    # '58': '04_Training\data\data_20250910_1011_fake'
    # '61': '04_Training\data\data_20251001_1641_fake',
    # '62': '04_Training\data\data_20251001_1708_fake',

}

mat_data_paths = get_paths_data(mat_data_names)
save_path = os.path.join(os.getcwd(), '04_Training\\plots\\test')
MASK = False        # plot masked datasets yes or no
OUTLIERS = False
PLOTMIKE = False
THREEDHIST = False
FILTERED = False
FILTERED_CS = False
PLOT_3D_SCATTER = True
FIND_CORRESPONDING_EPS = False
FILTER_DATASET = False

################## READ DATA #######################

if mat_data_names.keys() != None:
    numpy_data_dict = read_all_data(mat_data_paths)
    print('Finished reading data')

################## FORMAT DATA #########################

if mat_data_names.keys() != None:
    plot_data, plot_data_cut = format_all_data(numpy_data_dict, n=20000, outliers = OUTLIERS)         # for total range use: numpy_data_dict['1']['eps'].shape[0]
    print('Finished formatting plot_data')



################## PLOTTING ###########################

## plotting entire datasets

if PLOTMIKE:
    plots_mike_dataset(None, None, None, None, save_path, tag='data_no', tag2 = None, 
                        add_dict = plot_data_cut, outliers = OUTLIERS)

    plots_mike_dataset(None, None, None, None,  save_path, tag='data_no_D', tag2 = 'D', 
                        add_dict = plot_data, outliers = OUTLIERS, linel = True)


if THREEDHIST:
    three_D_histogram(plot_data, 7, 37, alpha_ = 0.3, save_path=save_path, linel=True)


if MASK:
    ## plotting masked datasets
    min_vals = np.array([-3e-4]*2 + [-4e-4] + [-30e-7]*2 + [-40e-7] + [-0.5e-4]*2)
    max_vals = np.array([5e-4]*2  + [4e-4]  + [50e-7]*2  + [40e-7]  + [0.5e-4]*2)

    mask2 = np.all((plot_data['2']['x_data'][:,:8] >= min_vals) & (plot_data['2']['x_data'][:,:8] <= max_vals))        # masking only needs to be carried out on larger dataset
    mask3 = np.all((plot_data['3']['x_data'][:,:8] >= min_vals) & (plot_data['3']['x_data'][:,:8] <= max_vals))

    D_data2_masked = plot_data['2']['D_data'][mask2]
    D_data3_masked = plot_data['3']['D_data'][mask3]

    if np.sum(mask2) != 0:
        plots_mike_dataset(plot_data['1']['x_data'], plot_data['2']['x_data'], plot_data['1']['D_data'], D_data2_masked, save_path, tag='data2_masked', tag2 = 'D')
    else: 
        print('Min value of mask2:', np.min(mask2), ' Max value of mask2:', np.max(mask2))
    if np.sum(mask3) != 0:
        plots_mike_dataset(plot_data['1']['x_data'], plot_data['3']['x_data'], plot_data['1']['D_data'], D_data3_masked, save_path, tag='data3_masked', tag2 = 'D')
    else:
        print('Min value of mask3:', np.min(mask3), ' Max value of mask3:', np.max(mask3))
        print('There are no datapoints in the large dataset(s) in the given range.')


if FILTERED: 
    plot_filtered_data(plot_data, idx_eps=[0], geom = [200, 0.02, 3], stiffness_plots=True, save_path = save_path)

if FILTERED_CS:

    eps_mat = {
        'eps_x = 0.002': [0.002] + [0]*7,
        'eps_x = 0.001': [0.001] + [0]*7,
        'eps_x = -0.001': [-0.001] + [0]*7,
        'eps_x = -0.002': [-0.002] + [0]*7,
        'eps_x = -0.0022': [-0.0022] + [0]*7,
        'eps_x = -0.0025': [-0.0025] + [0]*7,
        'eps_x = -0.0029': [-0.0029] + [0]*7,
    }

    # eps_mat = {
    #     'eps_x = -0.0029': [-0.0029] + [0.4985e-7]*2+ [0.4985e-9]*3 + [0.4985e-7]*2,
    # }


    plot_filtered_crosssection(eps = eps_mat, geom = [250, 0.015, 2], save_path = save_path, dir = 0)



    from data_work import plotting_sampled_scatter_2D

if PLOT_3D_SCATTER: 
    plotting_sampled_scatter_2D([
                                # 'data_20260108_1520_fake',
                                # 'data_20260105_1639_fake',
                                # 'data_20260108_2137_fake',
                                # 'data_20260108_1925_fake',
                                # 'data_20260105_1416_fake',
                                # 'data_20260113_1611_fake',
                                # 'data_20260113_1403_fake',
                                # 'data_20260109_1821_fake',
                                'data_20260110_1141_fake'
                                

                                          
                                ], 
                                filter_type = None,
                                plot_type = 'eps',
                                clean_data = None,      # if None: Does not mark outliers, else: marks outliers based on hull of clean_data
                                save_new_data = False,
                                save_path = save_path)
    
if FIND_CORRESPONDING_EPS:

    sig_arr = np.array([[-2505.248, 977.855, 1438.111],
                    [-1787.905, 232.3166, 2446.304],
                    [-2428.654, 1066.28, 1155.419],
                    [1141.227, -1656.719, 2231.976],
                    [1216.798, -1902.505, 1409.561],
                    [-251.7845, -1823.234, 2199.931],
                    [1423.762, 1418.998, 2073.133],
                    [1407.096, 1420.984, 905.0159],
                    [1388.913, 1416.436, -24.62631],
                    ])

    filtered_eps = find_corresponding_eps(sig = sig_arr, data_path = 'data_20251117_1527_fake')
    print(filtered_eps)


if FILTER_DATASET: 
    # filters dataset according to predefined criteria and saves a new dataset.
    data_path = 'data_20260108_1156_fake'
    filter_dataset(data_path, save_new_data=False)