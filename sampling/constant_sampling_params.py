#constant_sampling_params


c = {
        'n_samples_2D': 1e6, 
        'n_samples_3D': 1e6,         #4e9,47e3 (for 6 elements per dimension), 64e6 (20 elements per dimension)

        'min': [-3e-3]*2 + [-4e-3],
        'max': [5e-3]*2  + [4e-3],
        't': 350,
        'CC': 4,    #concrete class
        'n_layer': 20,
        'nu': 0.2,
        'fsy': 520,
        'fsu': 600,
        'Es': 205e3,
        'Esh': 9.4e3,
        'D_x': [14]*4 + [0]*12 + [14]*4,    # length of the array needs to correspond to the amount of layers.
        'D_y': [14]*4 + [0]*12 + [20]*4,
        'Dmax': 16,
        's': 200,
        'rho_x': [0.044]*4 + [0]*12 + [0.044]*4,    # length of the array needs to correspond to the amount of layers.
        'rho_y': [0.066]*4 + [0]*12 + [0.135]*4,
        'rho_sublayer': True,

    }

c_3D = {        
    'min': [-3e-3]*2 + [-4e-3] + [-0.02e-3]*2 + [-0.027e-3],        # units: [-], [1/mm]
    'max': [5e-3]*2  + [4e-3] +  [0.033e-3]*2 + [0.027e-3],         # units: [-], [1/mm]
    'min_log': [-3e-3]*2 + [-4e-3] + [-0.02e-3]*2 + [-0.027e-3],        # units: [-], [1/mm]
    'max_log': [5e-3]*2  + [4e-3] +  [0.033e-3]*2 + [0.027e-3],         # units: [-], [1/mm]
    # 'min_log': [-3e-3]*2 + [-50e-3] + [-0.02e-3]*2 + [-0.33e-3],        # units: [-], [1/mm]
    # 'max_log': [50e-3]*2  + [50e-3] +  [0.33e-3]*2 + [0.33e-3],         # units: [-], [1/mm]
    'p_samples_log': 0.5,                                           # percentage of data points to sample in log-based manner; 
                                                                    # if "None", resorts to half-half
}