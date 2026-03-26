# hyperparameters for training.


inp = {
    'input_size': 6,                                # 6 + GEOM_SIZE
    'out_size': 6,                                  # 6
    'hidden_layers': str([512]*5),
    'batch_size': 4096,                             # Can be defined here, if None: runs with single-batch
    'num_epochs': 300,
    'switch_step_percentage': 1,                    # Percentage after which to switch to LBFGS instead of Adam 
    'activation': 'ELU',
    'learning_rate': 0.005,
    'lr_scheduler':'standard',                      # 'standard': the one used by mike; 'plateau': reduceLRonPlateau
    'dropout_rate': 0,
    'BatchNorm': False,
    'num_samples': 5, 
    'fourier_mapping': False,  
    'loss_type': 'MSELoss',                         # can be 'MSELoss', 'HuberLoss', 'MSLELoss', 'wMSELoss', 'RMSELoss'                   

    # Network type (Sobolev, Pretrained, DeepONet, MoE)          
    'w_s': str([0.9,0.1]),                          # weights [w1, w2] for 1st, 2nd order loss with sobolev (neglected if sobolev = False), or 'max'
}


constant_inp = {
    'simple_m': True,
    'lr_scheduler':'standard',
    'num_samples': 5,
    'BatchNorm': False,                         
    'kfold': False,
    }



sweep_config = {
            "method":"random",
            "metric": {"goal": "minimize", "name": "best_val_loss"},
            "parameters":{
                "input_size": {"values": [8]},      # needs to be 8 for DeepONet
                "out_size": {"values": [8]},
                "num_epochs": {"values": [30000]},
                "learning_rate": {"values": [0.001]},
                "hidden_layers": {"values": [str([64]*20), str([64]*10), str([128]*10), str([128]*20)]},
                "dropout_rate": {"values": [0, 0.05]},
                "loss_type": {"values": ['MSELoss']},
                "activation": {"values": ['ELU','ReLU']},
                "fourier_mapping": {"values": [True, False]},
                "pretrain": {"values": [None]},
                "w_s": {"values": ['max', str([0.4, 0.6]), str([0.5, 0.5]), str([0.25, 0.75])]},    #, str([0.2, 0.8]), str([0.3, 0.7])]},
                "DeepONet":{"values": [True]},
            }
        }