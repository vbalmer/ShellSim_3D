import pickle
from FFNN_class_light import *
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import TensorDataset, DataLoader

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import wandb
import re
import random
from sklearn.model_selection import KFold


loss_mapping = {
    'MSELoss': nn.MSELoss,
    'HuberLoss': nn.HuberLoss,
    'MSLELoss': MSLELoss,
    'wMSELoss': wMSELoss,
    'RMSELoss': RMSELoss
}

# --------------------------------------------------Model Creation-------------------------------------------------------------

def model_instance(inp, inp_dict, save_path):
    '''
    Creates an instance of the model. 
    If required, adjust the hyperparameters of separate split nets here.    
    '''
    if inp['simple_m']:
        model_dict = {}
        if inp['pretrain'] == None:
            if inp['DeepONet'] == True:
                model = DeepONet_vb(inp)
            elif inp['cVAE'] == True:
                model = cVAE(inp)
            else: 
                model = FFNN(inp)
        elif inp['pretrain'] is not None:
            if not inp['simple_m']:
                my_pretrained_model_path = os.path.join(os.getcwd(), '04_Training\\logs\\train_log\\version_'+ inp['pretrain'] +'\\checkpoints\\best_model.ckpt') 
                my_pretrained_model = LitFFNN.load_from_checkpoint(my_pretrained_model_path)
                model = FFNN_pretrain(inp, my_pretrained_model)
            elif inp['simple_m']:
                # Get pretrained model
                pretrain_id = ast.literal_eval(inp['pretrain'])
                my_pretrained_model_path = os.path.join(os.getcwd(), '04_Training\\new_data\\_simple_logs\\'+ pretrain_id[0] +'\\best_trained_model_'+pretrain_id[1]+'.pt')
                inp_pretrained_path = os.path.join(os.getcwd(), '04_Training\\new_data\\_simple_logs\\'+ pretrain_id[0] +'\\inp.pkl')
                with open(inp_pretrained_path,'rb') as handle:
                    inp_pretrained = pickle.load(handle)
                # Define Models
                if not inp_pretrained['DeepONet']:
                    my_pretrained_model = FFNN(inp_pretrained)
                elif inp_pretrained['DeepONet']:
                    my_pretrained_model = DeepONet_vb(inp_pretrained)
                my_pretrained_model.load_state_dict(torch.load(my_pretrained_model_path))
                my_pretrained_model.eval()
                model = FFNN_pretrain(inp, my_pretrained_model)
        if inp['MoE']:
            model = MoE(inp)
            if inp['MoE-split']:
                raise Warning('This version of the MoE is outdated.')
                expert1 = Expert(inp)
                expert2 = Expert(inp)
                expert3 = Expert(inp)
                model = MoE(inp, [expert1, expert2, expert3])
                model_dict['exp1'] = expert1
                model_dict['exp2'] = expert2
                model_dict['exp3'] = expert3
            
        
        model_dict['standard'] = model
        model_dict['inp1'], model_dict['inp2'] = None, None
        model_m, model_b, model_s = None, None, None

    else:
        # using pytorch lightning
        raise RuntimeWarning('The pytorch lighnting version of this code is outdated. Please use simple_m = True')
        if not inp['Split_Net'] and not inp['Double_Net']:
            if inp['pretrain'] == None:
                model = LitFFNN(inp, FFNN(inp))
                # model.load_state_dict(torch.load(os.path.join(save_path, 'trained_model.pt')))                #load old trained model
            else:
                my_pretrained_model_path = os.path.join(os.getcwd(), 'logs\\train_log\\version_'+ inp['pretrain'] +'\\checkpoints\\best_model.ckpt') 
                my_pretrained_model = LitFFNN.load_from_checkpoint(my_pretrained_model_path)
                model = LitFFNN(inp, FFNN_pretrain(inp, my_pretrained_model))
            inp1, inp2 = None, None
            model_m, model_b, model_s = None, None, None
        elif inp['Double_Net']:
            if inp['pretrain']:
                raise TypeError("The double-Net is not set up for use with the pretrained model")
            inp1 = inp.copy()
            inp1['out_size'] = 64
            model = LitFFNN_doub(inp, FFNN(inp), FFNN(inp1))
            inp2 = None
            model_m, model_b, model_s = None, None, None
        elif inp['Split_Net']:
            # make sure no mistake happens:
            if inp['pretrain']:
                raise TypeError("The Split Net is not set up for use with the pretrained model")
            if inp['Sobolev']:
                    raise KeyError("Sobolev Loss and Split-Net is not implemented yet.")
            
            # Redefine inp-outp size
            inp0, inp1, inp2 = inp.copy(), inp.copy(), inp.copy()
            inp0['input_size'], inp0['out_size'] = 4, 3
            inp1['input_size'], inp1['out_size'] = 4, 3
            inp1['dropout_rate'] = 0.4
            inp2['input_size'], inp2['out_size'] = 3, 2
            inp2['hidden_layers'] = [64, 64, 64]
            model_m = LitFFNN(inp0, FFNN(inp0))
            model_b = LitFFNN(inp1, FFNN(inp1))
            model_s = LitFFNN(inp2, FFNN(inp2))
            model = None

            # save to file
            with open(os.path.join(save_path, 'inp0.pkl'), 'wb') as fp:
                    pickle.dump(inp0, fp)
            with open(os.path.join(save_path, 'inp1.pkl'), 'wb') as fp:
                    pickle.dump(inp1, fp)
            with open(os.path.join(save_path, 'inp2.pkl'), 'wb') as fp:
                    pickle.dump(inp2, fp)

        model_dict = {
            "standard":model, 
            "m":model_m,
            "b":model_b,
            "s":model_s,
            "inp1": None,
            "inp2": None,
        }

        inp_dict = None


    return model_dict, inp_dict


def model_print(model_dict, inp):
    '''
    Prints model architecture for reference in wandb
    '''
    model = model_dict['standard']
    print(model)
    return 


def plot_datasplit(mat_torch_exp1, mat_torch_exp2, mat_torch_exp3, save_path=None, tag = 'train'):
    '''
    plots 8x8 matrix of scatter plots (epsilon vs. sigma)
    in colors: marks the different regions of the data split of the MoE
    mat_torch_exp1...3      (dicts)         contain the variables X_train_tt, X_eval_tt, y_train_tt, y_eval_tt
    save_path               (str)           location to save the figure
    tag                     (str)           can be 'train' or 'eval'
    '''

    plotname_sig = np.array(['$n_x$', '$n_y$', '$n_{xy}$',
                            '$m_x$', '$m_y$', '$m_{xy}$',
                            '$v_{xz}$', '$v_{yz}$', '$v_y$'])
    
    plotname_eps = np.array([r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$',
                             r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$',
                             r'$\gamma_x$', r'$\gamma_y$', r'$\gamma_{xy}$',
                             ])

    nRows = 8
    nCols = 8
    fig, axs = plt.subplots(nCols, nRows, figsize=(2*nRows, 2*nCols))
    for i in range(nCols):
        for j in range(nRows):
            axs[i, j].scatter(mat_torch_exp3['X_'+tag+'_tt'][:,j], mat_torch_exp3['y_'+tag+'_tt'][:,i], color = 'g', s=1, label='expert 3', alpha = 0.1)
            axs[i, j].scatter(mat_torch_exp2['X_'+tag+'_tt'][:,j], mat_torch_exp2['y_'+tag+'_tt'][:,i], color = 'r', s=1, label='expert 2', alpha = 0.1)
            axs[i, j].scatter(mat_torch_exp1['X_'+tag+'_tt'][:,j], mat_torch_exp1['y_'+tag+'_tt'][:,i], color = 'b', s=1, label = 'expert 1', alpha = 0.1)
            if i == nRows-1:
                axs[i, j].set_xlabel(plotname_eps[j])
            if j == 0:
                axs[i, j].set_ylabel(plotname_sig[i], rotation = 90)
    plt.legend()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "data_split_MoE.png"))
        print('saved \'data split MoE plot\'')

    plt.close()

def data_split_MoE(mat_torch, stats_y_train, thresholds, type):
    ''' 
    splits the data given in mat_torch into subsets for training of MoE
    thresholds: boundaries for splitting train set to get 3 expert subsets
    mat_torch        (dict)             contains the data that shall be split
    stats_y_train    (dict)             contains statistical info about y-data
    thresholds       (list)             contains two thresholds according to which the data shall be split
    type             (str)              can be 'absolute', 'range', 'random' or 'cluster', where the cluster is not yet implemented
    '''
    
    num_samples_train = mat_torch['X_train_tt'].shape[0]
    num_samples_eval = mat_torch['X_eval_tt'].shape[0]
    desired_keys = ['X_train_tt', 'y_train_tt', 'X_eval_tt', 'y_eval_tt']
    desired_keys_train = ['X_train_tt', 'y_train_tt']
    desired_keys_eval = ['X_eval_tt', 'y_eval_tt']
    mat_torch_exp, mat_torch_MoE = {},{}
    for key in desired_keys_train:
        mat_torch_exp[key] = mat_torch[key][:int(num_samples_train/2)]
        mat_torch_MoE[key] = mat_torch[key][int(num_samples_train/2)+1:]
    for key in desired_keys_eval:
        mat_torch_exp[key] = mat_torch[key][:int(num_samples_eval/2)]
        mat_torch_MoE[key] = mat_torch[key][int(num_samples_eval/2)+1:]

    if type == 'absolute':
        thresholds_std = (10**(-5)*thresholds-stats_y_train['mean'][7])/stats_y_train['std'][7]
        mask1, mask2 = {},{}
        mask1['0'] = (abs(mat_torch_exp['y_train_tt'][:,7]) < thresholds_std[0])
        mask2['0'] = (abs(mat_torch_exp['y_eval_tt'][:,7]) < thresholds_std[0])
        mask1['1'] = (abs(mat_torch_exp['y_train_tt'][:,7]) >= thresholds_std[0]) & (abs(mat_torch_exp['y_train_tt'][:,7]) < thresholds_std[1])
        mask2['1'] = (abs(mat_torch_exp['y_eval_tt'][:,7]) >= thresholds_std[0]) & (abs(mat_torch_exp['y_eval_tt'][:,7]) < thresholds_std[1])
        mask1['2'] = (abs(mat_torch_exp['y_train_tt'][:,7]) >= thresholds_std[1])
        mask2['2'] = (abs(mat_torch_exp['y_eval_tt'][:,7]) >= thresholds_std[1])
    elif type == 'range':
        thresholds = [-0.5, 0.5]                # use the standardised thresholds
        mask1, mask2 = {},{}
        mask1['0'] = (mat_torch_exp['y_train_tt'][:,7] < thresholds[0])
        mask2['0'] = (mat_torch_exp['y_eval_tt'][:,7] < thresholds[0])
        mask1['1'] = (mat_torch_exp['y_train_tt'][:,7] >= thresholds[0]) & (mat_torch_exp['y_train_tt'][:,7] < thresholds[1])
        mask2['1'] = (mat_torch_exp['y_eval_tt'][:,7] >= thresholds[0]) & (mat_torch_exp['y_eval_tt'][:,7] < thresholds[1])
        mask1['2'] = (mat_torch_exp['y_train_tt'][:,7] >= thresholds[1])
        mask2['2'] = (mat_torch_exp['y_eval_tt'][:,7] >= thresholds[1])
    elif type == 'random':
        mask1, mask2 = {},{}
        splits_train = [0, int(num_samples_train/2) // 3, 2 * int(num_samples_train/2) // 3, int(num_samples_train/2)]
        splits_eval = [0, int(num_samples_eval/2) // 3, 2 * int(num_samples_eval/2) // 3, int(num_samples_eval/2)]
        for i in range(3):
            m_ = torch.zeros((int(num_samples_train/2)), dtype=torch.bool)
            _m_ = torch.zeros((int(num_samples_eval/2)), dtype=torch.bool)
            m_[splits_train[i]:splits_train[i+1]] = True
            mask1[str(i)] = m_.clone()
            _m_[splits_eval[i]:splits_eval[i+1]] = True
            mask2[str(i)] = _m_.clone()

    elif type == 'cluster':
        RuntimeError('To be implemented')

    mat_torch_exp1, mat_torch_exp2, mat_torch_exp3 = {key: None for key in desired_keys}, {key: None for key in desired_keys}, {key: None for key in desired_keys}
    for key in desired_keys_train: 
        mat_torch_exp1[key] = mat_torch_exp[key][mask1[str(0)],:]
        mat_torch_exp2[key] = mat_torch_exp[key][mask1[str(1)],:]
        mat_torch_exp3[key] = mat_torch_exp[key][mask1[str(2)],:]
    for key in desired_keys_eval:
        mat_torch_exp1[key] = mat_torch_exp[key][mask2[str(0)],:]
        mat_torch_exp2[key] = mat_torch_exp[key][mask2[str(1)],:]
        mat_torch_exp3[key] = mat_torch_exp[key][mask2[str(2)],:]
    
    print('The data distribution among the 3 experts in the training data is: ')
    print('Expert 1:', round(mat_torch_exp1['X_train_tt'].shape[0]/(num_samples_train/2)*100,0), 
            '% \n Expert 2:', round(mat_torch_exp2['X_train_tt'].shape[0]/(num_samples_train/2)*100,0),
            '% \n Expert 3:', round(mat_torch_exp3['X_train_tt'].shape[0]/(num_samples_train/2)*100,0), '%')
    
    print('The ratio of train to eval data is (for train-eval split = 70\%-20\%, ratio should be 3.5):  ')
    print('Expert 1:', round(mat_torch_exp1['X_train_tt'].shape[0]/mat_torch_exp1['X_eval_tt'].shape[0],1), 
            '\n Expert 2:', round(mat_torch_exp2['X_train_tt'].shape[0]/mat_torch_exp2['X_eval_tt'].shape[0],1),
            '\n Expert 3:', round(mat_torch_exp3['X_train_tt'].shape[0]/mat_torch_exp3['X_eval_tt'].shape[0],1))
    print('')

    path_plots = os.path.join(os.getcwd(), '04_Training\\plots')
    plot_datasplit(mat_torch_exp1, mat_torch_exp2, mat_torch_exp3, save_path = path_plots)

    return mat_torch_exp1, mat_torch_exp2, mat_torch_exp3, mat_torch_MoE

# ------------------------------------------------- Class Optimiser Switch------------------------------------------------------------

class Adam_LBFGS(torch.optim.Optimizer):
    """
    Switches from Adam to LBFGS after *switch_step* optimisation steps.
    """

    def __init__(self,
                 params,
                 switch_step,
                 lbfgs_params=None,
                 adam_hyper={"lr": 1e-3, "betas": (0.9, 0.99)},
                 lbfgs_hyper={"lr": 1., "max_iter": 20, "history_size": 100}):
        self._params = list(params)
        self.switch_step = switch_step
        self.adam = torch.optim.AdamW(self._params, **adam_hyper)
        self.lbfgs = torch.optim.LBFGS(self._params if lbfgs_params is None else lbfgs_params, **lbfgs_hyper)
        defaults = {}
        super().__init__(self._params, defaults)
        self.state["step"] = 0
        self.state["using_lbfgs"] = False

    # ------------------------------------------------------------
    def step(self, closure):  # type: ignore[override]
        """closure() should *zero* grads, compute loss, call backward, then return loss"""
        self.state["step"] += 1

        # —— Phase I : Adam ————————————————————————————————
        if not self.state["using_lbfgs"]:
            loss, loss_logs = closure()  # gradient already computed inside closure
            self.adam.step()
            if self.state["step"] >= self.switch_step:
                print(f"[Adam_LBFGS] Switching to LBFGS at optimiser step {self.state['step']}")
                self.state["using_lbfgs"] = True
            return loss, loss_logs

        # —— Phase II : LBFGS ————————————————————————————————
        def lbfgs_closure():
            return closure()[0]  # LBFGS calls this several times internally
        loss = self.lbfgs.step(lbfgs_closure)
        return loss, closure(backward = False)[1]

    # Convenience
    def using_lbfgs(self):
        return self.state["using_lbfgs"]

# --------------------------------------------------Training with lightning-------------------------------------------------------------

def trainer_instance(inp, n_patience, log_every = 8, gpu_id = None, inp_all = None):
    raise RuntimeWarning('The pytorch lighnting version of this code is outdated. Please use simple_m = True')

    '''
    Defines the trainers and corresponding loggers and callbacks.
    inp             (dict)      inp: relevant model parameters
    n_patience      (int)       patience of trainer for early stopping
    log_every       (int)       every how many steps the trainer should log
    inp_dict        (dict)      only needed in the case of 'Split_Net_all': contains a dict of inp-dicts for every net.
    '''

    logger_csv = CSVLogger("logs", name = "train_log")   
    LR_callback = LearningRateMonitor(logging_interval='epoch')


    # define trainers
    if not inp['Split_Net'] and not inp['Split_Net_all']:
        logger_wandb = WandbLogger()
        early_stop_callback = EarlyStopping(monitor= "val_loss", mode = "min", patience=n_patience)
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode = "min", filename='best_model')       # model-{epoch:02d}-{val_loss:.2f}
        trainer = L.Trainer(max_epochs=inp['num_epochs'], 
                            devices = [0], accelerator = 'gpu',
                            logger = [logger_csv, logger_wandb], 
                            callbacks=[checkpoint_callback, LR_callback, early_stop_callback],
                            log_every_n_steps=log_every)
        
        trainer_m, trainer_b, trainer_s = None, None, None
        trainer_dict = {
         "standard": trainer,
         "m": trainer_m,
         "b": trainer_b,
         "s": trainer_s,
        }
        
    elif inp['Split_Net']:
        # m-net
        logger_wandb_m = WandbLogger(name = 'membrane')
        checkpoint_callback_m = ModelCheckpoint(monitor="val_loss", mode = "min", filename='best_model_m')
        early_stop_callback_m = EarlyStopping(monitor= "val_loss", mode = "min", patience=n_patience)
        trainer_m = L.Trainer(max_epochs=inp['num_epochs'], 
                              devices = [0], accelerator = 'gpu',
                              logger = [logger_csv, logger_wandb_m], 
                              callbacks=[checkpoint_callback_m, LR_callback, early_stop_callback_m],
                              gradient_clip_val=inp['gradient_clip_val'],
                              log_every_n_steps=log_every)

        # b-net
        logger_wandb_b = WandbLogger(name = 'bending')
        checkpoint_callback_b = ModelCheckpoint(monitor="val_loss", mode = "min", filename='best_model_b')
        early_stop_callback_b = EarlyStopping(monitor= "val_loss", mode = "min", patience=n_patience)
        trainer_b = L.Trainer(max_epochs=inp['num_epochs'], 
                              devices = [0], accelerator = 'gpu',
                              logger = [logger_csv, logger_wandb_b], 
                              callbacks=[checkpoint_callback_b, LR_callback, early_stop_callback_b],
                              gradient_clip_val=inp['gradient_clip_val'],
                              log_every_n_steps=log_every)
        
        # s-net
        logger_wandb_s = WandbLogger(name = 'shear')
        checkpoint_callback_s = ModelCheckpoint(monitor="val_loss", mode = "min", filename='best_model_s')
        early_stop_callback_s = EarlyStopping(monitor= "val_loss", mode = "min", patience=n_patience)
        trainer_s = L.Trainer(max_epochs=inp['num_epochs'], 
                              devices = [0], accelerator = 'gpu',
                              logger = [logger_csv, logger_wandb_s], 
                              callbacks=[checkpoint_callback_s, LR_callback, early_stop_callback_s],
                              gradient_clip_val=inp['gradient_clip_val'],
                              log_every_n_steps=log_every)
        
        trainer = None
        trainer_dict = {
         "standard": trainer,
         "m": trainer_m,
         "b": trainer_b,
         "s": trainer_s,
        }
    
    elif inp['Split_Net_all']: 
        trainer_dict = {}
        checkpoint_callback = {}
        early_stop_callback = {}
        logger_wandb = {}
        
        if torch.cuda.is_available():
            for i in range(8):
                logger_wandb[str(i)] = WandbLogger(name = 'model'+str(i))
                checkpoint_callback[str(i)] = ModelCheckpoint(monitor="val_loss", mode = "min", filename='best_model_'+str(i))
                early_stop_callback[str(i)] = EarlyStopping(monitor= "val_loss", mode = "min", patience=n_patience)
                trainer_dict[str(i)] = L.Trainer(max_epochs=inp_all[str(i)]['num_epochs'], 
                                                # fast_dev_run=True,
                                                devices = [gpu_id], accelerator = "gpu", # strategy = "ddp_spawn",
                                                logger = [logger_csv, logger_wandb[str(i)]], 
                                                callbacks=[checkpoint_callback[str(i)], LR_callback, early_stop_callback[str(i)]],
                                                gradient_clip_val=inp_all[str(i)]['gradient_clip_val'],
                                                log_every_n_steps=log_every)
        else:
            for i in range(8):
                logger_wandb[str(i)] = WandbLogger(name = 'model'+str(i))
                checkpoint_callback[str(i)] = ModelCheckpoint(monitor="val_loss", mode = "min", filename='best_model_'+str(i))
                early_stop_callback[str(i)] = EarlyStopping(monitor= "val_loss", mode = "min", patience=n_patience)
                trainer_dict[str(i)] = L.Trainer(max_epochs=inp_all[str(i)]['num_epochs'], 
                                                logger = [logger_csv, logger_wandb[str(i)]], 
                                                callbacks=[checkpoint_callback[str(i)], LR_callback, early_stop_callback[str(i)]],
                                                gradient_clip_val=inp_all[str(i)]['gradient_clip_val'],
                                                log_every_n_steps=log_every)

    return trainer_dict


# -------------------------------------------------- Training with pytorch ------------------------------------------------------------

def simple_train(inp, model_dict, mat_data_TrainEvalTest, best_model_path, no = ''):
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if no == 'exp1': 
        model = model_dict['exp1']
    elif no == 'exp2':
        model = model_dict['exp2']
    elif no == 'exp3':
        model = model_dict['exp3']
    else:
        model = model_dict['standard']
    if inp['batch_size'] is None: 
        no_switch_step = int(inp['switch_step_percentage']*inp['num_epochs'])
    else: 
        no_switch_step = int(inp['switch_step_percentage']*inp['num_epochs']*int(mat_data_TrainEvalTest['X_train_tt'].shape[0]/inp['batch_size']))
    optimizer = Adam_LBFGS(model.parameters(), no_switch_step)
    # optimizer = torch.optim.Adam(model.parameters(), lr=inp['learning_rate'])
    lossFn = loss_mapping[inp['loss_type']]()
    if inp['lr_scheduler'] == 'standard':
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer.adam, step_size=100, gamma=0.99)
    elif inp['lr_scheduler'] == 'plateau':
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.adam, 'min')
    if inp['w_s'] == 'max':
        w_s = 'max'
    else:
        w_s = ast.literal_eval(inp['w_s'])
    model.to(device)

    MAXEPOCHS = inp['num_epochs']

    if inp['kfold']:
        raise RuntimeWarning('kfold is outdated. Please use kfold = \'False\'')
        kf = KFold(n_splits = 5, shuffle=True, random_state = SEED)
        X_kf = torch.concat((mat_data_TrainEvalTest['X_train_tt'], mat_data_TrainEvalTest['X_eval_tt']), axis=0).cpu().numpy()
        y_kf = torch.concat((mat_data_TrainEvalTest['y_train_tt'], mat_data_TrainEvalTest['y_eval_tt']), axis=0).cpu().numpy()

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_kf)):
            print(f'Starting fold {fold+1}/{5}')
            X_train, X_val = X_kf[train_idx], X_kf[val_idx]
            y_train, y_val = y_kf[train_idx], y_kf[val_idx]

            X0 = torch.tensor(X_train, dtype=torch.float32).to(device)
            y0 = torch.tensor(y_train, dtype=torch.float32).to(device)
            X0e = torch.tensor(X_val, dtype=torch.float32).to(device)
            y0e = torch.tensor(y_val, dtype=torch.float32).to(device)

            simple_train_aux_(X0, y0, X0e, y0e, model, optimizer, scheduler, lossFn, int(MAXEPOCHS/5), w_s, best_model_path, inp, no)


    elif not inp['kfold'] and inp['batch_size'] is None:
        X1 = mat_data_TrainEvalTest['X_train_tt'].to(device)
        X1e = mat_data_TrainEvalTest['X_eval_tt'].to(device)
        y1 = mat_data_TrainEvalTest['y_train_tt'].to(device)
        y1e = mat_data_TrainEvalTest['y_eval_tt'].to(device)
        train_loader = None
        eval_loader = None
        model = simple_train_aux_(X1, y1, X1e, y1e, train_loader, eval_loader, model, optimizer, scheduler, lossFn, MAXEPOCHS, w_s, best_model_path, inp, no)

    elif inp['batch_size'] is not None:
        batch_size = inp['batch_size']  # Set default batch size
        train_dataset = TensorDataset(mat_data_TrainEvalTest['X_train_tt'], mat_data_TrainEvalTest['y_train_tt'])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_dataset = TensorDataset(mat_data_TrainEvalTest['X_eval_tt'], mat_data_TrainEvalTest['y_eval_tt'])
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle = True)
        X1, y1 = None, None
        X1e = mat_data_TrainEvalTest['X_eval_tt'].to(device)
        y1e = mat_data_TrainEvalTest['y_eval_tt'].to(device)
        model = simple_train_aux_(X1, y1, X1e, y1e, train_loader, eval_loader, model, optimizer, scheduler, lossFn, MAXEPOCHS, w_s, best_model_path, inp, no)

    return model



def simple_train_aux_(X_train, y_train, X_eval, y_eval, train_loader, eval_loader, model, optimizer, scheduler, lossFn, MAXEPOCHS, w_s, best_model_path, inp, no):
    best_val_loss = float('inf')
    for epoch in range(MAXEPOCHS):
        # Training (batch size = sample size)
        model.train()

        if train_loader is not None:
            batch = 0
            epoch_loss = 0
            for X_train_batch, y_train_batch in train_loader:
                X_torch, sig_torch = X_train_batch.to(device), y_train_batch.to(device)

                def closure(backward = True):
                    optimizer.zero_grad()

                    if inp['Sobolev']:
                        if not inp['DeepONet']:
                            if 'energy' not in inp or not inp['energy']:
                                custom_crit = CustomLosses(inp, model, None)
                                loss, loss_logs = custom_crit.Sobolev_CustomLoss(X_torch, sig_torch)
                            elif inp['energy']:
                                custom_crit = CustomLosses(inp, model, None)
                                loss, loss_logs = custom_crit.Sobolev_Energy_CustomLoss(X_torch, sig_torch)
                        else: 
                            custom_crit = CustomLosses(inp, None, model)
                            loss, loss_logs = custom_crit.Sobolev_CustomLoss(X_torch, sig_torch)
                        # loss, loss_logs = custom_crit.comp_SobolevCustom(X_torch, sig_torch)
                    elif inp['cVAE']:
                        pred, decoded, mean, logvar = model(X_torch, sig_torch[:,0:8])
                        custom_crit = CustomLosses(inp, None, None)
                        loss, loss_logs = custom_crit.cVAE_Loss(X_torch, decoded, sig_torch[:,0:8], pred, mean, logvar)
                    else:
                        if not inp['DeepONet']:
                            pred = model(X_torch)
                        elif inp['DeepONet']:
                            if 'num_trunk' in inp:
                                num_trunk = inp['num_trunk']
                            else: 
                                num_trunk = 1
                            pred = model(X_torch[:,0:8], X_torch[:,8:].reshape(-1,num_trunk))
                        loss = lossFn(pred, sig_torch[:,0:inp['out_size']])
                        loss_logs = {'train_loss': loss}
                    if backward:
                        loss.backward()
                    return loss, loss_logs
        
                # Take steps
                optimizer.zero_grad()
                loss, loss_logs = optimizer.step(closure)

                epoch_loss  += loss.item()

                if inp['lr_scheduler'] == 'standard':
                    scheduler.step()
                elif inp['lr_scheduler'] == 'plateau':
                    pass

                # Logs
                wandb.log(loss_logs)
                wandb.log({'Learning Rate': scheduler.get_last_lr()[0]})
                wandb.log({'epoch': epoch})
                batch = batch+1

                if epoch % 100 == 0 or batch % int(len(train_loader)/2) == 0:
                    print(f"Epoch [{epoch}/{MAXEPOCHS}] and Batch [{batch}/{len(train_loader)}] with LR {scheduler.get_last_lr()[0]:.2e}: \tLoss: {loss.item():.4e}")
                if batch == int(len(train_loader))-1: 
                    avg_train_loss = epoch_loss / batch
                    wandb.log({'Average train loss per epoch': avg_train_loss})
                
        else:
            def closure(backward = True):
                optimizer.zero_grad()
                X_torch = X_train.to(device)
                sig_torch = y_train.to(device)

                if inp['Sobolev']:
                    if not inp['DeepONet']:
                        if ('energy' not in inp or not inp['energy']) and ('only_diag' not in inp or not inp['only_diag']):
                            custom_crit = CustomLosses(inp, model, None)
                            loss, loss_logs = custom_crit.Sobolev_CustomLoss(X_torch, sig_torch)
                        elif inp['energy']:
                            custom_crit = CustomLosses(inp, model, None)
                            loss, loss_logs = custom_crit.Sobolev_Energy_CustomLoss(X_torch, sig_torch)
                        elif inp['only_diag']:
                            custom_crit = CustomLosses(inp, model, None)
                            loss, loss_logs = custom_crit.Sobolev_DiagonalLoss(X_torch, sig_torch)
                    else: 
                        custom_crit = CustomLosses(inp, None, model)
                        loss, loss_logs = custom_crit.Sobolev_CustomLoss(X_torch, sig_torch)
                    # loss, loss_logs = custom_crit.comp_SobolevCustom(X_torch, sig_torch)
                elif inp['cVAE']:
                    pred, decoded, mean, logvar = model(X_torch, sig_torch[:,0:8])
                    custom_crit = CustomLosses(inp, None, None)
                    loss, loss_logs = custom_crit.cVAE_Loss(X_torch, decoded, sig_torch[:,0:8], pred, mean, logvar)
                else:
                    if not inp['DeepONet']:
                        pred = model(X_torch)
                    elif inp['DeepONet']:
                        if 'num_trunk' in inp:
                            num_trunk = inp['num_trunk']
                        else: 
                            num_trunk = 1
                        pred = model(X_torch[:,0:8], X_torch[:,8:].reshape(-1,num_trunk))
                    loss = lossFn(pred, sig_torch[:,0:inp['out_size']])
                    loss_logs = {'train_loss': loss}
                if backward:
                    loss.backward()
                return loss, loss_logs
    
            # Take steps
            optimizer.zero_grad()
            loss, loss_logs = optimizer.step(closure)
            if inp['lr_scheduler'] == 'standard':
                scheduler.step()
            elif inp['lr_scheduler'] == 'plateau':
                pass

            # Logs
            wandb.log(loss_logs)
            wandb.log({'Learning Rate': scheduler.get_last_lr()[0]})
            wandb.log({'epoch': epoch})

            if epoch % 100 == 0:
                print(f"Epoch [{epoch}/{MAXEPOCHS}] with LR {scheduler.get_last_lr()[0]:.2e}: \tLoss: {loss.item():.4e}")


        # Evaluation (once every epoch)
        if eval_loader is not None:
            epoch_val_loss = 0
            val_batch = 0

            for X_eval_batch, y_eval_batch in eval_loader:
                X_eval, y_eval = X_eval_batch.to(device), y_eval_batch.to(device)
                model.eval()
                with torch.no_grad():
                    sig_torch_val = y_eval
                    if not inp['DeepONet'] and not inp['cVAE']:
                        pred_val = model(X_eval)
                    elif inp['DeepONet']:
                        if 'num_trunk' in inp:
                            num_trunk = inp['num_trunk']
                        else: 
                            num_trunk = 1
                        pred_val = model(X_eval[:,0:8], X_eval[:,8:].reshape(-1,num_trunk))
                    elif inp['cVAE']:
                        pred_val, decoded, mean, logvar = model(X_eval, sig_torch_val[:,0:8])
                                    
                    
                    if not inp['Sobolev'] and not inp['cVAE']:
                        val_loss = lossFn(pred_val, sig_torch_val[:,0:inp['out_size']])
                    elif inp['cVAE']:
                        custom_crit = CustomLosses(inp, None, None)
                        val_loss, loss_logs = custom_crit.cVAE_Loss(X_eval, decoded, sig_torch_val[:,0:8], pred_val, mean, logvar) 
                    elif inp['Sobolev']: 
                        if not inp['DeepONet']:
                            if 'energy' not in inp or not inp['energy']:
                                custom_crit = CustomLosses(inp, model, None)
                                val_loss, loss_logs = custom_crit.Sobolev_CustomLoss(X_eval, sig_torch_val)
                            elif inp['energy']:
                                custom_crit = CustomLosses(inp, model, None)
                                val_loss, loss_logs = custom_crit.Sobolev_Energy_CustomLoss(X_eval, sig_torch_val)
                        else: 
                            custom_crit = CustomLosses(inp, None, model)
                            val_loss, loss_logs = custom_crit.Sobolev_CustomLoss(X_eval, sig_torch_val)          
                    wandb.log({"val_loss": val_loss})
                    epoch_val_loss  += val_loss.item()
                    val_batch += 1

                    if epoch % 100 == 0 or val_batch % int(len(eval_loader)/2) == 0:
                        print(f"Epoch [{epoch}/{MAXEPOCHS}] and Batch [{val_batch}/{len(eval_loader)}]: \tValidation Loss: {val_loss.item():.4e}")

                if val_batch == int(len(train_loader))-1: 
                    avg_val_loss = epoch_val_loss / val_batch
                    wandb.log({'Average evaluation loss per epoch': avg_val_loss})

        else: 
            model.eval()
            with torch.no_grad():
                sig_torch_val = y_eval
                if not inp['DeepONet'] and not inp['cVAE']:
                    pred_val = model(X_eval)
                elif inp['DeepONet']:
                    if 'num_trunk' in inp:
                        num_trunk = inp['num_trunk']
                    else: 
                        num_trunk = 1
                    pred_val = model(X_eval[:,0:8], X_eval[:,8:].reshape(-1,num_trunk))
                elif inp['cVAE']:
                    pred_val, decoded, mean, logvar = model(X_eval, sig_torch_val[:,0:8])
                                
                
                if not inp['Sobolev'] and not inp['cVAE']:
                    val_loss = lossFn(pred_val, sig_torch_val[:,0:inp['out_size']])
                elif inp['cVAE']:
                    custom_crit = CustomLosses(inp, None, None)
                    val_loss, loss_logs = custom_crit.cVAE_Loss(X_eval, decoded, sig_torch_val[:,0:8], pred_val, mean, logvar) 
                elif inp['Sobolev']: 
                    if not inp['DeepONet']:
                        if 'energy' not in inp or not inp['energy']:
                            custom_crit = CustomLosses(inp, model, None)
                            val_loss, loss_logs = custom_crit.Sobolev_CustomLoss(X_eval, sig_torch_val)
                        elif inp['energy']:
                            custom_crit = CustomLosses(inp, model, None)
                            val_loss, loss_logs = custom_crit.Sobolev_Energy_CustomLoss(X_eval, sig_torch_val)
                    else: 
                        custom_crit = CustomLosses(inp, None, model)
                        val_loss, loss_logs = custom_crit.Sobolev_CustomLoss(X_eval, sig_torch_val)          
                wandb.log({"val_loss": val_loss})
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wandb.log({"best_val_loss": best_val_loss})
            save_best_model(model, epoch, no, val_loss, save_dir=best_model_path)

        if inp['lr_scheduler'] == 'standard':
            pass
        elif inp['lr_scheduler'] == 'plateau':
            scheduler.step(val_loss)

    return model



def save_best_model(model, epoch, no, val_loss, save_dir):
     # save current model
     os.makedirs(save_dir, exist_ok=True)
     model_filename =f'best_trained_model_{no}_{epoch}.pt'
     torch.save(model.state_dict(), os.path.join(save_dir, model_filename))
     if epoch > 200:
            print(f'Best model saved at epoch {epoch} with validation loss: {val_loss:.4f}')
     
     # remove old models
     model_pattern = re.compile(fr"best_trained_model_{no}_(\d+)\.pt")
     
     for filename in os.listdir(save_dir):
          match = model_pattern.match(filename)
          if match: 
               epoch_ = int(match.group(1))
               if epoch_ < epoch:
                    retries = 3
                    delay = 1
                    file_path = os.path.join(save_dir, filename)
                    os.remove(file_path)
                    '''for _ in range(retries):
                        try:
                            os.remove(file_path)
                        except PermissionError as e:
                            print('Permission error during deletion of {file_path}')
                            time.sleep(delay)'''

     
     return
