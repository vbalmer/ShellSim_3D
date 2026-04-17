import pickle
import torch
from FFNN_class_light import *
from data_work import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.utils.data import TensorDataset, DataLoader




def load_data(path: str, only_test = True, add_path = None):
    '''
    Loads saved data from "data work" before training. Model needs to be loaded separately!
    path:       (str)   Path where model and data is saved
    only_test:  (bool)  put false if it should be logged directly after executing training
                        put true if it is logged standalone (without prior training)
    load_model  (bool)  if false: does not load model but just the data (inp, mat_data_[...])
    add_path    (str)   Path to specific old version of model and data (e.g. 'data_20241013_2125_casexx')
    '''

    if add_path == None:
        save_path = os.path.join(path, 'new_data')
    else: 
        aux_ =  'new_data\\' + add_path
        save_path = os.path.join(path, aux_)
    
    with open(os.path.join(save_path, 'inp.pkl'),'rb') as handle:
        inp = pickle.load(handle)
    with open(os.path.join(save_path, 'mat_data_stats.pkl'),'rb') as handle:
        mat_data_stats = pickle.load(handle)
    with open(os.path.join(save_path, 'mat_data_TrainEvalTest.pkl'),'rb') as handle:
        mat_data_TrainEvalTest = pickle.load(handle)
    with open(os.path.join(save_path, 'mat_data_np_TrainEvalTest.pkl'),'rb') as handle:
            mat_data_np_TrainEvalTest = pickle.load(handle)
    

    inp1, inp2 = None, None
    
    if only_test:
        run = init_wandb(inp, inp1, inp2, 'ShellSim_FFNN_v1_onlyTest')
    
    stats_y_test = mat_data_stats['stats_y_test']
    stats_X_test = mat_data_stats['stats_X_test']
    stats_y_train = mat_data_stats['stats_y_train']
    stats_X_train = mat_data_stats['stats_X_train']


    data_model = {
        'eval_model': None,
        'inp': inp,
        'inp1': inp1, 
        'inp2': inp2, 
        'mat_data_stats': mat_data_stats,
        'mat_data_TrainEvalTest': mat_data_TrainEvalTest,
        'mat_data_np_TrainEvalTest': mat_data_np_TrainEvalTest,
        # 'test_loader': test_loader,
        'stats_y_test': stats_y_test, 
        'stats_X_test': stats_X_test,
        'stats_y_train': stats_y_train, 
        'stats_X_train': stats_X_train,
    }

    return data_model


def predict_D(data_model:dict, transf_type: str,sc = False, dn = False):
    '''
    Carries out prediction of stiffness matrix D based on given data loader and eval_model
    
    
    Relevant variables in data_model
    test_loader:        (DataLoader)        Test data (or any other data for which prediction should be carried out)
    eval_model:         (FFNN)              FFNN Model (in eval mode)
    inp:                (dict)              Configuration / Specific architecture of FFNN
    stats_y_test:       (dict)              Statistics corresponding to test data
    '''
    
    # make prediction of D (via derivatives) and transform back to original number format and to numpy
    stats_y_train = data_model['stats_y_train']
    out_shape = data_model['inp']['out_size']
    
    if transf_type == 'mixed':
        transf_type_list_x = ['x-std']*3+['x-range']*3+['x-std']*3
        transf_type_list_y = ['y-std']*3+['y-range']*3+['y-std']*66
    elif transf_type == 'st-stitched':
        transf_type_list_x = ['x-std']*data_model['mat_data_np_TrainEvalTest']['X_test'].shape[1]
        transf_type_list_y = ['y-std']*out_shape+['y-st-stitched']*out_shape*out_shape
    else: 
        transf_type_list_x = ['x-'+transf_type]*data_model['mat_data_np_TrainEvalTest']['X_test'].shape[1]
        transf_type_list_y = ['y-'+transf_type]*data_model['mat_data_np_TrainEvalTest']['y_test'].shape[1]

    X_test_t = transform_data(data_model['mat_data_np_TrainEvalTest']['X_test'], data_model['mat_data_stats'], forward=True, type=transf_type_list_x, sc=sc, dn=dn)
    
    D_pred = np.zeros((X_test_t.shape[0], out_shape, out_shape))

    if data_model['inp']['batch_size'] is not None and X_test_t.shape[0]>100e3:
        batch_size = int((data_model['inp']['batch_size'])/1000)
        print(f'Reduced batch size for testing stiffness matrix to {batch_size}. Note that this will take significantly more time.')
        test_dataset = TensorDataset(torch.Tensor(X_test_t).to(device))
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    else: 
        X_test_tt = torch.Tensor(X_test_t).to(device)
        X_test_tt.requires_grad = True


    # create_graph can be off (no gradient of D required at inference)
    # J = torch.cat([torch.autograd.functional.jacobian(data_model['eval_model'], X_test_tt[i:i+1], create_graph = False) for i in range(len(X_test_tt))], dim=0)[:,:,0,:]
    if not data_model['inp']['DeepONet'] and ('energy' not in data_model['inp'] or not data_model['inp']['energy']):
        if data_model['inp']['batch_size'] is not None and X_test_t.shape[0]>100e3:             
            J = []
            batch_no = 0
            for (X_test_tt, ) in test_loader:
                X_test_tt = X_test_tt.cpu()
                model_cpu = data_model['eval_model'].to(torch.device('cpu'))
                try:
                    for x in X_test_tt:
                        x_local = x.unsqueeze(0).detach().clone().requires_grad_(True)
                        j = torch.autograd.functional.jacobian(model_cpu.forward, x_local)
                        J.append(j)
                        del j
                        # torch.cuda.empty_cache()
                    batch_no+=1
                    if batch_no % 5000  == 0: 
                        print(f'Calculated batch_no {batch_no}/{len(test_loader)}')
                except torch.cuda.OutOfMemoryError:
                    print(f'Stopped at batch_no {batch_no}/{len(test_loader)}')
                    break
            D_t = np.array(J)
        else: 
            # standard case, without batching
            J = vmap(torch.func.jacrev(data_model['eval_model'].forward), randomness='different')(X_test_tt)
            D_t = J.cpu().detach().numpy()
        
    elif data_model['inp']['DeepONet']:
        if data_model['inp']['batch_size'] is not None: 
                raise UserWarning('This is not yet implemented for batchsize != None.')
        if 'num_trunk' in data_model['inp']:
            num_trunk = data_model['inp']['num_trunk']
        else: 
            num_trunk = 1
        J = vmap(torch.func.jacrev(data_model['eval_model'].forward), randomness='different')(X_test_tt[:,0:8], X_test_tt[:,8:].reshape(-1,num_trunk))
    elif data_model['inp']['energy']:
        if data_model['inp']['batch_size'] is not None: 
                raise UserWarning('This is not yet implemented for batchsize != None.')
        H = vmap(torch.func.jacrev(torch.func.jacrev(data_model['eval_model'].forward)), randomness='different')(X_test_tt)
        J = H[:,0,:out_shape,:out_shape]
    
    
    if data_model['inp']['batch_size'] is not None and X_test_t.shape[0]>100e3:
        D_t = D_t[:,0,:out_shape,0,:out_shape]
    elif len(D_t.shape)>3:
        D_t = D_t[:,0,:out_shape,:out_shape]
    else: 
        D_t = D_t[:, :out_shape, :out_shape]
    
    # Transform back D
    for idx in range(X_test_t.shape[0]):
        D_t_i = D_t[idx,:,:].reshape((1,out_shape*out_shape))
        added_transform = np.concatenate((np.zeros((1,out_shape)), D_t_i), axis=1)
        added_transform_ = transform_data(added_transform, data_model['mat_data_stats'], forward = False, type=transf_type_list_y, sc=sc, dn = dn)
        D_pred_i = added_transform_[:,out_shape:]
        D_pred[idx,:,:] = D_pred_i.reshape((out_shape,out_shape))


    # format labels of simulation correctly
    D_sim = data_model['mat_data_np_TrainEvalTest']['y_test'][:, out_shape:]
    D_sim = D_sim.reshape((D_sim.shape[0], out_shape, out_shape))
     
    plot_data = {
         'D_sim': D_sim,
         'D_t': D_t,
         'D_pred': D_pred,
    }
    return plot_data


def predict_pure_D(data_model:dict, inp:dict, model_test_dict, transf_type:str, sc=False, dn=False):
    '''
    predict only D --> have 38 values instead of 64 (only nonzero values)
    '''
    inp_shape = data_model['mat_data_np_TrainEvalTest']['X_test'].shape[1]
    out_shape = data_model['mat_data_np_TrainEvalTest']['y_test'].shape[1]

    if transf_type == 'mixed':
        transf_type_list_x = ['x-std']*3+['x-range']*3+['x-std']*3
        transf_type_list_y = ['y-std']*3+['y-range']*3+['y-std']*66
    elif transf_type == 'st-stitched':
        transf_type_list_x = ['x-std']*inp_shape
        transf_type_list_y = ['y-std']*8+['y-st-stitched']*64
    else: 
        transf_type_list_x = ['x-'+transf_type]*inp_shape
        transf_type_list_y = ['y-'+transf_type]*out_shape

    X_test_t = transform_data(data_model['mat_data_np_TrainEvalTest']['X_test'], data_model['mat_data_stats'], forward = True, type = transf_type_list_x, sc=sc, dn=dn)
    X_train_t = transform_data(data_model['mat_data_np_TrainEvalTest']['X_train'], data_model['mat_data_stats'], forward = True, type = transf_type_list_x, sc=sc, dn=dn)

    if inp['simple_m']:
        if not inp['DeepONet'] and not inp['MoE']:
            model_test_dict['standard'].to(device)
            predictions_t = model_test_dict['standard'](torch.Tensor(X_test_t).to(device))
            predictions_t_train = model_test_dict['standard'](torch.Tensor(X_train_t).to(device))
        elif inp['DeepONet']: 
            raise RuntimeError('This is not implemented yet')
        elif inp['MoE']: 
            raise RuntimeError('This is not implemented yet')
    else:
        raise RuntimeWarning('This should not be used anymore. Please switch to simple_m = True')
    
    if not inp['MoE']:
        predictions = transform_data(predictions_t.cpu().detach().numpy(), data_model['mat_data_stats'], forward = False, type = transf_type_list_y, sc=sc, dn=dn)
        predictions_train = transform_data(predictions_t_train.cpu().detach().numpy(), data_model['mat_data_stats'], forward = False, type = transf_type_list_y, sc=sc, dn=dn)
        test_labels_t = transform_data(data_model['mat_data_np_TrainEvalTest']['y_test'], data_model['mat_data_stats'], forward=True, type = transf_type_list_y, sc=sc, dn=dn)
        test_labels = data_model['mat_data_np_TrainEvalTest']['y_test']
        train_labels_t = transform_data(data_model['mat_data_np_TrainEvalTest']['y_train'], data_model['mat_data_stats'], forward=True, type = transf_type_list_y, sc=sc, dn=dn)
        train_labels = data_model['mat_data_np_TrainEvalTest']['y_train']
    elif inp['MoE']:
        raise RuntimeError('This is not implemented yet')

    # Assemble the stiffness matrices
    # Values which should be zero are fixed to zero here
    D_sim_t, D_sim = np.zeros((X_test_t.shape[0], 8, 8)), np.zeros((X_test_t.shape[0], 8, 8))
    D_pred_t, D_pred = np.zeros((X_test_t.shape[0], 8, 8)), np.zeros((X_test_t.shape[0], 8, 8))

    D_sim[:,:6,:6], D_sim_t[:,:6,:6] = test_labels[:,:36].reshape((-1,6,6)), test_labels_t[:,:36].reshape((-1,6,6))
    D_sim[:,6,6], D_sim_t[:,6,6] = test_labels[:,36].reshape((-1,)), test_labels_t[:,36].reshape((-1,))
    D_sim[:,7,7], D_sim_t[:,7,7] = test_labels[:,37].reshape((-1,)), test_labels_t[:,37].reshape((-1,))

    
    predictions_t = predictions_t.cpu().detach().numpy()
    D_pred[:,:6,:6], D_pred_t[:,:6,:6] = predictions[:,:36].reshape((-1,6,6)), predictions_t[:,:36].reshape((-1,6,6))
    D_pred[:,6,6], D_pred_t[:,6,6] = predictions[:,36].reshape((-1,)), predictions_t[:,36].reshape((-1,))
    D_pred[:,7,7], D_pred_t[:,7,7] = predictions[:,37].reshape((-1,)), predictions_t[:,37].reshape((-1,))


    plot_data_d_pure = {
            'D_sim_t': D_sim_t,
            'D_pred_t': D_pred_t,
            'D_sim': D_sim,
            'D_pred': D_pred,


            # 'all_train_labels_t': train_labels_t,
            # 'all_predictions_t_train': predictions_t_train.cpu().detach().numpy(),
            # 'all_train_labels': train_labels,
            # 'all_predictions_train': predictions_train,
            }



    return plot_data_d_pure



'''-----------------------------------------------------NOT IN USE ----------------------------------------------------------------------------'''

def predict_sig_D(random_eps_h:np.array, path_data_name:str, lit_model_path: str, stats: str, transf_type: str, sc = False, dn=False):
    '''
    Predicts output sig_h and D, based on random_eps_h input and the trained model given in path_data_name

    random_eps_h    (np.array)      input vector: 8 eps + t --> shape: (1,9)
    path_data_name  (str)           path to data for trained model
    lit_model_path  (str)           path to trained model (lightning)
    stats           (str)           statistics with which normalisation is carried out, can be either 'train' or 'test'
    stats_id:       (str)           If stats_id = stitched: transform D with stats of sig, eps
                                    If stats_id = direct: transform D with stats of D

    Output: 
    mat_pred         (dict)         Containing 'D_t', 'D_pred' and 'sig_h'

    '''
    random_sig_h = np.zeros((1,8))
    # Plot to check:
    # nbins = 50
    # histogram(random_eps_h[0:8], random_sig_h, random_eps_h.shape[0], nbins, 'eps')
    # print(random_eps_h[0:8])

    # Load statistics information
    with open(os.path.join(path_data_name, 'mat_data_stats.pkl'),'rb') as handle:
        mat_data_stats = pickle.load(handle)
    with open(os.path.join(path_data_name, 'inp.pkl'),'rb') as handle:
        inp = pickle.load(handle)
    
    if stats == 'train':
        stats_y = mat_data_stats['stats_y_train']
        stats_y_sig = {key: value[0:8] for key, value in stats_y.items()}
        stats_X = mat_data_stats['stats_X_train']
    elif stats =='test':
        stats_y = mat_data_stats['stats_y_test']
        stats_y_sig = {key: value[0:8] for key, value in stats_y.items()}
        stats_X = mat_data_stats['stats_X_test']

    # Transform into normalised coordinates
    transf_type_list_x = ['x-'+transf_type]*9
    transf_type_list_y = ['y-'+transf_type]*72
    X_depl_t = transform_data(random_eps_h, mat_data_stats, forward = True, type = transf_type_list_x, sc=sc, dn=dn)
    if X_depl_t.any() > 10 or X_depl_t.any() < -10:
        raise Exception("The normalisation yielded instable results. Please check input data")

    # Create dataset in correct format
    X_depl_tt = torch.from_numpy(X_depl_t)
    X_depl_tt = X_depl_tt.type(torch.float32)
    Y_depl_tt = torch.from_numpy(random_sig_h)
    Y_depl_tt = Y_depl_tt.type(torch.float32)
    
    # Plot to check
    # histogram_torch(deploy_loader, random_eps_h.shape[0], nbins, 'eps')

    # Load the trained model
    if inp['simple_m']:
        model = FFNN(inp)
        model.load_state_dict(torch.load(lit_model_path))
        model.eval()
        model_D = model
    else:
        if not inp['Split_Net']:
            model_sig = LitFFNN.load_from_checkpoint(lit_model_path)
            model_D = LitFFNN.load_from_checkpoint(lit_model_path['D'])
            model_sig.eval()
            model_D.eval()
        else: 
            model_m = LitFFNN.load_from_checkpoint(lit_model_path['m'])
            model_b = LitFFNN.load_from_checkpoint(lit_model_path['b'])
            model_s = LitFFNN.load_from_checkpoint(lit_model_path['s'])
            model_D = LitFFNN.load_from_checkpoint(lit_model_path['D'])
            model_m.eval()
            model_b.eval()
            model_s.eval()
            model_D.eval()


    # Make predictions of sig_generalised
    if inp['simple_m']:
        model.to(device)
        predictions_t = model(torch.Tensor(X_depl_t).to(device))
    else:
        if not inp['Split_Net']:
            predictions_t = model_sig(X_depl_tt.to(device))
        else: 
            predictions_t_m = model_m(torch.Tensor(np.hstack((X_depl_t[:,0:3], X_depl_t[:,8].reshape(-1, 1)))).to(device))
            predictions_t_b = model_b(torch.Tensor(np.hstack((X_depl_t[:,3:6], X_depl_t[:,8].reshape(-1, 1)))).to(device))
            predictions_t_s = model_s(torch.Tensor(np.hstack((X_depl_t[:,6:8], X_depl_t[:,8].reshape(-1, 1)))).to(device))
            predictions_t = torch.hstack((predictions_t_m, predictions_t_b, predictions_t_s))
    # Transform back sigma
    predictions_sig = transform_data(predictions_t.cpu().detach().numpy(), mat_data_stats, forward = False, type = transf_type_list_y, sc=sc, dn=dn)


    # Make predictions of D-matrix (i.e. the individual derivatives)
    D_pred = np.zeros((X_depl_t.shape[0], 8, 8))
    X_depl_tt.requires_grad = True
    J = torch.cat([torch.autograd.functional.jacobian(model_D, X_depl_tt[i:i+1], create_graph = True) for i in range(len(X_depl_tt))], dim=0)[:,:,0,:]
    D_t = J.cpu().detach().numpy()
    D_t = D_t[:, :8, :8]

    # Transform back D
    for idx in range(X_depl_t.shape[0]):
        random_add_values = np.zeros((1,8))
        D_pred_i = transform_data(np.concatenate(random_add_values, D_t[idx,:,:].reshape((1, 64)), axis = 1), mat_data_stats, forward = False, type = transf_type_list_y, sc=sc, dn=dn)
        D_pred[idx,:,:] = D_pred_i.reshape((8,8))

    # Collect relevant data
    mat_pred = {
        'D_pred': D_pred,
        'D_t': D_t,
        'sig_h': predictions_sig
    }

    return mat_pred

def D_an(eps:np.array, t: float):
    '''
    returns analytically calculated sig for given eps and t  (linear elastic)
    E, nu are assumed constant (as we assume analytical formulation for steel)
    eps expected in [-] or [1/mm]; t in [mm]
    D_analytical in [N/mm], [Nmm], [N]; sig in [N/mm] or [N]
    '''
    nu = 0.3
    E = 210000

    D_p = (E/(1+nu**2))*np.array([[1, nu, 0], 
                                [nu, 1, 0], 
                                [0, 0, 0.5 * (1 - nu)]])

    Dse = (5/6)*t*(2*E)/(4*(1+nu))

    D_an_1 = np.hstack([t*D_p, 0*D_p, np.zeros((3,2))])
    D_an_2 = np.hstack([0*D_p, (1/12)*(t**3)*D_p, np.zeros((3,2))])
    D_an_3 = np.hstack([np.zeros((2,3)), np.zeros((2,3)), np.array([[Dse, 0], [0, Dse]])])
    D_analytical = np.vstack([D_an_1, D_an_2, D_an_3])

    sig_analytical = np.matmul(D_analytical,eps)

    mat_analytical = {
        'D_a': D_analytical,
        'sig_a': sig_analytical
    }

    return mat_analytical


def inp_out_plt(eps: str, sig:str, data_model: dict, path: str, path_plots: str):
    '''
    Plots analytical function vs. NN function
    Uses deployment strategy, i.e. one prediction at a time, ignores batches.
    eps             (str)       x-axis variable to be plotted (can be one of the strings listed below)
    sig             (str)       y-axis variable to be plotted (can be one of the strings listed below)
    data_model      (dict)      optional data (eps, t and sig values of e.g. training or test data set) 
    path            (str)       path where model is saved
    save_path       (str)       path where figures are saved
    '''
    # Parameters to be plotted
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": 12,
        })
    
    eps_id = np.array(['eps_x', 'eps_y', 'eps_xy', 'chi_x', 'chi_y', 'chi_xy', 'gam_x', 'gam_y'])
    sig_id = np.array(['n_x', 'n_y', 'n_xy', 'm_x', 'm_y', 'm_xy', 'v_x', 'v_y'])
    eps_name = np.array([r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$', 
                         r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$',
                         r'$\gamma_{xz}$', r'$\gamma_{yz}$'])
    sig_name = np.array([r'$n_x$', r'$n_y$', r'$n_{xy}$', 
                         r'$m_x$', r'$m_y$', r'$m_{xy}$',
                         r'$v_{xz}$', r'$v_{yz}$'])

    eps_units = np.array(['[-]', '[-]', '[-]', '[1/cm]', '[1/cm]', '[1/cm]', '[-]', '[-]'])
    sig_units = np.array(['[MN/cm]', '[MN/cm]', '[MN/cm]', '[MN]', '[MN]', '[MN]', '[MN/cm]', '[MN/cm]'])
    
    # 0 - sort eps, t and sig data from simulation according to desired eps in ascending order
    if data_model is not None:
        data = np.concatenate((data_model['mat_data_np_TrainEvalTest']['X_train'][:,0:9], data_model['mat_data_np_TrainEvalTest']['y_train'][:,0:8]), axis = 1)
        data_sort = data[data[:, np.where(eps_id == eps)[0][0]].argsort()]
        num_rows = data_sort.shape[0]
    else:
        # here would be just random numbers at which the prediction and analytical solution should be evaluated
        # (if not the training  data points)
        # data_sort = ...
        # num_rows = ...
        pass
    
    path_data_name = os.path.join(path, 'new_data')
    # 1 - calculate corresponding predictions and analytical solutions
    sig_pred = np.zeros((num_rows, 8))
    sig_analytic = np.zeros((num_rows,8))
    for k in range(num_rows):
        mat_pred = predict_sig_D(data_sort[k,0:9].reshape((1,9)), path_data_name, 'train', 'direct')
        sig_pred[k,:] = mat_pred['sig_h']
        data_sort_analytic = data_sort.copy()
        data_sort_analytic[3:6] = data_sort_analytic[3:6]*10**(-1)      #adjust units of chi to 1/mm
        mat_a = D_an(data_sort_analytic[k,0:8], data_sort[k,8])
        sig_analytic[k,:] = mat_a['sig_a']
    # change units of analytical sigma to units of training / evaluation sigma
    sig_analytic[:,0:3], sig_analytic[:,6:8] = sig_analytic[:,0:3]*10**(-5), sig_analytic[:,6:8]*10**(-5)
    sig_analytic[:,3:6] = sig_analytic[:,3:6]*10**(-6)


    # 2 - select desired data for plot
    x_plt = data_sort[:,np.where(eps_id == eps)[0][0]]
    y_plt = sig_pred[:,np.where(sig_id == sig)[0][0]]
    y_plt_analytic = sig_analytic[:,np.where(sig_id == sig)[0][0]]
    y_plt_sim = data_sort[:,9+np.where(sig_id == sig)[0][0]]
    t_plt = data_sort[:,8]


    # 3 - find unique t-values
    # unique_t_sel = np.array([50, 60, 80, 140])
    unique_t = np.unique(t_plt)           # uncomment this row to see actual unique values
    unique_t_sel = unique_t[0:5]
    amt_t = len(unique_t_sel)
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, amt_t))

    # 4 - plot
    fig, ax = plt.subplots(1, 1, figsize = (5,5))
    # ax.plot(x_plt, y_plt, label = 'prediction', color = 'lightgrey')
    for j in range(amt_t):
        ax.plot(x_plt[t_plt == unique_t[j]], y_plt[t_plt == unique_t[j]], 
                label = 'prediction, t = ' + np.array2string(unique_t[j]) + 'mm', 
                color = colors[j])
        if j == amt_t-1:
            ax.scatter(x_plt[t_plt == unique_t[j]], y_plt_analytic[t_plt == unique_t[j]], label = 'analytic', color = colors[j], marker = 'x')
            ax.scatter(x_plt[t_plt == unique_t[j]], y_plt_sim[t_plt == unique_t[j]], label = 'simulation', color = 'grey', marker = 'o', facecolors='none')
        else: 
            ax.scatter(x_plt[t_plt == unique_t[j]], y_plt_analytic[t_plt == unique_t[j]], label = 'analytic', color = colors[j], marker = 'x')
            ax.scatter(x_plt[t_plt == unique_t[j]], y_plt_sim[t_plt == unique_t[j]], color = 'grey', marker = 'o', facecolors='none')
    ax.set_xlabel(eps_name[np.where(eps_id == eps)[0][0]]+' '+eps_units[np.where(eps_id == eps)[0][0]])
    ax.set_ylabel(sig_name[np.where(eps_id == eps)[0][0]]+' '+sig_units[np.where(sig_id == sig)[0][0]])
    fig.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
    # plt.show()

    plt.tight_layout()
    if path_plots is not None:
        filename = os.path.join(path_plots, 'inp-outp_'+eps+'_'+sig+'.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()
    wandb.log({"inp-outp_"+eps+'_'+sig: wandb.Image(filename)})
    return


def predict_sig_XGB(data_model: dict, trained_model:dict):
    '''
    Carries out prediction of sigma based on given data loader and eval_model
    
    
    Relevant variables in data_model
    test_loader:        (DataLoader)        Test data (or any other data for which prediction should be carried out)
    eval_model:         (FFNN)              FFNN Model (in eval mode)
    inp:                (dict)              Configuration / Specific architecture of FFNN
    stats_y_test:       (dict)              Statistics corresponding to test data
    '''
    
    # Create predictions and transform to numpy
    num_test_points = data_model['mat_data_np_TrainEvalTest']['X_test'].shape[0]
    print('Amount of test points', num_test_points)

    X_test_t = transform_data(data_model['mat_data_np_TrainEvalTest']['X_test'], data_model['stats_X_train'], forward = True)
    all_predictions_t = np.zeros((num_test_points, 8))
    for i in range(8):
        all_predictions_t[:,i] = trained_model[f'model_{i}'].predict(X_test_t)
    
    # Transform labels to normalised labels
    all_test_labels_t = transform_data(data_model['mat_data_np_TrainEvalTest']['y_test'][:,0:8], 
                                       {key: value[0:8] for key, value in data_model['stats_y_train'].items()}, 
                                       forward = True)

    # Transform data back to original scale
    all_test_labels = transform_data(all_test_labels_t, {key: value[0:8] for key, value in data_model['stats_y_train'].items()}, forward=False)
    all_predictions = transform_data(all_predictions_t, {key: value[0:8] for key, value in data_model['stats_y_train'].items()}, forward=False)
    
    plot_data = {
        'all_test_labels_t': all_test_labels_t,
        'all_predictions_t': all_predictions_t,
        'all_test_labels': all_test_labels,
        'all_predictions': all_predictions,
    }

    return plot_data
