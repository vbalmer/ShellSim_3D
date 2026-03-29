import os
from FFNN_class_light import *
import torch
import numpy
from data_work import transform_data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import pickle

from torch.utils.data import TensorDataset, DataLoader

def test_model_instance(inp, path, v_num, epoch):
    '''
    loads model from version v_num, creates model instance
    inp     (dict)       input vector with definitions of model architecture
    path    (str)        cwd
    v_num   (str)        str (!); version number of model to load
    epoch   (str)        epoch at which best model was saved
    '''
    if inp['simple_m']:
        model_path = os.path.join(path, 'new_data\\_simple_logs\\v_'+ v_num+'\\best_trained_model_'+epoch+'.pt')
    if inp['MoE-split']: 
        model_path = {}
        keys = ['MoE', 'exp1', 'exp2', 'exp3']
        for i in range(4):
            model_path[keys[i]] = os.path.join(path, 'new_data\\_simple_logs\\v_'+ v_num+'\\best_trained_model_'+epoch[i]+'.pt')



    # define instance of model with loaded data from path
    if inp['simple_m']:
        if not inp['DeepONet'] and not inp['MoE'] and not inp['pretrain'] and ('cVAE' not in inp or not inp['cVAE']) and ('MoE-split' not in inp or not inp['MoE-split']):
            model_test = FFNN(inp)
            model_test.load_state_dict(torch.load(model_path, map_location=device))
            model_test.eval()
            model_test_dict = {
                'standard': model_test
            }
        elif 'cVAE' in inp and inp['cVAE']:
            model_test = cVAE(inp)
            model_test.load_state_dict(torch.load(model_path, map_location=device))
            model_test.eval()
            model_test_dict = {
                'standard': model_test
            }
        elif inp['MoE']: 
            model_test = MoE(inp)
            model_test.load_state_dict(torch.load(model_path, map_location=device))
            model_test.eval()
            model_test_dict = {
                'standard': model_test
            }
        elif inp['MoE-split']:
            expert1 = Expert(inp)
            expert2 = Expert(inp)
            expert3 = Expert(inp)
            trained_experts = [expert1, expert2, expert3]
            model_test_dict = {
                'MoE': MoE(inp, trained_experts), 
                'exp1': expert1,
                'exp2': expert2,
                'exp3': expert3, 
                'standard': None}
            for i in ['MoE', 'exp1', 'exp2', 'exp3']:
                model_test_dict[i].load_state_dict(torch.load(model_path[i], map_location=device))
                model_test_dict[i].eval()
                model_test_dict[i].to(device)

        elif inp['DeepONet']: 
            model_test = DeepONet_vb(inp)
            model_test.load_state_dict(torch.load(model_path, map_location=device))
            model_test.eval()
            model_test_dict = {
                'standard': model_test
            }
        elif inp['pretrain']:
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
            my_pretrained_model = my_pretrained_model.to(device)
            my_pretrained_model.eval()
            model_test = FFNN_pretrain(inp, my_pretrained_model)
            model_test.load_state_dict(torch.load(model_path))
            model_test.eval()
            model_test_dict = {
                'standard': model_test
            }
    else:
        raise RuntimeWarning('Note: The Pytorch Lightning version of the code is outdated. Please use simple_m = True') 
        if inp['Double_Net']:
            model_test = LitFFNN_doub.load_from_checkpoint(lit_model_path)
            model_test.eval()

            model_test_dict = {
            "standard": model_test,
            "m": model_test_m,
            "b": model_test_b,
            "s": model_test_s
            }
        elif inp['Split_Net']:
            model_test_m = LitFFNN.load_from_checkpoint(lit_model_path_m)
            model_test_b = LitFFNN.load_from_checkpoint(lit_model_path_b)
            model_test_s = LitFFNN.load_from_checkpoint(lit_model_path_s)
            model_test_m.eval()
            model_test_b.eval()
            model_test_s.eval()
            model_test = None

            model_test_dict = {
            "standard": model_test,
            "m": model_test_m,
            "b": model_test_b,
            "s": model_test_s
        }
        elif inp['Split_Net_all']:
            model_test_dict = {}
            for i in range(7):
                model_test_dict[str(i)] = LitFFNN.load_from_checkpoint(lit_model_path[str(i)])
        else:
            model_test = LitFFNN.load_from_checkpoint(lit_model_path)
            model_test.eval()
            model_test_m, model_test_b, model_test_s = None, None, None
            model_test_dict = {
            "standard": model_test,
            "m": model_test_m,
            "b": model_test_b,
            "s": model_test_s
        }

    return model_test_dict



def make_prediction(inp, model_test_dict, data_model, transf_type: str, sc = False, dn = False):
    inp_shape = data_model['mat_data_np_TrainEvalTest']['X_test'].shape[1]
    out_shape = inp['out_size']

    if transf_type == 'mixed':
        transf_type_list_x = ['x-std']*3+['x-range']*3+['x-std']*(inp_shape-6)
        transf_type_list_y = ['y-std']*3+['y-range']*3+['y-std']*66
    elif transf_type == 'st-stitched':
        transf_type_list_x = ['x-std']*inp_shape
        transf_type_list_y = ['y-std']*out_shape+['y-st-stitched']*out_shape**2
    else: 
        transf_type_list_x = ['x-'+transf_type]*inp_shape
        transf_type_list_y = ['y-'+transf_type]*data_model['mat_data_np_TrainEvalTest']['y_test'].shape[1]
    
    X_test_t = transform_data(data_model['mat_data_np_TrainEvalTest']['X_test'], data_model['mat_data_stats'], forward=True, type = transf_type_list_x, sc = sc, dn = dn)
    X_train_t = transform_data(data_model['mat_data_np_TrainEvalTest']['X_train'], data_model['mat_data_stats'], forward=True, type = transf_type_list_x, sc = sc, dn = dn)

    if inp['batch_size'] is not None: 
        batch_size = inp['batch_size']
        test_dataset = TensorDataset(torch.Tensor(X_test_t).to(device))                 # don't need the labels.
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
        # don't calculate training data predictions.

    if inp['simple_m']:
        if not inp['DeepONet'] and ('MoE-split' not in inp or not inp['MoE-split']) and ('cVAE' not in inp or not inp['cVAE']) and ('energy' not in inp or not inp['energy']):
            model_test_dict['standard'].to(device)
            if inp['batch_size'] is not None: 
                predictions_t = []
                for (X_test_tt,) in test_loader: 
                    preds = model_test_dict['standard'](X_test_tt)
                    predictions_t.append(preds)
                predictions_t = torch.cat(predictions_t, dim=0)
                predictions_t_train = torch.zeros_like(predictions_t)
                print('Note: For testing with batch_size != None, the predictions of the training data are set to zero')
            else: 
                predictions_t = model_test_dict['standard'](torch.Tensor(X_test_t).to(device))
                predictions_t_train = model_test_dict['standard'](torch.Tensor(X_train_t).to(device))
        elif 'cVAE' in inp and inp['cVAE']:
            if inp['batch_size'] is not None: 
                raise UserWarning('This is not yet implemented for batchsize != None.')
            model_test_dict['standard'].to(device)
            enocded_t, predictions_t, mean_t, logvar_t = model_test_dict['standard'].encoder.forward(torch.Tensor(X_test_t).to(device))
            encoded_t_train, predictions_t_train, mean_t_train, logvar_t_train = model_test_dict['standard'].encoder.forward(torch.Tensor(X_train_t).to(device))
        elif inp['DeepONet']:
            if inp['batch_size'] is not None: 
                raise UserWarning('This is not yet implemented for batchsize != None.')
            model_test_dict['standard'].to(device)
            X_test_tt = torch.Tensor(X_test_t).to(device)
            X_train_tt = torch.Tensor(X_train_t).to(device)
            predictions_t = model_test_dict['standard'](X_test_tt[:,0:8], X_test_tt[:,8:].reshape(-1,inp_shape-8))
            predictions_t_train = model_test_dict['standard'](X_train_tt[:,0:8], X_train_tt[:,8:].reshape(-1,inp_shape-8))
        elif inp['MoE-split']:
            # using MoE with data split. This is not entirely up to date...
            raise UserWarning('This model is not up to date. Please double-check whether you really want to call this functionality.')
            # model is moved to device already in test_model_instance
            X_test_tt = torch.Tensor(X_test_t).to(device)
            y_test_t = transform_data(data_model['mat_data_np_TrainEvalTest']['y_test'], data_model['mat_data_stats'], forward=True, type = transf_type_list_y, sc = sc, dn = dn)
            y_test_tt = torch.Tensor(y_test_t).to(device)
            mat_torch_test = {'X_test_tt': X_test_tt, 'y_test_tt': y_test_tt}
            thresholds = np.array([200, 1000])
            mat_torch_exp1, mat_torch_exp2, mat_torch_exp3, mat_torch_MoE = data_split_MoE_test(mat_torch_test, data_model['stats_y_train'], thresholds)
            mat_torch_sort = {'exp1': mat_torch_exp1, 'exp2': mat_torch_exp2, 'exp3': mat_torch_exp3, 'MoE': mat_torch_MoE}

            predictions_t_dict = {}
            predictions_t_dict['exp1'] = model_test_dict['exp1'](mat_torch_exp1['X_test_tt'])
            predictions_t_dict['exp2'] = model_test_dict['exp2'](mat_torch_exp2['X_test_tt'])
            predictions_t_dict['exp3'] = model_test_dict['exp3'](mat_torch_exp3['X_test_tt'])
            predictions_t_dict['MoE'] = model_test_dict['MoE'](mat_torch_MoE['X_test_tt'])
        elif inp['energy']:
            model_test_dict['standard'].to(device)
            out_t = vmap(torch.func.jacrev(model_test_dict['standard'].forward), randomness='different')(torch.Tensor(X_test_t).to(device))
            out_t_train = vmap(torch.func.jacrev(model_test_dict['standard'].forward), randomness='different')(torch.Tensor(X_train_t).to(device))
            predictions_t = out_t[:,0,:8]
            predictions_t_train = out_t_train[:,0,:8]

    else:
        if not inp['Split_Net'] and not inp['Split_Net_all']:
            predictions_t = model_test_dict['standard'](torch.Tensor(X_test_t).to(device))
        elif inp['Split_Net']:
            predictions_t_m = model_test_dict['m'](torch.Tensor(np.hstack((X_test_t[:,0:3], X_test_t[:,8:].reshape(-1, inp_shape-8)))).to(device))
            predictions_t_b = model_test_dict['b'](torch.Tensor(np.hstack((X_test_t[:,3:6], X_test_t[:,8:].reshape(-1, inp_shape-8)))).to(device))
            predictions_t_s = model_test_dict['s'](torch.Tensor(np.hstack((X_test_t[:,6:8], X_test_t[:,8:].reshape(-1, inp_shape-8)))).to(device))
            predictions_t = torch.hstack((predictions_t_m, predictions_t_b, predictions_t_s))
        elif inp['Split_Net_all']:
            predictions_t = torch.zeros((X_test_t.shape[0],8))
            mask1 = np.array([0, 0, 0, 3, 3, 3, 6, 6, 6])
            mask2 = np.array([3, 3, 3, 6, 6, 6, 8, 8])
            for i in range(7):
                model_test_dict[str(i)].to(device)
                out = model_test_dict[str(i)](torch.Tensor(np.hstack((X_test_t[:,mask1[i]:mask2[i]], X_test_t[:,8].reshape(-1, inp_shape-8)))).to(device)) # (before last bracket): .to(device)
                predictions_t[:, i] = out.flatten()

    if not inp['MoE-split']: 
        predictions = transform_data(predictions_t.cpu().detach().numpy(), data_model['mat_data_stats'], forward = False, type = transf_type_list_y, sc = sc, dn = dn)
        predictions_train = transform_data(predictions_t_train.cpu().detach().numpy(), data_model['mat_data_stats'], forward = False, type = transf_type_list_y, sc = sc, dn = dn)
        test_labels_t = transform_data(data_model['mat_data_np_TrainEvalTest']['y_test'], data_model['mat_data_stats'], forward=True, type = transf_type_list_y, sc = sc, dn = dn)
        test_labels = data_model['mat_data_np_TrainEvalTest']['y_test']
        train_labels_t = transform_data(data_model['mat_data_np_TrainEvalTest']['y_train'], data_model['mat_data_stats'], forward=True, type = transf_type_list_y, sc = sc, dn = dn)
        train_labels = data_model['mat_data_np_TrainEvalTest']['y_train']

        plot_data = {
            'all_test_labels_t': test_labels_t,
            'all_predictions_t': predictions_t.cpu().detach().numpy(),
            'all_test_labels': test_labels,
            'all_predictions': predictions,

            'all_train_labels_t': train_labels_t,
            'all_predictions_t_train': predictions_t_train.cpu().detach().numpy(),
            'all_train_labels': train_labels,
            'all_predictions_train': predictions_train,
            }
    
    elif inp['MoE-split']:
        plot_data = {}
        for model in ['exp1', 'exp2', 'exp3', 'MoE']:
            predictions_t = predictions_t_dict[model]
            predictions = transform_data(predictions_t.cpu().detach().numpy(), data_model['mat_data_stats'], forward = False, type = transf_type_list_y, sc = sc, dn = dn)
            test_labels_t = mat_torch_sort[model]['y_test_tt'].cpu().detach().numpy()
            test_labels = transform_data(test_labels_t, data_model['mat_data_stats'], forward=False, type = transf_type_list_y, sc = sc, dn = dn)

            plot_data[model] = {
                'all_test_labels_t': test_labels_t,
                'all_predictions_t': predictions_t.cpu().detach().numpy(),
                'all_test_labels': test_labels,
                'all_predictions': predictions,
                }


    return plot_data


def make_inv_prediction(inp, model_test_dict, data_model, transf_type: str, num_samples, sc = False, dn = False):
    '''
    NOT IN USE
    Inverse prediction / sample generator for cVAE model
    note: inverse sobolev not implemented / not required --> transf_type 'st-stitched not possible here.'
    '''

    # transform y data into normalised units
    inp_shape = data_model['mat_data_np_TrainEvalTest']['X_test'].shape[1]
    if transf_type == 'mixed':
        transf_type_list_x = ['x-std']*3+['x-range']*3+['x-std']*(inp_shape-6)
        transf_type_list_y = ['y-std']*3+['y-range']*3+['y-std']*66
    else: 
        transf_type_list_x = [transf_type]*inp_shape
        transf_type_list_y = [transf_type]*data_model['mat_data_np_TrainEvalTest']['y_test'].shape[1]

    model_test_dict['standard'].to(device)
    y_test_t = transform_data(data_model['mat_data_np_TrainEvalTest']['y_test'], data_model['mat_data_stats'], forward=True, type = transf_type_list_y, sc = sc, dn = dn)
    y_test_tt = torch.Tensor(y_test_t).to(device)

    # make "prediction"
    x_pred_tt = model_test_dict['standard'].sample(num_samples, y_test_tt)
    

    # transform x data back into desired units
    x_pred = transform_data(x_pred_tt.cpu().detach().numpy(), data_model['mat_data_stats'], forward = False, type = transf_type_list_x, sc = sc, dn = dn)
    test_features_t = transform_data(data_model['mat_data_np_TrainEvalTest']['X_test'], data_model['mat_data_stats'], forward=True, type = transf_type_list_x, sc = sc, dn = dn)
    test_features = data_model['mat_data_np_TrainEvalTest']['X_test']

    # collect data for plotting
    plot_data_inv = {
            'all_test_features_t': test_features_t,
            'all_pred_features_t': x_pred_tt.cpu().detach().numpy(),
            'all_test_features': test_features,
            'all_pred': x_pred,
            }


    return plot_data_inv



def data_split_MoE_test(mat_torch, stats_y_train, thresholds):
    ''' 
    splits the data given in mat_torch into subsets for training of MoE
    thresholds: boundaries for splitting train set to get 3 expert subsets

    '''
    
    num_samples_train = mat_torch['X_test_tt'].shape[0]
    desired_keys = ['X_test_tt', 'y_test_tt']
    mat_torch_exp, mat_torch_MoE = {},{}
    for key in desired_keys:
        mat_torch_exp[key] = mat_torch[key][:int(num_samples_train/2)]
        mat_torch_MoE[key] = mat_torch[key][int(num_samples_train/2)+1:]

    thresholds_std = (10**(-5)*thresholds-stats_y_train['mean'][7])/stats_y_train['std'][7]
    mask1 = {}
    mask1['0'] = (abs(mat_torch_exp['y_test_tt'][:,7]) < thresholds_std[0])
    mask1['1'] = (abs(mat_torch_exp['y_test_tt'][:,7]) >= thresholds_std[0]) & (abs(mat_torch_exp['y_test_tt'][:,7]) < thresholds_std[1])
    mask1['2'] = (abs(mat_torch_exp['y_test_tt'][:,7]) >= thresholds_std[1])
    
    mat_torch_exp1, mat_torch_exp2, mat_torch_exp3 = {key: None for key in desired_keys}, {key: None for key in desired_keys}, {key: None for key in desired_keys}
    for key in desired_keys: 
        mat_torch_exp1[key] = mat_torch_exp[key][mask1[str(0)],:]
        mat_torch_exp2[key] = mat_torch_exp[key][mask1[str(1)],:]
        mat_torch_exp3[key] = mat_torch_exp[key][mask1[str(2)],:]
    
    print('The data distribution among the 3 experts in the training data is: ')
    print('Expert 1:', round(mat_torch_exp1['X_test_tt'].shape[0]/(num_samples_train/2)*100,0), 
            '% \n Expert 2:', round(mat_torch_exp2['X_test_tt'].shape[0]/(num_samples_train/2)*100,0),
            '% \n Expert 3:', round(mat_torch_exp3['X_test_tt'].shape[0]/(num_samples_train/2)*100,0), '%')

    return mat_torch_exp1, mat_torch_exp2, mat_torch_exp3, mat_torch_MoE