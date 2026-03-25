import os
import numpy as np
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import vmap
import ast
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Implementation according to this guideline:
# https://lightning.ai/docs/pytorch/stable/expertise_levels.html 
# https://lightning.ai/docs/pytorch/stable/model/train_model_basic.html

 
############################################### Definition of loss functions and Fourier Mapping ###############################################

class FourierMapping(nn.Module):
    # https://arxiv.org/pdf/2006.10739
    def __init__(self, input_dim, num_samples=5):
        super(FourierMapping, self).__init__()
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = input_dim * (2 * num_samples + 1)
        self.freqs = nn.Parameter(torch.Tensor(1, input_dim, num_samples), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.freqs, -torch.pi, torch.pi)

    def forward(self, x):
        if len(x.shape)<2:
            x = x.unsqueeze(0)
        fourier_features = torch.cat([
            torch.sin(x.unsqueeze(-1) * self.freqs),
            torch.cos(x.unsqueeze(-1) * self.freqs)
        ], dim=-1).reshape(x.size(0), self.output_dim - self.input_dim)
        return torch.cat([x, fourier_features], dim=-1)
    
    @property
    def in_features(self):
        return self.input_dim
    @property
    def out_features(self):
        return self.output_dim
    
    def __repr__(self):
        # Custom representation to display in_features and out_features like Linear layer
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features})"


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss
    
class MSLELoss(nn.Module):
    def __init__(self, eps =1, offset = 10):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        self.offset = offset

    def forward(self, yhat, y):
        loss = self.mse(torch.log(self.eps+self.offset+y),torch.log(self.eps+self.offset+yhat))
        return loss

class wMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y, y_hat, weight=10):
        loss_ind = torch.zeros(y.shape).to(device)
        mask0 = torch.abs(y)<0.5
        mask1 = torch.abs(y)>=0.5
        loss_ind[mask0] = weight*(y[mask0]- y_hat[mask0])**2
        loss_ind[mask1] = (y[mask1]- y_hat[mask1])**2
        loss = torch.mean(loss_ind)
        return loss


loss_mapping = {
    'MSELoss': nn.MSELoss,
    'HuberLoss': nn.HuberLoss,
    'MSLELoss': MSLELoss,
    'wMSELoss': wMSELoss,
    'RMSELoss': RMSELoss,
}



################################################ Definition of Architectures ###############################################

class FFNN(nn.Module):
    def __init__(self, inp:dict):
        super().__init__()
        self.inp = inp
        self.activation = getattr(nn, self.inp['activation'])
        self.hidden_layers = ast.literal_eval(self.inp['hidden_layers'])
        if self.inp['fourier_mapping']:
            self.fourier_mapping = FourierMapping(self.inp['input_size'])

        layers = []
        for i in range(len(self.hidden_layers) - 1):
            if self.inp['BatchNorm']:
                layers.append(nn.BatchNorm1d(self.hidden_layers[i]))
            layers.append(nn.Dropout(self.inp['dropout_rate']))
            layers.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))
            layers.append(self.activation())
        
        if self.inp['fourier_mapping']:
            inp_size = self.fourier_mapping.output_dim
        else: 
            inp_size = self.inp['input_size']

        self.layers = nn.Sequential(
            nn.Linear(inp_size, self.hidden_layers[0]),
            self.activation(),
            *layers,
            nn.Linear(self.hidden_layers[-1], self.inp['out_size'])
        )

    def forward(self, x):
        if self.inp['fourier_mapping']:
            out = self.fourier_mapping(x)
        else: 
            out = x
            
        for layer in self.layers:
            out = layer(out)
        return out

# Definition of DeepONets

class BranchNet(nn.Module):
    def __init__(self, inp:dict):
        super(BranchNet, self).__init__()
        self.inp = inp
        self.inp.update({'input_size': 8}, allow_val_change = True)
        self.ffnn = FFNN(self.inp)

    def forward(self, x):
        return self.ffnn.forward(x)

class TrunkNet(nn.Module):
    def __init__(self, inp:dict):
        super(TrunkNet, self).__init__()
        self.inp = inp
        # self.inp.update({'input_size': 1}, allow_val_change = True)
        # self.ffnn = FFNN(self.inp)

        self.activation = getattr(nn, self.inp['activation'])
        self.hidden_layers = ast.literal_eval(self.inp['hidden_layers'])
        if self.inp['fourier_mapping']:
            self.fourier_mapping = FourierMapping(1)

        layers = []
        for i in range(len(self.hidden_layers) - 1):
            layers.append(nn.Dropout(self.inp['dropout_rate']))
            if self.inp['BatchNorm']:
                layers.append(nn.BatchNorm1d(self.hidden_layers[i]))
            layers.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))
            layers.append(self.activation())
        
        if self.inp['fourier_mapping']:
            inp_size = self.fourier_mapping.output_dim
        else:
            if 'num_trunk' in self.inp: 
                inp_size = self.inp['num_trunk']
            else: 
                inp_size = 1

        self.layers = nn.Sequential(
            nn.Linear(inp_size, self.hidden_layers[0]),
            self.activation(),
            *layers,
            nn.Linear(self.hidden_layers[-1], self.inp['out_size'])
        )

    def forward(self, x):
        if self.inp['fourier_mapping']:
            out = self.fourier_mapping(x)
        else: 
            out = x

        for layer in self.layers:
            out = layer(out)

        # self.ffnn.forward(x)
        return out

class DeepONet_vb(nn.Module):
    def __init__(self, inp:dict):
        super(DeepONet_vb, self).__init__()
        self.inp = inp
        self.activation = getattr(nn, self.inp['activation'])
        self.hidden_layers = ast.literal_eval(self.inp['hidden_layers'])
        self.branch_net = BranchNet(self.inp)
        self.trunk_net = TrunkNet(self.inp)
    
    def forward(self, x_b, x_t):
        branch_out = self.branch_net(x_b)
        trunk_out = self.trunk_net(x_t)
        if len(branch_out.shape) >1: 
            out = torch.einsum("bi,bi->bi", branch_out, trunk_out)
        else: 
            out = torch.einsum("b,b->b", branch_out, trunk_out)
        # trunk_out = trunk_out.expand(-1, branch_out.size(1),-1)
        # out = torch.sum(branch_out * trunk_out, dim = -1)
        return out

# Definition of pretrained FFNN
class FFNN_pretrain(nn.Module):
    def __init__(self, inp, my_pretrained_model):
        super(FFNN_pretrain, self).__init__()
        self.inp = inp
        self.activation_new = getattr(nn, self.inp['activation'])
        self.hidden_layers_new = ast.literal_eval(self.inp['hidden_layers_new'])

        # Modify pretrained model to remove the last layer and freeze parameters of pretrained model
        if self.inp['fourier_mapping']:
            self.fourier_mapping = my_pretrained_model.ffnn.fourier_mapping
            self.pretrained_ffnn = nn.Sequential(*list(my_pretrained_model.ffnn.layers.children())[:-2])
        elif not self.inp['DeepONet']:
            self.pretrained_ffnn = nn.Sequential(*list(my_pretrained_model.layers.children())[:-1])
            new_last_layer = list(my_pretrained_model.layers.children())[-1]
            pretrain_out_size = new_last_layer.in_features
        elif self.inp['DeepONet']:
            self.branch_net = nn.Sequential(*list(my_pretrained_model.branch_net.layers.children()[:,-1]))
            self.trunk_net = nn.Sequential(*list(my_pretrained_model.trunk_net.layers.children()[:,-1]))
            # TODO! --> also adjust in forward function...
            print('This is not yet implemented.')
            pass

        for param in self.pretrained_ffnn.parameters():
            param.requires_grad = False
        
        
        # Add new layers
        new_layers = []
        for i in range(len(self.hidden_layers_new)-1):
            new_layers.append(nn.Dropout(self.inp['dropout_rate']))
            if self.inp['BatchNorm_new']:
                new_layers.append(nn.BatchNorm1d(self.hidden_layers_new[i]))
            new_layers.append(nn.Linear(self.hidden_layers_new[i], self.hidden_layers_new[i + 1]))
            new_layers.append(self.activation_new())
        
        self.new_layers = nn.Sequential(
            nn.Linear(pretrain_out_size, self.hidden_layers_new[0]),
            self.activation_new(),
            *new_layers,
            nn.Linear(self.hidden_layers_new[-1], self.inp['out_size'])
        )

    def forward(self, x):
        if self.inp['fourier_mapping']:
            x = self.fourier_mapping(x)
        x = self.pretrained_ffnn(x)
        x = self.new_layers(x)
        return x

# Definition of MoE

class Expert(nn.Module):
    def __init__(self, inp):
        super(Expert, self).__init__()
        self.inp = inp
        self.ffnn  = FFNN(self.inp)

    def forward(self, x):
        return self.ffnn.forward(x)

class Gating(nn.Module):
    def __init__(self, inp, num_experts):
        super().__init__()
        self.inp = inp
        self.activation = getattr(nn, self.inp['activation'])
        self.hidden_layers = ast.literal_eval(self.inp['hidden_layers'])
        self.num_experts = num_experts
        if self.inp['fourier_mapping']:
            self.fourier_mapping = FourierMapping(self.inp['input_size'])

        layers = []
        for i in range(len(self.hidden_layers) - 1):
            layers.append(nn.Dropout(self.inp['dropout_rate']))
            if self.inp['BatchNorm']:
                layers.append(nn.BatchNorm1d(self.hidden_layers[i]))
            layers.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))
            layers.append(self.activation())
        
        if self.inp['fourier_mapping']:
            inp_size = self.fourier_mapping.output_dim
        else: 
            inp_size = self.inp['input_size']

        self.layers = nn.Sequential(
            nn.Linear(inp_size, self.hidden_layers[0]),
            self.activation(),
            *layers,
            nn.Linear(self.hidden_layers[-1], self.num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        if self.inp['fourier_mapping']:
            out = self.fourier_mapping(x)
        else: 
            out = x
        if out.dim() == 1:
            out = out.unsqueeze(0)

        for layer in self.layers:
            out = layer(out)
        return out

class MoE(nn.Module):
    def __init__(self, inp, num_experts = 10):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([FFNN(inp) for _ in range(num_experts)])

        self.gating = Gating(inp, num_experts)

    def forward(self, x):
        # freeze weights in expert net
        # for expert in self.experts: 
        #     for param in expert.parameters():
        #         param.requires_grad = False
        
        # calculate "weights" of different experts with gating net 
        if x.dim() == 1:
            x = x.unsqueeze(0)
        weights = self.gating(x)
        outputs = torch.stack([expert(x) for expert in self.experts], dim = 1)
        # weights = weights.unsqueeze(1).expand_as(outputs)
        out = torch.sum(outputs*weights.unsqueeze(-1), dim =1)
        
        return out

# Definition of cVAE

class Encoder(nn.Module):
    def __init__(self, inp):
        super(Encoder, self).__init__()
        self.inp = inp
        self.activation = getattr(nn, self.inp['activation'])
        self.hidden_layers_enc = ast.literal_eval(self.inp['layers_enc'])
        self.mean_layer = nn.Linear(self.inp['latent_dim'], 2)
        self.logvar_layer = nn.Linear(self.inp['latent_dim'], 2)
        self.pred_layer = nn.Linear(self.inp['latent_dim'], self.inp['out_size'])
        if self.inp['fourier_mapping']:
            self.fourier_mapping = FourierMapping(self.inp['input_size'])

        layers = []
        for i in range(len(self.hidden_layers_enc) - 1):
            layers.append(nn.Dropout(self.inp['dropout_rate']))
            if self.inp['BatchNorm']:
                layers.append(nn.BatchNorm1d(self.hidden_layers_enc[i]))
            layers.append(nn.Linear(self.hidden_layers_enc[i], self.hidden_layers_enc[i + 1]))
            layers.append(self.activation())
        if self.inp['fourier_mapping']:
            inp_size = self.fourier_mapping.output_dim
        else: 
            inp_size = self.inp['input_size']

        self.layers = nn.Sequential(
            nn.Linear(inp_size, self.hidden_layers_enc[0]),
            self.activation(),
            *layers,
            nn.Linear(self.hidden_layers_enc[-1], self.inp['latent_dim'])
        )
    

    def forward(self, x):
        if self.inp['fourier_mapping']:
            out = self.fourier_mapping(x)
        else: 
            out = x
        for layer in self.layers:
            out = layer(out)
        
        pred = self.pred_layer(out)
        mean, logvar = self.mean_layer(out), self.logvar_layer(out)

        return out, pred, mean, logvar

class Decoder(nn.Module):
    def __init__(self,inp):
        super(Decoder, self).__init__()
        self.inp = inp
        self.activation = getattr(nn, self.inp['activation'])
        self.hidden_layers_dec = ast.literal_eval(self.inp['layers_dec'])

        layers = []
        for i in range(len(self.hidden_layers_dec) - 1):
            layers.append(nn.Dropout(self.inp['dropout_rate']))
            if self.inp['BatchNorm']:
                layers.append(nn.BatchNorm1d(self.hidden_layers_dec[i]))
            layers.append(nn.Linear(self.hidden_layers_dec[i], self.hidden_layers_dec[i + 1]))
            layers.append(self.activation())
        inp_size = 2
        latent_dim = self.inp['latent_dim']

        self.layers = nn.Sequential(
            nn.Linear(inp_size, latent_dim),
            self.activation(),
            nn.Linear(latent_dim, self.hidden_layers_dec[0]),
            self.activation(),
            *layers,
            nn.Linear(self.hidden_layers_dec[-1], self.inp['input_size'])
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

class cVAE(nn.Module):
    def __init__(self, inp):
        super(cVAE, self).__init__()
        self.inp = inp
        self.encoder = Encoder(self.inp)
        self.decoder = Decoder(self.inp)
        self.label_projector = nn.Sequential(nn.Linear(self.inp['out_size'], 2))
    
    def condition_on_label(self, z, y):
        projected_label = self.label_projector(y.float())
        return z + projected_label

    def reparametrisation(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon
        return z

    def forward(self, x, y):
        encoded, pred, mean, logvar = self.encoder.forward(x)
        z = self.reparametrisation(mean, logvar)
        decoded = self.decoder.forward(self.condition_on_label(z, y))
        return pred, decoded, mean, logvar
    
    def sample(self, num_samples, y):
        with torch.no_grad():
            z = torch.randn(num_samples, self.inp['latent_dim']).to(device)
            samples = self.decoder(self.condition_on_label(z, y))
        return samples

################################################ Definition of Sobolev Losses ###############################################

class CustomLosses:
    def __init__(self, inp:dict, ffnn:FFNN, deeponet:DeepONet_vb):
        super().__init__()
        self.ffnn = ffnn
        self.deeponet = deeponet
        self.inp = inp
        self.criterion = loss_mapping[self.inp['loss_type']]()

    def Sobolev_CustomLoss(self, features, labels):
        if not self.inp['Sobolev']:
            raise SyntaxError("to access Sobolev_CustomLoss please set inp['Sobolev'] = True")
        if self.inp['w_s'] == 'max':
            w_s = 'max'
        else:
            w_s = ast.literal_eval(self.inp['w_s'])
        if self.inp['w_smooth'] is not None: 
            w_smooth = ast.literal_eval(self.inp['w_smooth'])

        ###### calculate "first order loss" ######
        features.requires_grad_(True)
        if not self.inp['DeepONet']:
            predictions = self.ffnn(features)
        elif self.inp['DeepONet']:
            if 'num_trunk' in self.inp:
                num_trunk = self.inp['num_trunk']
            else: 
                num_trunk = 1
            predictions = self.deeponet(features[:,0:8], features[:,8:].reshape(-1,num_trunk))

        if not self.inp['DeepONet']:
            geom_size = self.inp['input_size']-self.inp['out_size']
            loss1 = self.criterion(predictions, labels[:,0:self.inp['input_size']-geom_size])
        elif self.inp['DeepONet']:
            loss1 = self.criterion(predictions, labels[:,0:8])

        ####### calculate "second order loss" #####
        # J = torch.cat([torch.autograd.functional.jacobian(self.ffnn.forward, features[i:i+1], create_graph = True) for i in range(len(features))], dim=0)[:,:,0,:]
        if not self.inp['DeepONet']:
            J = vmap(torch.func.jacrev(self.ffnn.forward), randomness='different')(features)
        elif self.inp['DeepONet']:
            if 'num_trunk' in self.inp:
                num_trunk = self.inp['num_trunk']
            else: 
                num_trunk = 1
            J = vmap(torch.func.jacrev(self.deeponet.forward), randomness='different')(features[:,0:8], features[:,8:].reshape(-1,num_trunk))

        sz = self.inp['out_size']
        if len(J.shape)>3:
            J = J.squeeze(1)
        # to calculate the loss directly based on the entire stiffness matrix (including the values that should be zero)
        if ('w_diag' not in self.inp or self.inp['w_diag'] == None) and ('w_range_D' not in self.inp or not self.inp['w_range_D']) and ('w_nonzero' not in self.inp or not self.inp['w_nonzero']):
            # calculate the loss directly based on entire stiffness matrix
            loss2 = self.criterion(J[:,:sz,:sz], labels[:,sz:].reshape(-1,sz,sz))
        elif self.inp['w_range_D'] and not self.inp['w_nonzero']:
            # weight the individual loss terms according to the ranges of the labels
            range_i = torch.max(labels[:,sz:], axis = 0).values - torch.min(labels[:,sz:], axis = 0).values
            range_i[range_i == 0] = 1
            range_tot = torch.sum(torch.divide(1,range_i))
            sf_i = (torch.divide(1,range_i)/range_tot).reshape(-1,8,8)
            loss2 = self.criterion(sf_i*J[:,:sz,:sz], sf_i*(labels[:,sz:].reshape(-1,sz,sz)))
        elif self.inp['w_nonzero'] and not self.inp['w_range_D']:
            # to calculate the loss only based on the non-zero values in the stiffness matrix:
            J_ = torch.cat((J[:,:sz-2,:sz-2].reshape((-1,36)), J[:,6,6].reshape((-1,1)), J[:,7,7].reshape((-1,1))), axis=1)
            labels_8_8 = labels[:,sz:].reshape((-1,sz,sz))
            labels_ = torch.cat((labels_8_8[:,:sz-2,:sz-2].reshape((-1,36)), labels_8_8[:,6,6].reshape((-1,1)), labels_8_8[:,7,7].reshape((-1,1))), axis = 1)
            loss2 = self.criterion(J_, labels_)
        elif self.inp['w_nonzero'] and self.inp['w_range_D']:
            J_ = torch.cat((J[:,:sz-2,:sz-2].reshape((-1,36)), J[:,6,6].reshape((-1,1)), J[:,7,7].reshape((-1,1))), axis=1)
            labels_8_8 = labels[:,sz:].reshape((-1,sz,sz))
            labels_ = torch.cat((labels_8_8[:,:sz-2,:sz-2].reshape((-1,36)), labels_8_8[:,6,6].reshape((-1,1)), labels_8_8[:,7,7].reshape((-1,1))), axis = 1)
            range_i_ = torch.max(labels_, axis = 0).values - torch.min(labels_, axis=0).values
            range_tot_ = torch.sum(torch.divide(1, range_i_))
            sf_i_ = torch.divide(1,range_i_)/range_tot_   
            loss2 = self.criterion(sf_i_*J_, sf_i_*labels_)
        elif self.inp['w_diag']:
            if self.inp['w_nonzero'] or self.inp['w_range_D']:
                raise NameError('The code for w_diag in combination with w_nonzero or w_range_D is not implemented. Please re-consider the choice.')
            # add weights to the diagonal terms in the stiffness matrix to change their relative importance.
            w_diag = ast.literal_eval(self.inp['w_diag'])
            # D_weights = (w_diag[1]-1)*torch.tensor([[0, 1, 1, 0, 1, 1, 0, 0],
            #                                         [1, 0, 1, 1, 0, 1, 0, 0],
            #                                         [1, 1, 0, 1, 1, 0, 0, 0],
            #                                         [0, 1, 1, 0, 1, 1, 0, 0],
            #                                         [1, 0, 1, 1, 0, 1, 0, 0],
            #                                         [1, 1, 0, 1, 1, 0, 0, 0],
            #                                         [0, 0, 0, 0, 0, 0, 1, 0],
            #                                         [0, 0, 0, 0, 0, 0, 0, 1]], device = device)+1
            # D_weights = (w_diag[1]-1)*torch.tensor([[0, 1, 0, 0, 1, 0, 0, 0],
            #                                         [1, 0, 0, 1, 0, 0, 0, 0],
            #                                         [0, 0, 0, 0, 0, 0, 0, 0],
            #                                         [0, 1, 0, 0, 1, 0, 0, 0],
            #                                         [1, 0, 0, 1, 0, 0, 0, 0],
            #                                         [0, 0, 0, 0, 0, 0, 0, 0],
            #                                         [0, 0, 0, 0, 0, 0, 1, 0],
            #                                         [0, 0, 0, 0, 0, 0, 0, 1]], device = device)+1
            D_weights = (w_diag[1]-1)*torch.tensor([[0, 1, 0],
                                                    [1, 0, 1],
                                                    [0, 1, 0]], device = device)+1


            # D_weights = (w_diag[1]-1)*torch.tensor([[0, 1, 1, 0, 0, 0, 0, 0],
            #                                         [1, 0, 1, 0, 0, 0, 0, 0],
            #                                         [1, 1, 0, 0, 0, 0, 0, 0],
            #                                         [0, 0, 0, 0, 1, 1, 0, 0],
            #                                         [0, 0, 0, 1, 0, 1, 0, 0],
            #                                         [0, 0, 0, 1, 1, 0, 0, 0],
            #                                         [0, 0, 0, 0, 0, 0, 1, 0],
            #                                         [0, 0, 0, 0, 0, 0, 0, 1]], device = device)+1

            J_ = torch.divide(J[:,:sz,:sz], D_weights)
            labels_ = torch.divide(labels[:,sz:].reshape(-1,sz,sz), D_weights)
            loss2 = self.criterion(J_, labels_)
        
        ######## calculate fourth-order loss if w_smooth is not None #######
        if self.inp['w_smooth'] is not None: 

            # Calculate full third-order derivative
            if w_smooth[1] == 'full':
                print('For this method please use batch-size smaller than 16 for not overloading the memory.')
                J3 = vmap(torch.func.jacrev(torch.func.jacrev(torch.func.jacrev(self.ffnn.forward))), 
                        randomness='different')(features)
                loss3 = self.criterion(J3[:,:sz, :sz, :sz, :sz], torch.zeros_like(J3[:,:sz, :sz, :sz, :sz]).to(device))
            else: 
                # Calculate stochastic version - estimation trick (see Sobolev paper)
                x = features
                v = torch.randn_like(x)
                J3 = torch.autograd.functional.jvp(
                    lambda x: torch.autograd.functional.jvp(
                        lambda x: torch.autograd.functional.jvp(self.ffnn.forward, (x,), (v,), create_graph=True)[1],
                        (x,), (v,), create_graph=True
                    )[1],
                    (x,), (v,)
                )[1]
                loss3 = self.criterion(J3, torch.zeros_like(J3).to(device))


        ######## stitch together according to weights #######
        if w_s == 'max':
            loss = max(loss1, loss2)
        else: 
            if self.inp['w_smooth'] is not None:
                loss = (w_s[0]*loss1 + w_s[1]*loss2) + w_smooth[0]*loss3
            else: 
                loss = w_s[0]*loss1 + w_s[1]*loss2
        
        loss_logs = {"loss1": loss1,
                     "loss2": loss2,
                     "train_loss": loss}
        if self.inp['w_smooth'] is not None:
            loss_logs['loss3'] = loss3

        return loss, loss_logs
    
    comp_SobolevCustom = torch.compile(Sobolev_CustomLoss)
    
    def Sobolev_Energy_CustomLoss(self, features, labels):
        '''
        expects out_size = 1
        expects labels to include both sigma and D values --> sobolev needs to be set to true
        implementation so far just for pure NN, not for DeepONet
        '''

        if not self.inp['Sobolev']:
            raise SyntaxError("to access Sobolev_Energy_CustomLoss please set inp['Sobolev'] = True")
        if self.inp['w_s'] == 'max':
            w_s = 'max'
        else:
            w_s = ast.literal_eval(self.inp['w_s'])

        if self.inp['DeepONet']: 
            raise RuntimeError('Implementation for DeepONet not available.')

        # calculate first derivative and corresponding loss:
        J = vmap(torch.func.jacrev(self.ffnn.forward), randomness='different')(features)        # this will need to be compared to the stresses
        loss1 = self.criterion(J[:,0,:8], labels[:,:8].reshape(-1,8))

        # calculate the second derivative and corresponding loss:
        H = vmap(torch.func.jacrev(torch.func.jacrev(self.ffnn.forward)), randomness='different')(features)     # this will be compared to the stiffness matrix
        loss2 = self.criterion(H[:,0,:8,:8], labels[:,8:].reshape(-1,8,8))


        if w_s == 'max':
            loss = max(loss1, loss2)
        else: 
            loss = w_s[0]*loss1 + w_s[1]*loss2   
        
        loss_logs = {"loss1": loss1,
                     "loss2": loss2,
                     "train_loss": loss}
        
        return loss, loss_logs

    def Split_CustomLoss(self, y_hat, y):
        w = self.inp['w_mbs']
        # criterion = getattr(nn, self.inp['loss_type'])()
        loss_m = self.criterion(y_hat[:,0:3], y[:,0:3])
        loss_b = self.criterion(y_hat[:,3:6],y[:,3:6])
        loss_s = self.criterion(y_hat[:,6:8], y[:,6:8])
        loss = w[0]*loss_m + w[1]*loss_b + w[2]*loss_s
        loss_logs = {
            "loss_m": loss_m,
            "loss_b": loss_b,
            "loss_s": loss_s
        }
        return loss, loss_logs
    
    def Sobolev_DiagonalLoss(self, features, labels):
        if not self.inp['Sobolev']:
            raise SyntaxError("to access Sobolev_CustomLoss please set inp['Sobolev'] = True")
        if self.inp['w_s'] == 'max':
            w_s = 'max'
        else:
            w_s = ast.literal_eval(self.inp['w_s'])

        ###### calculate "first order loss" ######
        features.requires_grad_(True)
        if not self.inp['DeepONet']:
            predictions = self.ffnn(features)
        elif self.inp['DeepONet']:
            if 'num_trunk' in self.inp:
                num_trunk = self.inp['num_trunk']
            else: 
                num_trunk = 1
            predictions = self.deeponet(features[:,0:8], features[:,8:].reshape(-1,num_trunk))

        if not self.inp['DeepONet']:
            geom_size = self.inp['input_size']-self.inp['out_size']
            loss1 = self.criterion(predictions, labels[:,0:self.inp['input_size']-geom_size])
        elif self.inp['DeepONet']:
            loss1 = self.criterion(predictions, labels[:,0:8])

        ####### calculate "second order loss" #####
        # J = torch.cat([torch.autograd.functional.jacobian(self.ffnn.forward, features[i:i+1], create_graph = True) for i in range(len(features))], dim=0)[:,:,0,:]
        if not self.inp['DeepONet']:
            J = vmap(torch.func.jacrev(self.ffnn.forward), randomness='different')(features)
        elif self.inp['DeepONet']:
            if 'num_trunk' in self.inp:
                num_trunk = self.inp['num_trunk']
            else: 
                num_trunk = 1
            J = vmap(torch.func.jacrev(self.deeponet.forward), randomness='different')(features[:,0:8], features[:,8:].reshape(-1,num_trunk))

        sz = self.inp['out_size']
        if len(J.shape)>3:
            J = J.squeeze(1)
        # to calculate the loss directly based on the entire stiffness matrix (including the values that should be zero)
        if ('w_diag' not in self.inp or self.inp['w_diag'] == None) and ('w_range_D' not in self.inp or not self.inp['w_range_D']) and ('w_nonzero' not in self.inp or not self.inp['w_nonzero']):
            # calculate the loss directly based on entire stiffness matrix
            loss2 = self.criterion(J[:,:sz,:sz].diagonal(dim1 = -2, dim2 = -1), labels[:,sz:].reshape(-1,sz,sz).diagonal(dim1 = -2, dim2 = -1))
        else: 
            raise UserWarning('These features are not implemented in diag_loss: w_diag, w_range_D, w_nonzero')
        
        if w_s == 'max':
            loss = max(loss1, loss2)
        else: 
            loss = w_s[0]*loss1 + w_s[1]*loss2   
        
        loss_logs = {"loss1": loss1,
                     "loss2": loss2,
                     "train_loss": loss}

        return loss, loss_logs
        
    
    def Sobolev_Split_CustomLoss(self, features, labels):
        # criterion = getattr(nn, self.inp['loss_type'])()
        
        # calculate "first order loss"
        features.requires_grad_(True)
        predictions = self.ffnn(features)
        loss1 = self.Split_CustomLoss(predictions, labels[:,0:8])
        self.log("loss1", loss1)

        # calculate "second order loss"
        J = torch.cat([torch.autograd.functional.jacobian(self.forward, features[i:i+1], create_graph = True) for i in range(len(features))], dim=0)[:,:,0,:]
        w = self.inp['w_Dmbs']
        loss_Dm = self.criterion(J[:,0:3,0:3], labels[:,8:].reshape(-1,8,8)[:,0:3,0:3])
        loss_Db = self.criterion(J[:,3:6,3:6], labels[:,8:].reshape(-1,8,8)[:,3:6,3:6])
        loss_Ds = self.criterion(J[:,6:8,6:8], labels[:,8:].reshape(-1,8,8)[:,6:8,6:8])
        loss2 = w[0]*loss_Dm + w[1]*loss_Db+w[2]*loss_Ds
        loss_logs = {"loss_Dm": loss_Dm,
                  "loss_Db": loss_Db,
                  "loss_Ds": loss_Ds, 
                  "loss2": loss2}

        # stitch together according to weights
        w_s = self.inp['w_s']
        loss = w_s[0]*loss1 + w_s[1]*loss2   
        return loss, loss_logs


    def cVAE_Loss(self, features, pred_features, perform, pred_perform, mean, log_var):
        
        w_cVAE = ast.literal_eval(self.inp['w_cVAE'])
        reproduction_loss = self.criterion(pred_features,features)
        performance_loss = self.criterion(pred_perform, perform)
        KLD = -0.5*torch.sum(1+log_var-mean.pow(2)-log_var.exp())
        total_loss = w_cVAE[0]*reproduction_loss + w_cVAE[1]*KLD + w_cVAE[2]*performance_loss

        loss_logs = {"loss_recon": reproduction_loss,
                     "loss_perf": performance_loss,
                     "loss_KLD": KLD, 
                     "train_loss": total_loss}

        return total_loss, loss_logs

################################################ Definition of Lightning Modules [OUTDATED] ###############################################


class LitFFNN(L.LightningModule):
    def __init__(self, inp: dict, ffnn: FFNN):
        super().__init__()
        self.ffnn = ffnn
        self.inp = inp
        if type(self.inp['loss_type'])==str:
            # self.criterion = getattr(nn, self.inp['loss_type'])()
            self.criterion = loss_mapping[self.inp['loss_type']]()
        else: 
            self.criterion = self.inp['loss_type']
        self.custom_crit = CustomLosses(self.inp, self.ffnn, None)
        self.save_hyperparameters()
    
    def forward(self, x):
        return self.ffnn.forward(x)
    
    
    def share_step(self, batch, mode):
        x, y = batch
        y_hat = self.forward(x)
        if self.inp['Sobolev'] and self.inp['Split_Loss']:
            loss, loss_logs = self.custom_crit.Sobolev_Split_CustomLoss(x,y)
            self.log_dict(loss_logs)
        elif self.inp['Sobolev']:
            loss, loss_logs = self.custom_crit.Sobolev_CustomLoss(x, y)
            self.log_dict(loss_logs)
        elif self.inp['Split_Loss']:
            loss, loss_logs = self.custom_crit.Split_CustomLoss(y_hat, y[:,0:8])
            self.log_dict(loss_logs)
            self.log("loss1",loss)
        else:
            # loss = self.criterion(y_hat, y[:,0:8])
            loss = self.criterion(y_hat, y)
            self.log("loss1", loss)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.share_step(batch,"train")
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        val_loss = self.share_step(batch, "val")
        self.log("val_loss", val_loss, prog_bar = True)
        return val_loss

    def test_step(self, batch, batch_idx):
        test_loss = self.share_step(batch,"test")
        self.log("test_loss", test_loss, prog_bar = True)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.inp['learning_rate'], weight_decay = self.inp['weight_decay'])
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.5)
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10),
            "monitor": "val_loss",
            "name": "ReduceLROnPlateau_Scheduler",
            "interval": "epoch",
            }
        return [optimizer], [lr_scheduler]
    

class LitFFNN_doub(L.LightningModule):
    def __init__(self, inp: dict, ffnn_sig: FFNN, ffnn_D:FFNN):
        super().__init__()
        self.ffnn_sig = ffnn_sig
        self.ffnn_D = ffnn_D
        self.inp = inp
        # self.criterion = getattr(nn, self.inp['loss_type'])()
        self.criterion = loss_mapping[self.inp['loss_type']]()
        self.custom_crit = CustomLosses(self.inp, self.ffnn_sig, None)    # CustomCrit is only used for ffnn_sig!
        self.save_hyperparameters()

        # according to this website, if I have two optimizers, I need to code them manually in the training step.
        # https://lightning.ai/docs/pytorch/2.4.0/common/lightning_module.html#configure-optimizers
        self.automatic_optimization = False
    
    def forward(self, x):
        # the standard forward pass should be done purely with the sig-net
        return self.ffnn_sig.forward(x)
    
    def forward_sig(self, x):
        return self.ffnn_sig.forward(x)

    def forward_D(self,x):
        return self.ffnn_D.forward(x)

    def loss_sig(self, x, y):
        return self.custom_crit.Sobolev_CustomLoss(x, y)

    def loss_D(self, x, y):

        # calculate "first order loss"
        x.requires_grad_(True)
        predictions = self.forward_D(x)
        loss1 = self.criterion(predictions, y[:,8:])

        # calculate "second order loss" (which is the same as in loss_sig; but loss2 is calculated with output of D-net, not "real D")
        J = torch.cat([torch.autograd.functional.jacobian(self.forward_sig, x[i:i+1], create_graph = True) for i in range(len(x))], dim=0)[:,:,0,:]
        loss2 = self.criterion(J[:,:8,:8], predictions.reshape(-1,8,8))

        # stitch together according to weights
        w_s = self.inp['w_s']
        loss = w_s[0]*loss1 + w_s[1]*loss2   
        
        loss_logs = {"loss1_D": loss1,
                     "loss2_D": loss2}

        return loss, loss_logs
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        sig_opt, D_opt = self.optimizers()
        sig_lr, D_lr = self.lr_schedulers()

        # optimize sig-net
        loss_sig, loss_logs_sig = self.loss_sig(x, y)
        self.log_dict(loss_logs_sig)
        sig_opt.zero_grad()
        self.manual_backward(loss_sig)
        sig_opt.step()
        # sig_lr.step(self.trainer.callback_metrics["loss"])
        
        # optimize D-helper-net
        loss_D, loss_logs_D = self.loss_D(x, y)
        self.log_dict(loss_logs_D)
        D_opt.zero_grad()
        self.manual_backward(loss_D)
        D_opt.step()
        # D_lr.step(self.trainer.callback_metrics["loss"])

        self.log_dict({"train_loss_sig": loss_sig, "train_loss_D": loss_D}, prog_bar=True)
        
    def validation_step(self, batch, batch_idx):
        # evaluation only carried out for sig-net... (is this correct?)
        x, y = batch
        val_loss, loss_logs = self.custom_crit.Sobolev_CustomLoss(x,y)
        self.log("val_loss", val_loss, prog_bar = True)
        return val_loss

    def test_step(self, batch, batch_idx):
        # testing only carried out for sig-net... (is this correct?)
        x,y = batch
        test_loss, loss_logs = self.custom_crit.Sobolev_CustomLoss(x, y)
        self.log("test_loss", test_loss, prog_bar = True)
        return test_loss

    def configure_optimizers(self):
        optimizer_sig = torch.optim.Adam(self.ffnn_sig.parameters(), lr = self.inp['learning_rate'])
        optimizer_D = torch.optim.Adam(self.ffnn_D.parameters(), lr = self.inp['learning_rate'])
        lr_scheduler_sig = {
            "scheduler": ReduceLROnPlateau(optimizer_sig, mode='min', factor=0.5, patience=10),
            "name": "ReduceLROnPlateau_Scheduler_sig",
            "interval": "epoch",
            }
        lr_scheduler_D = {
            "scheduler": ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=10),
            "name": "ReduceLROnPlateau_Scheduler_D",
            "interval": "epoch",
            }
        return [optimizer_sig, optimizer_D], [lr_scheduler_sig, lr_scheduler_D]
    
    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None):
    #     if optimizer_idx == 0:
    #         if epoch % self.inp['num_ep_doub'][0] < self.num_w:
    #             optimizer.step(closure=optimizer_closure)
    #     elif optimizer_idx == 1: 
    #         if epoch % :
    #             optimizer.step(closure = optimizer_closure)


############################################### old definition of pretrained net (before david + exp. data) ###################################


class FFNN_pretrain(nn.Module):
    def __init__(self, inp, my_pretrained_model):
        super(FFNN_pretrain, self).__init__()
        self.inp = inp
        self.activation_new = getattr(nn, self.inp['activation'])
        self.hidden_layers_new = ast.literal_eval(self.inp['hidden_layers_new'])

        # Modify pretrained model to remove the last layer and freeze parameters of pretrained model
        if self.inp['fourier_mapping']:
            self.fourier_mapping = my_pretrained_model.ffnn.fourier_mapping
            self.pretrained_ffnn = nn.Sequential(*list(my_pretrained_model.ffnn.layers.children())[:-2])
        elif not self.inp['DeepONet']:
            self.pretrained_ffnn = nn.Sequential(*list(my_pretrained_model.layers.children())[:-2])
        elif self.inp['DeepONet']:
            self.branch_net = nn.Sequential(*list(my_pretrained_model.branch_net.layers.children()[:,-2]))
            self.trunk_net = nn.Sequential(*list(my_pretrained_model.trunk_net.layers.children()[:,-2]))
            # TODO! --> also adjust in forward function...
            print('This is not yet implemented.')
            pass

        for param in self.pretrained_ffnn.parameters():
            param.requires_grad = False
        
        new_last_layer = self.pretrained_ffnn[-1]
        pretrain_out_size = new_last_layer.out_features
        
        # Add new layers
        new_layers = []
        for i in range(len(self.hidden_layers_new)-1):
            new_layers.append(nn.Dropout(self.inp['dropout_rate']))
            if self.inp['BatchNorm_new']:
                new_layers.append(nn.BatchNorm1d(self.hidden_layers_new[i]))
            new_layers.append(nn.Linear(self.hidden_layers_new[i], self.hidden_layers_new[i + 1]))
            new_layers.append(self.activation_new())
        
        self.new_layers = nn.Sequential(
            nn.Linear(pretrain_out_size, self.hidden_layers_new[0]),
            self.activation_new(),
            *new_layers,
            nn.Linear(self.hidden_layers_new[-1], self.inp['out_size'])
        )

    def forward(self, x):
        if self.inp['fourier_mapping']:
            x = self.fourier_mapping(x)
        x = self.pretrained_ffnn(x)
        x = self.new_layers(x)
        return x