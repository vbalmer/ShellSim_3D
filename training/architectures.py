#vb, 26.03.2026

import torch
from torch import nn
import ast
from torch import vmap


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################################### Loss classes ###############################################

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





############################################### Auxiliary functions ###############################################

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



############################################### Architecture classes ###############################################


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
    


############################################### Optimiser class ###############################################

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
    


############################################### Custom Loss ###############################################


class CustomLosses:
    def __init__(self, inp:dict, ffnn:FFNN):
        super().__init__()
        self.ffnn = ffnn
        self.inp = inp
        self.criterion = loss_mapping[self.inp['loss_type']]()

    def Sobolev_CustomLoss(self, features, labels):
        if not self.inp['Sobolev']:
            raise SyntaxError("To access Sobolev_CustomLoss please set inp['Sobolev'] = True")
        
        ###### calculate "first order loss" ######
        loss1 = self.get_first_order_loss(features, labels)

        ####### calculate "second order loss" #####
        loss2 = self.get_second_order_loss(features, labels)
       
        ######## stitch together according to weights #######
        
        loss, loss_logs = self.get_combined_loss(loss1, loss2)

        return loss, loss_logs
    
    comp_SobolevCustom = torch.compile(Sobolev_CustomLoss)


    def get_first_order_loss(self, features, labels):
        features.requires_grad_(True)
        predictions = self.ffnn(features)

        geom_size = self.inp['input_size']-self.inp['out_size']
        loss1 = self.criterion(predictions, labels[:,0:self.inp['input_size']-geom_size])

        return loss1
    
    def get_second_order_loss(self, features,labels):
        # J = torch.cat([torch.autograd.functional.jacobian(self.ffnn.forward, features[i:i+1], create_graph = True) for i in range(len(features))], dim=0)[:,:,0,:]
        J = vmap(torch.func.jacrev(self.ffnn.forward), randomness='different')(features)

        sz = self.inp['out_size']
        if len(J.shape)>3:
            J = J.squeeze(1)
        # to calculate the loss directly based on the entire stiffness matrix (including the values that should be zero)
        loss2 = self.criterion(J[:,:sz,:sz], labels[:,sz:].reshape(-1,sz,sz))

        return loss2
    

    def get_combined_loss(self, loss1, loss2):        

        if self.inp['w_s'] == 'max':
            loss = max(loss1, loss2)
        else:
            w_s = ast.literal_eval(self.inp['w_s'])
            loss = w_s[0]*loss1 + w_s[1]*loss2
        
        loss_logs = {"loss1": loss1,
                     "loss2": loss2,
                     "train_loss": loss}
        
        return loss, loss_logs