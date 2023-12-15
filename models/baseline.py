import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, args, activation=nn.ReLU(), dropout=0.5):
        super(MLP, self).__init__()
        self.seq = None
        input_dim = args['input_dim']
        hidden_dims = args['hidden_dims']
        output_dim = args['output_dim']

        layers = list()
        all_layers_dim = hidden_dims
        all_layers_dim.insert(0, input_dim+1)
        for i in range(len(all_layers_dim) - 1):
            layers.append(nn.Linear(all_layers_dim[i], all_layers_dim[i+1]))
            layers.append(activation)
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(all_layers_dim[-1], output_dim))
        self.seq = nn.Sequential(*layers)

        self.optimizer = torch.optim.Adam(params=list(self.seq.parameters()), lr=args['lr'])

    def forward(self, x, t):
        y_hat = self.seq(torch.cat((x, t), dim=1))
        return y_hat

    def get_loss(self, y, y_hat):
        loss = F.mse_loss(y, torch.squeeze(y_hat))
        return loss

    def backward(self, y, y_hat):
        self.optimizer.zero_grad()
        loss = self.get_loss(y, y_hat)
        loss.backward()
        self.optimizer.step()

class MLP2(nn.Module):
    def __init__(self, args, activation=nn.ReLU(), dropout=0.5):
        super(MLP2, self).__init__()
        self.seq1 = None
        self.seq2 = None
        input_dim = args['input_dim']
        z_hidden_dims = args['z_hidden_dims']
        z_dim = args['z_dim']
        p_hidden_dims = args['p_hidden_dims']
        output_dim = args['output_dim']

        z_layers = list()
        all_z_layers_dim = z_hidden_dims
        all_z_layers_dim.insert(0, input_dim)
        for i in range(len(all_z_layers_dim) - 1):
            z_layers.append(nn.Linear(all_z_layers_dim[i], all_z_layers_dim[i + 1]))
            z_layers.append(activation)
            z_layers.append(nn.Dropout(p=dropout))
        z_layers.append(nn.Linear(all_z_layers_dim[-1], z_dim))
        self.seq1 = nn.Sequential(*z_layers)

        p_layers = list()
        all_p_layers_dim = p_hidden_dims
        all_p_layers_dim.insert(0, z_dim)
        for i in range(len(all_p_layers_dim) - 1):
            p_layers.append(nn.Linear(all_p_layers_dim[i]+1, all_p_layers_dim[i + 1]))
            p_layers.append(activation)
            p_layers.append(nn.Dropout(p=dropout))
        p_layers.append(nn.Linear(all_p_layers_dim[-1]+1, output_dim))
        self.seq2 = nn.Sequential(*p_layers)

        self.optimizer = torch.optim.Adam(params=list(self.seq1.parameters()) + list(self.seq2.parameters()) , lr=args['lr'])

        
    def forward(self, x, t):
        z = self.seq1(x)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t_ = torch.zeros(t.shape[0], 1).to(device)
        t_[:, 0] = torch.squeeze(torch.pow(t, 1))
        modules = [module for module in self.seq2.modules() if not isinstance(module, nn.Sequential)]
        for l in modules:
            if isinstance(l, nn.Linear):
                z = l(torch.cat((z, t_), dim=1))
            else:
                z = l(z)
        return z
    def get_loss(self, y, y_hat):
        loss = F.mse_loss(y, torch.squeeze(y_hat))
        return loss

    def backward(self, y, y_hat):
        self.optimizer.zero_grad()
        loss = self.get_loss(y, y_hat)
        loss.backward()
        self.optimizer.step()
    



