import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class DRNet(nn.Module):
    def __init__(self, args):
        super(DRNet, self).__init__()
        self.args = args
        self.NetE = EncoderNet(self.args['enc']['input_dim'], self.args['enc']['hidden_dims'],
                               self.args['enc']['output_dim'])
        
        self.NetP1 = PredNet(self.args['pred']['input_dim'], self.args['pred']['hidden_dims'],
                             self.args['pred']['output_dim'])
        self.NetP2 = PredNet(self.args['pred']['input_dim'], self.args['pred']['hidden_dims'],
                             self.args['pred']['output_dim'])
        self.NetP3 = PredNet(self.args['pred']['input_dim'], self.args['pred']['hidden_dims'],
                             self.args['pred']['output_dim'])
        self.NetP4 = PredNet(self.args['pred']['input_dim'], self.args['pred']['hidden_dims'],
                             self.args['pred']['output_dim'])
        self.NetP5 = PredNet(self.args['pred']['input_dim'], self.args['pred']['hidden_dims'],
                             self.args['pred']['output_dim'])

        self.optimizer = torch.optim.Adam(
            params=list(self.NetP1.parameters()) + list(self.NetP2.parameters()) + list(self.NetP3.parameters()) + list(self.NetP4.parameters()) +
                   list(self.NetP5.parameters()) + list(self.NetE.parameters()),
            lr=self.args['lr'])
    def forward(self, x, t):
        z = self.NetE(x)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ys = torch.zeros(t.shape[0], 5).to(device)
        ys[:, 0] = torch.squeeze(self.NetP1(z, t))
        ys[:, 1] = torch.squeeze(self.NetP2(z, t))
        ys[:, 2] = torch.squeeze(self.NetP3(z, t))
        ys[:, 3] = torch.squeeze(self.NetP4(z, t))
        ys[:, 4] = torch.squeeze(self.NetP5(z, t))
        y_hat = ys.gather(1, torch.floor((t-1e-10)*5).long())
        return y_hat

    def get_loss(self, y, y_hat):
        return F.mse_loss(torch.unsqueeze(y, 1), y_hat)

    def backward(self, y, y_hat):
        self.optimizer.zero_grad()
        loss = self.get_loss(y, y_hat)
        loss.backward()
        self.optimizer.step()


class EncoderNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU(), dropout=0.5):
        super(EncoderNet, self).__init__()
        self.seqenc = None

        layers = list()
        all_layers_dim = copy.deepcopy(hidden_dims)
        all_layers_dim.insert(0, input_dim)

        for i in range(len(all_layers_dim)-1):
            layers.append(nn.Linear(all_layers_dim[i], all_layers_dim[i + 1]))
            layers.append(activation)
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(all_layers_dim[-1], output_dim))
        self.seqenc = nn.Sequential(*layers)

    def forward(self, x):
        return self.seqenc(x)

class PredNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, activation=nn.ReLU(), dropout=0.5):
        super(PredNet, self).__init__()
        self.seq_pred = None
        layers = list()
        all_layers_dim = copy.deepcopy(hidden_dims)
        all_layers_dim.insert(0, input_dim)

        for i in range(len(all_layers_dim) - 1):
            layers.append(nn.Linear(all_layers_dim[i] + 1, all_layers_dim[i + 1]))
            layers.append(activation)
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(all_layers_dim[-1] + 1, output_dim))
        self.seq_pred = nn.Sequential(*layers)


    def forward(self, x, t):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        modules = [module for module in self.seq_pred.modules() if not isinstance(module, nn.Sequential)]
        for l in modules:
            if isinstance(l, nn.Linear):
                x = l(torch.cat((x, t), dim=1))
            else:
                x = l(x)
        return x
