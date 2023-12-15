import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

class ACFR(nn.Module):
    def __init__(self, args):
        super(ACFR, self).__init__()
        self.args = args
        self.gamma = self.args['gamma']
        self.NetE = EncoderNet(self.args['enc']['input_dim'], self.args['enc']['hidden_dims'],
                               self.args['enc']['output_dim'])

        # self.NetP = RBFPredNet(self.args['pred']['input_dim'], self.args['pred']['hidden_dims'],
        #                            self.args['pred']['output_dim'], self.args['pred']['degree'])

        self.NetP = CrossAttention(input_dim = self.args['atten']['input_dim'], hidden_dim=self.args['atten']['input_dim'], 
                                    output_dim=self.args['atten']['output_dim'], degree=self.args['atten']['degree'])

        self.NetD = DiscNet(self.args['disc']['input_dim'], self.args['disc']['hidden_dims'],
                            self.args['disc']['output_dim'])
        ## G means combination of Encoder and Predictor
        G_parameters = list(self.NetE.parameters()) + list(self.NetP.parameters())
        self.optimizer_G = torch.optim.Adam(params=G_parameters, lr=args['lr1'])
        self.optimizer_D = torch.optim.Adam(params=self.NetD.parameters(), lr=args['lr2'])
        # Access the parameter groups
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.99)
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.99)
    
    def forward_G(self, x, t):
        z = self.NetE(x)
        y_hat1 = self.NetP(z, t)
        t_hat = self.NetD(z)

        return y_hat1, t_hat

    def forward_D(self, x):
        z = self.NetE(x)
        t_hat = self.NetD(z)
        return t_hat

    def forward(self, x, t):
        z = self.NetE(x)
        y_hat = self.NetP(z, t)
        return y_hat, z

    def backward_G(self, t, y, t_hat, y_hat1):
        for name, param in self.NetP.state_dict().items():
            param.requires_grad = True
        for name, param in self.NetE.state_dict().items():
            param.requires_grad = True
        for name, param in self.NetD.state_dict().items():
            param.requires_grad = False

        self.optimizer_G.zero_grad()

        g1 = F.mse_loss(torch.unsqueeze(y, 1), y_hat1)
        g2 = F.mse_loss(torch.unsqueeze(t, 1), t_hat)
        G_loss = F.relu(g1 - self.gamma * g2)
        G_loss.backward()
        self.optimizer_G.step()
        return G_loss

    def backward_D(self, t, t_hat):
        for name, param in self.NetP.state_dict().items():
            param.requires_grad = False
        for name, param in self.NetE.state_dict().items():
            param.requires_grad = False
        for name, param in self.NetD.state_dict().items():
            param.requires_grad = True
        self.optimizer_D.zero_grad()
        D_loss = F.mse_loss(torch.unsqueeze(t, 1), t_hat)
        D_loss.backward()
        self.optimizer_D.step()
        return D_loss


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


class DiscNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=2, activation=nn.ReLU(), dropout=0.5):
        super(DiscNet, self).__init__()
        self.seqdisc = None

        layers = list()
        all_layers_dim = copy.deepcopy(hidden_dims)
        all_layers_dim.insert(0, input_dim)

        for i in range(len(all_layers_dim) - 1):
            layers.append(nn.Linear(all_layers_dim[i], all_layers_dim[i + 1]))
            layers.append(activation)
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(all_layers_dim[-1], output_dim))
        self.seqdisc = nn.Sequential(*layers)

    def forward(self, x):
        return self.seqdisc(x)


class RBFPredNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, degree=10, activation=nn.ReLU()) -> None:
        super(RBFPredNet, self).__init__()
        self.seqpred = None
        layers = list()
        all_layers_dim = copy.deepcopy(hidden_dims)
        all_layers_dim.insert(0, input_dim)

        for i in range(len(all_layers_dim) - 1):
            layers.append(RBF_FC(all_layers_dim[i], all_layers_dim[i + 1], degree, activation, is_bias=True))
        layers.append(RBF_FC(all_layers_dim[-1], output_dim, degree, None, is_bias=True, is_last_layer=True))
        self.seqpred = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.seqpred((x, t))

class RBF_FC(nn.Module):
    def __init__(self, input_dim, output_dim, degree, activation, is_bias=True, is_last_layer=False) -> None:
        super(RBF_FC, self).__init__()
        
        self.num_basis = degree
        mu_vec = torch.randn(degree)
        std_vec = [1]* degree
        self.basis = RBF(self.num_basis, mu_vec, std_vec)
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim, self.num_basis), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(output_dim, self.num_basis), requires_grad=True)
        self.activation = activation
        self.is_bias = is_bias
        self.is_last_layer = is_last_layer

    def forward(self, inp):
        x, t = inp
        weighted_x = torch.matmul(self.weights.T, x.T).T
        t_basis = self.basis.forward(t)
        uns_t_basis = torch.unsqueeze(t_basis, 1)
        z = torch.sum(weighted_x * uns_t_basis, dim=2)
        if self.is_bias:
            bias_z = torch.matmul(self.bias, t_basis.T).T
            z += bias_z
        if self.activation is not None:
            z = self.activation(z)
        if self.is_last_layer:
            return z

        return z, t

class RBF:
    def __init__(self, num_basis, mu_vec, std_vec) -> None:

        self.num_basis = num_basis
        self.mu_vec = mu_vec
        self.std_vec = std_vec
        self.exp = torch.exp
    
    def get_rbf(self, t):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t = t.squeeze()
        out = torch.zeros(t.shape[0], self.num_basis).to(device)

        for _ in range(self.num_basis):
            out[:, _] = self.exp( -1 *  ((t - self.mu_vec[_])**2 / self.std_vec[_]**2))
        return out


class CrossAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=1, degree=5, knots=[1/3, 2/3]):
        super(CrossAttention, self).__init__()
        
        
        self.trunc = Truncated_power(degree=degree, knots=knots)

        # degree = 10
        # mu_vec = torch.randn(degree)
        # std_vec = [1]* degree
        # self.rbf = RBF(degree, mu_vec, std_vec)

        self.hidden_dim = hidden_dim


        self.query_projection = nn.Linear(self.trunc.num_of_basis, hidden_dim)
        self.key_projection = nn.Linear(input_dim, hidden_dim)
        self.value_projection = nn.Linear(input_dim, hidden_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, t):
        t_out = self.trunc.get_spline(t)

        # Compute query, key, and value
        Q = self.query_projection(t_out)
        K = self.key_projection(x)
        V = self.value_projection(x)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))

        attention_weights = self.softmax(scores/math.sqrt(self.hidden_dim))

        # Apply attention weights to values
        output = torch.matmul(attention_weights, V)
        

        output = self.fc(output+x)

        return output

class Truncated_power:
    def __init__(self, degree=2, knots=[1 / 3, 2 / 3]):
        """
        This class construct the truncated power basis; the data is assumed in [0,1]
        :param degree: int, the degree of truncated basis
        :param knots: list, the knots of the spline basis; two end points (0,1) should not be included
        """
        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)

        if self.degree == 0:
            print('Degree should not set to be 0!')
            raise ValueError

        if not isinstance(self.degree, int):
            print('Degree should be int')
            raise ValueError

    def get_spline(self, t):
        """
        :param t: torch.tensor, batch_size * 1
        :return: the value of each basis given t; batch_size * self.num_of_basis
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t = t.squeeze()
        out = torch.zeros(t.shape[0], self.num_of_basis).to(device)
        for _ in range(self.num_of_basis):
            if _ <= self.degree:
                if _ == 0:
                    out[:, _] = 1.
                else:
                    out[:, _] = t ** _
            else:
                if self.degree == 1:
                    out[:, _] = (self.relu(t - self.knots[_ - self.degree]))
                else:
                    out[:, _] = (self.relu(t - self.knots[_ - self.degree - 1])) ** self.degree
        return out