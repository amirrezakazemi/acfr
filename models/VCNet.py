import torch
import torch.nn as nn
import torch.nn.functional as F
import copy



class VCNet(nn.Module):
    def __init__(self, args):
        super(VCNet, self).__init__()
        self.args = args
        self.NetE = EncoderNet(self.args['enc']['input_dim'], self.args['enc']['hidden_dims'],
                               self.args['enc']['output_dim'])
        self.NetP = DynamicPredNet(self.args['pred']['input_dim'], self.args['pred']['hidden_dims'],
                                   self.args['pred']['output_dim'])
        self.NetD = DensityNet(self.args['dens']['input_dim'], self.args['dens']['hidden_dims'],
                               self.args['dens']['output_dim'])
        self.optimizer = torch.optim.Adam(
            params=list(self.NetP.parameters()) + list(self.NetE.parameters()) + list(self.NetD.parameters()), lr=args['lr'])

    def forward(self, x, t):
        z = self.NetE(x)
        gps = self.NetD(z, t)
        y_hat = self.NetP(z, t)
        return y_hat, z, gps

    def get_loss(self, y, y_hat, gps):
        return F.mse_loss(torch.unsqueeze(y, 1), y_hat) - 0.1 * torch.mean(torch.log(gps))

    def backward(self, y, y_hat, gps):
        self.optimizer.zero_grad()
        loss = self.get_loss(y, y_hat, gps)
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



class DensityNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU(), dropout=0.5):
        super(DensityNet, self).__init__()
        self.seqdensity = None
        self. output_dim = output_dim
        layers = list()
        all_layers_dim = copy.deepcopy(hidden_dims)
        all_layers_dim.insert(0, input_dim)

        for i in range(len(all_layers_dim) - 1):
            layers.append(nn.Linear(all_layers_dim[i], all_layers_dim[i + 1]))
            layers.append(activation)
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(all_layers_dim[-1], output_dim))
        self.seqdensity = nn.Sequential(*layers)

    def forward(self, x, t):
        gps_vector = self.seqdensity(x)
        gps_vector = torch.softmax(gps_vector, dim=1)
        gps = self.lin_inter(gps_vector, t)
        #print(gps.shape)
        return gps

    def lin_inter(self, gps_vector, t):
        #print(gps_vector)
        U = torch.ceil(t * (gps_vector.shape[1]-1))
        inter = 1 - (U - t * (gps_vector.shape[1]-1))
        L = U - 1
        L += (L < 0).int()
        L_out = gps_vector.gather(1, L.long().view(-1, 1))
        U_out = gps_vector.gather(1, U.long().view(-1, 1))
        gps = torch.squeeze(L_out, 1) + torch.mul(torch.squeeze((U_out - L_out), 1),  inter)
        return gps


# Sequence of dynamic FC layers to predict y using Z and t.
class DynamicPredNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, degree=2, knots=[1 / 3, 2 / 3], activation=nn.ReLU()):
        super(DynamicPredNet, self).__init__()
        self.seqpred = None
        layers = list()
        all_layers_dim = copy.deepcopy(hidden_dims)
        all_layers_dim.insert(0, input_dim)

        for i in range(len(all_layers_dim) - 1):
            layers.append(Dynamic_FC(all_layers_dim[i], all_layers_dim[i + 1], degree, knots, activation, is_bias=True))
        layers.append(Dynamic_FC(all_layers_dim[-1], output_dim, degree, knots, None, is_bias=True, is_last_layer=True))
        self.seqpred = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.seqpred((x, t))



# Dynamic Fully Connected class
# Stacking multiple dynamic_fc will map (Z, t) to y.
class Dynamic_FC(nn.Module):
    def __init__(self, input_dim, output_dim, degree, knots, activation, is_bias=True, is_last_layer=False):
        super(Dynamic_FC, self).__init__()
        self.basis = Truncated_power(degree=degree, knots=knots)
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim, self.basis.num_of_basis), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(output_dim, self.basis.num_of_basis), requires_grad=True)
        self.is_bias = is_bias
        self.activation = activation
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
        #print("z", z)
        if self.is_last_layer:
            return z

        return z, t



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

    def forward(self, t):
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
