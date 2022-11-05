from math import pi as PI
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear
from torch_scatter import scatter
from torch_geometric.nn import radius_graph


class update_e(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters, num_gaussians, cutoff):
        super(update_e, self).__init__()
        self.cutoff = cutoff
        self.lin = Linear(hidden_channels, num_filters, bias=False)
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)

    def forward(self, v, dist, dist_emb, edge_index):
        j, _ = edge_index
        C = 0.5 * (torch.cos(dist * PI / self.cutoff) + 1.0)
        W = self.mlp(dist_emb) * C.view(-1, 1)
        v = self.lin(v)
        e = v[j] * W
        return e

class update_e2(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters, num_gaussians, cutoff):
        super(update_e2, self).__init__()
        self.cutoff = cutoff
        self.hidden_channels = hidden_channels
        self.Wsf = Linear(2*hidden_channels+num_gaussians, 2*hidden_channels)
        self.BN1 = torch.nn.BatchNorm1d(2*hidden_channels)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.Wsf.weight)
        self.Wsf.bias.data.fill_(0)

    def forward(self, v, dist, dist_emb, edge_index):
        j, i = edge_index
        x = torch.cat((v[i], v[j], dist_emb), dim=-1)
        x = self.Wsf(x)
        x = self.BN1(x)
        c = x[:, :self.hidden_channels]
        f = x[:, self.hidden_channels:]
        c = F.softplus(c)
        f = F.sigmoid(f)
        x = c * f
        C = 0.5 * (torch.cos(dist * PI / self.cutoff) + 1.0)
        x = x * C.view(-1, 1)
        return x

class mygru(torch.nn.Module):
    def __init__(self, embedfea):
        super(mygru, self).__init__()
        self.embedfea = embedfea
        self.W = Linear(2*self.embedfea, self.embedfea)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W.weight)
        self.W.bias.data.fill_(0)

    def forward(self, oa, na):
        x = torch.cat((oa, na), dim=-1)
        x = self.W(x)
        x = F.sigmoid(x)
        return x * oa + (1 - x) * na


class update_v(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters):
        super(update_v, self).__init__()
        self.act = ShiftedSoftplus()
        self.lin1 = Linear(num_filters, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.mygru = mygru(hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        self.mygru.reset_parameters()

    def forward(self, v, e, edge_index):
        _, i = edge_index
        out = scatter(e, i, dim=0)
        out = self.lin1(out)
        out = self.act(out)
        out = self.lin2(out)
        # return v + out
        return self.mygru(v, out)

class update_v2(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters):
        super(update_v2, self).__init__()
        self.BN2 = torch.nn.BatchNorm1d(hidden_channels)
        self.mygru = mygru(hidden_channels)

    def reset_parameters(self):
        self.mygru.reset_parameters()

    def forward(self, v, e, edge_index):
        _, i = edge_index
        x = scatter(e, i, dim=0)
        x = self.BN2(x)
        x = self.mygru(v, x)
        return F.softplus(x)

class update_u(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(update_u, self).__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, v, batch):
        v = self.lin1(v)
        v = self.act(v)
        v = self.lin2(v)
        u = scatter(v, batch, dim=0)
        return u

class update_u2(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(update_u2, self).__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels * 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels * 2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, v, batch, fea=False):
        u = scatter(v, batch, dim=0)
        if fea:
            return u
        u = self.lin1(u)
        u = self.act(u)
        u = self.lin2(u)
        return u

class emb(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(emb, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class emb_atom(torch.nn.Module):
    def __init__(self, iniatomfea, embedfea):
        super(emb_atom, self).__init__()
        self.iniatomfea = iniatomfea
        self.natomfea = iniatomfea.shape[1]
        self.embedfea = embedfea
        self.lin = Linear(self.natomfea, self.embedfea)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, n):
        return self.lin(self.iniatomfea[n])


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class Conv(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters, num_gaussians, cutoff):
        super(Conv, self).__init__()
        self.cutoff = cutoff
        self.hidden_channels = hidden_channels
        self.Wsf = Linear(2*hidden_channels+num_gaussians, 2*hidden_channels)
        self.BN1 = torch.nn.BatchNorm1d(2*hidden_channels)
        self.BN2 = torch.nn.BatchNorm1d(hidden_channels)
        self.mygru = mygru(hidden_channels)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.Wsf.weight)
        self.Wsf.bias.data.fill_(0)
        self.mygru.reset_parameters()

    def forward(self, v, dist, dist_emb, edge_index):
        j, i = edge_index
        x = torch.cat((v[i], v[j], dist_emb), dim=-1)
        x = self.Wsf(x)
        x = self.BN1(x)
        c = x[:, :self.hidden_channels]
        f = x[:, self.hidden_channels:]
        c = F.softplus(c)
        f = F.sigmoid(f)
        x = c * f
        C = 0.5 * (torch.cos(dist * PI / self.cutoff) + 1.0)
        x = x * C.view(-1, 1)
        x = scatter(x, i, dim=0)
        x = self.BN2(x)
        x = self.mygru(v, x)
        return F.softplus(x)


class wcgcnn(torch.nn.Module):
    r"""
        The re-implementation for SchNet from the `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper
        under the 3DGN gramework from `"Spherical Message Passing for 3D Graph Networks" <https://arxiv.org/abs/2102.05013v2>`_ paper.

        Args:
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the negative of the derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)
            num_layers (int, optional): The number of layers. (default: :obj:`6`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            num_filters (int, optional): The number of filters to use. (default: :obj:`128`)
            num_gaussians (int, optional): The number of gaussians :math:`\mu`. (default: :obj:`50`)
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`10.0`).
    """

    def __init__(self, energy_and_force=False, cutoff=10.0, num_layers=6, hidden_channels=128, num_filters=128, num_gaussians=50, Mean=None, Std=None, iniatomfea=None, device=None):
        super(wcgcnn, self).__init__()

        self.energy_and_force = energy_and_force
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians
        if Mean is None:
            self.mean = torch.tensor(0)
        else:
            self.mean = Mean
        if Std is None:
            self.std = torch.tensor(1)
        else:
            self.std = Std
        self.device = device
        self.iniatomfea = torch.tensor(
            iniatomfea, dtype=torch.float32).to(self.device)

        #self.init_v = Embedding(100, hidden_channels)
        self.init_v = emb_atom(self.iniatomfea, hidden_channels)
        self.dist_emb = emb(0.0, cutoff, num_gaussians)

        self.convblock = torch.nn.ModuleList(
            [Conv(hidden_channels, num_filters, num_gaussians, cutoff) for _ in range(num_layers)])

        # self.update_vs = torch.nn.ModuleList(
        #     [update_v(hidden_channels, num_filters) for _ in range(num_layers)])

        # self.update_es = torch.nn.ModuleList([
        #     update_e(hidden_channels, num_filters, num_gaussians, cutoff) for _ in range(num_layers)])

        self.update_u = update_u2(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.init_v.reset_parameters()
        # for update_e in self.update_es:
        #     update_e.reset_parameters()
        # for update_v in self.update_vs:
        #     update_v.reset_parameters()
        for conv in self.convblock:
            conv.reset_parameters()
        self.update_u.reset_parameters()

    def forward(self, batch_data, fea=False):
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        if self.energy_and_force:
            pos.requires_grad_()

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        dist_emb = self.dist_emb(dist)

        v = self.init_v(z)

        # for update_e, update_v in zip(self.update_es, self.update_vs):
        #     e = update_e(v, dist, dist_emb, edge_index)
        #     v = update_v(v, e, edge_index)
        for conv in self.convblock:
            v = conv(v, dist, dist_emb, edge_index)
        if fea:
            return self.update_u(v, batch, fea=fea)
        u = self.update_u(v, batch)  # ;print(v.size());exit() #[91*50,64]

        return u * self.std + self.mean
