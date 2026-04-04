import torch.nn as nn
import torch
from lib.networks.vision_mamba2.mamba2 import VMAMBA2Block

class CircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4):
        super(CircConv, self).__init__()

        self.n_adj = n_adj
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj*2+1)

    def forward(self, input, adj, poly=None):
        input = torch.cat([input[..., -self.n_adj:], input, input[..., :self.n_adj]], dim=2)
        L = None
        return self.fc(input), L


class DilatedCircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4, dilation=1):
        super(DilatedCircConv, self).__init__()

        self.n_adj = n_adj
        self.dilation = dilation
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj*2+1, dilation=self.dilation)

    def forward(self, input, adj, poly=None):
        if self.n_adj != 0:
            input = torch.cat([input[..., -self.n_adj*self.dilation:], input, input[..., :self.n_adj*self.dilation]], dim=2)
        L = None
        return self.fc(input), L


_conv_factory = {
    'grid': CircConv,
    'dgrid': DilatedCircConv,
    'vm2': VMAMBA2Block
}



class BasicBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim, conv_type, n_adj=4, dilation=1):
        super(BasicBlock, self).__init__()
        self.state_dim = state_dim
        self.out_state_dim = out_state_dim
        self.conv = _conv_factory[conv_type](state_dim, out_state_dim, n_adj, dilation)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(out_state_dim)

    def forward(self, x, adj=None, poly=None):  # adj为环形卷积的邻接点个数
        x, L = self.conv(x, adj, poly) # x（105,66,128/40）
        x = self.relu(x)
        x = self.norm(x)
        # y = self.state_dim
        # z = self.out_state_dim
        return x, L


class Snake(nn.Module):
    def __init__(
        self,
        state_dim,
        feature_dim,
        conv_type='dgrid',
        res_layers: int = 7,
        fusion_dim: int = 256,
    ):
        super(Snake, self).__init__()

        self.head = BasicBlock(feature_dim, state_dim, 'dgrid')

        self.res_layer_num = int(res_layers)
        if self.res_layer_num == 7:
            dilation = [1, 1, 1, 2, 2, 4, 4]
        else:
            dilation = [1] * self.res_layer_num
        for i in range(self.res_layer_num):
            conv = BasicBlock(state_dim, state_dim, conv_type=conv_type, n_adj=4, dilation=dilation[i])
            self.__setattr__('res'+str(i), conv)

        fusion_state_dim = int(fusion_dim)
        self.fusion = nn.Conv1d(state_dim * (self.res_layer_num + 1), fusion_state_dim, 1)
        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 1)
        )

    def forward(self, x, adj, polys=None):  # adj为环形卷积的邻接点个数
        states = []

        x, _ = self.head(x, adj, polys)
        states.append(x)
        for i in range(self.res_layer_num):
            x, L = self.__getattr__('res'+str(i))(x, adj, polys)
            x += states[-1]
            states.append(x)

        state = torch.cat(states, dim=1)
        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
        global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))
        state = torch.cat([global_state, state], dim=1)  # (105,1280,128/40)
        x = self.prediction(state)

        return x, L
