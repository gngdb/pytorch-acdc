# other linear layers for testing
import math
from functools import reduce

import torch
import torch.nn as nn

from pytorch_acdc.layers import Permute, BlockDiagonalACDC

# courtesy of https://github.com/kuangliu/pytorch-cifar/models/shufflenet.py#L10-L19
class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)


class ShuffleNetLinear(nn.Module):
    """A stack of grouped 1x1 convolutions, interleaved with riffle
    shuffles."""
    def __init__(self, in_channels, out_channels, n_layers, n_groups):
        super(ShuffleNetLinear, self).__init__()
        assert in_channels == out_channels
        channels = in_channels
        layers = []
        for i in range(n_layers):
            conv1x1 = nn.Conv2d(channels, channels, 1, groups=n_groups,
                    bias=False if i < n_layers-1 else True)
            riffle = ShuffleBlock(n_groups)
            layers += [conv1x1, riffle]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        n, d = x.size()
        x = x.view(n, d, 1, 1)
        x = self.layers(x)
        return x.view(n, d)


class AFDF(nn.Module):
    # expects complex input and gives complex output
    def __init__(self, in_channels, out_channels):
        super(AFDF, self).__init__()
        assert in_channels == out_channels
        self.A = nn.Parameter(torch.Tensor(1, out_channels, 2))
        self.D = nn.Parameter(torch.Tensor(1, out_channels, 2))
        self.reset_parameters()

    def reset_parameters(self):
        self.A.data.normal_(1., 1e-2)
        self.D.data.normal_(1., 1e-2)

    def forward(self, x):
        x = self.A*x
        f = torch.fft(x, 1)
        f = self.D*f
        return torch.ifft(f, 1)


class StackedAFDF(nn.Module):
    """Stack of AFDF layers, casting to complex and then taking the real
    component of the complex result."""
    def __init__(self, in_channels, out_channels, n_layers):
        super(StackedAFDF, self).__init__()
        assert in_channels == out_channels
        channels = in_channels
        layers = []
        for i in range(n_layers):
            afdf = AFDF(channels, channels)
            permute = Permute(channels)
            layers += [afdf, permute]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        n, d = x.size()
        x = x.view(n,d,1)
        j = torch.zeros_like(x).to(x.device)
        x = torch.cat([x,j],2)
        x = self.layers(x)
        return x[:,:,0] # only keep real component


class GroupedStackedACDC(nn.Module):
    """Stack of ACDC layers that also happen to use block diagonal weight
    matrices."""
    def __init__(self, in_channels, out_channels, n_layers, n_groups):
        super(GroupedStackedACDC, self).__init__()
        assert in_channels == out_channels
        channels = in_channels
        layers = []
        for i in range(n_layers):
            acdc = BlockDiagonalACDC(channels, channels, n_groups)
            permute = Permute(channels)
            layers += [acdc, permute]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ComplexGroupedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, groups, bias=False):
        super(ComplexGroupedLinear, self).__init__()
        assert bias == False, "bias not supported yet"
        self.A = nn.Conv1d(in_dim, out_dim, 1, groups=groups, bias=bias) # real weights
        self.B = nn.Conv1d(in_dim, out_dim, 1, groups=groups, bias=bias) # real weights

    def forward(self, x):
        n,d,c = x.size()
        r = x[:,:,[0]] # real
        j = x[:,:,[1]] # complex
        Ar = self.A(r)
        Bj = self.B(j)
        Br = self.B(r)
        Aj = self.A(j)
        return torch.cat([Ar-Bj, Br+Aj], 2) # concat real and imaginary


class BlockDiagonalAFDF(nn.Module):
    # expects complex input and gives complex output
    def __init__(self, in_channels, out_channels, groups):
        super(BlockDiagonalAFDF, self).__init__()
        assert in_channels == out_channels
        c = in_channels
        self.A = ComplexGroupedLinear(c, c, groups)
        self.D = ComplexGroupedLinear(c, c, groups)

    def forward(self, x):
        x = self.A(x)
        f = torch.fft(x, 1)
        f = self.D(f)
        return torch.ifft(f, 1)


class GroupedStackedAFDF(nn.Module):
    """Stack of AFDF layers that also happen to use block diagonal weight
    matrices."""
    def __init__(self, in_channels, out_channels, n_layers, n_groups):
        super(GroupedStackedAFDF, self).__init__()
        assert in_channels == out_channels
        channels = in_channels
        layers = []
        for i in range(n_layers):
            afdf = BlockDiagonalAFDF(channels, channels, n_groups)
            permute = Permute(channels)
            layers += [afdf, permute]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        n, d = x.size()
        x = x.view(n,d,1)
        j = torch.zeros_like(x).to(x.device)
        x = torch.cat([x,j],2)
        x = self.layers(x)
        return x[:,:,0] # only keep real component
