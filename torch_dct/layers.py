
import math

import torch
import torch.nn as nn
import _dct as dct

class ACDC(nn.Module):
    """
    A structured efficient layer, consisting of four steps:
        1. Scale by diagonal matrix
        2. Discrete Cosine Transform
        3. Scale by diagonal matrix
        4. Inverse Discrete Cosine Transform
    """
    def __init__(self, in_features, out_features, groups=1, bias=True):
        super(ACDC, self).__init__()
        self.in_features, self.out_features = in_features, out_features

        assert in_features == out_features, "output size must equal input"
        self.A = nn.Parameter(torch.Tensor(1, in_features))
        self.D = nn.Parameter(torch.Tensor(1, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1,out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.groups = groups
        self.pack, self.unpack = PackGroups(groups), UnPackGroups(groups)

    def reset_parameters(self):
        # used in original code: https://github.com/mdenil/acdc-torch/blob/master/FastACDC.lua
        self.A.data.normal_(1., 1e-2)
        self.D.data.normal_(1., 1e-2)
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.out_features)
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        n, d = x.size()
        x = self.A*x # first diagonal matrix
        x = self.pack(x)
        x = dct.dct(x) # forward DCT
        x = self.unpack(x)
        x = self.D*x # second diagonal matrix
        x = self.pack(x)
        x = dct.idct(x) # inverse DCT
        x = self.unpack(x)
        if self.bias is not None:
            return x + self.bias
        else:
            return x


class Riffle(nn.Module):
    def forward(self, x):
        # based on shufflenet shuffle
        # and https://en.wikipedia.org/wiki/Shuffling#Riffle
        n, d = x.data.size()
        assert d%2 == 0, "dim must be even, was %i"%d
        groups = d//2
        x = x.view(n, groups, 2).permute(0,2,1).contiguous()
        return x.view(n, d)


class PackGroups(nn.Module):
    def __init__(self, groups):
        super(PackGroups, self).__init__()
        self.groups = groups

    def forward(self, x):
        n, d = x.size()
        return x.view(n*self.groups, d//self.groups)


class UnPackGroups(nn.Module):
    def __init__(self, groups):
        super(UnPackGroups, self).__init__()
        self.groups = groups

    def forward(self, x):
        n, d = x.size()
        return x.view(n//self.groups, d*self.groups)


class PadLinearTo(nn.Linear):
    """Pad by concatenating a linear layer."""
    def __init__(self, input_features, to):
        super(PadLinearTo, self).__init__(input_features, to-input_features, bias=False)
    
    def forward(self, x):
        pad = super(PadLinearTo, self).forward(x)
        return torch.cat([x, pad], 1)


class DropLinearTo(nn.Linear):
    """Drop dimensions after providing shortcut by Linear Layer. Not expecting
    to use this much."""
    def __init__(self, input_features, to):
        super(DropLinearTo, self).__init__(input_features-to, to, bias=False)
        self.to = to

    def forward(self, x):
        #residual = super(DropLinearTo, self).forward(x[:,self.to:])
        return x[:, :self.to] #+ residual


class StackedACDC(nn.Module):
    """
    A series of ACDC layers, with batchnorm, relu and riffle shuffles in between.
    Input is divided into groups, groups are rounded to nearest power of 2 and
    padding or dropping groups is used to map between different input sizes.
    """
    def __init__(self, in_features, out_features, n_layers, groups=1):
        super(StackedACDC, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.n_layers = n_layers
        # for non-matching input/output sizes
        if in_features != out_features:
            # nearest power of 2 in input groups
            group_size = 2**(math.ceil(math.log(in_features//groups,2)))
            # number of groups we'll need at output (minimum)
            n_groups_out = math.ceil(float(out_features)/group_size)
            # how many more is this than we start with?
            n_groups_in = math.ceil(float(in_features)/group_size)
            n_groups_diff = n_groups_out - n_groups_in
            # evenly spread the steps in groups over the number of layers we have
            steps = [n_groups_in+round(n_groups_diff*i/float(n_layers+1))
                     for i in range(1,n_layers+1)]
            # steps in dimensionality
            dim_steps = [group_size*s for s in steps]
        else:
            dim_steps = [in_features]*n_layers
        layers = []
        d = in_features
        for n, d_to in zip(range(n_layers), dim_steps):
            if d_to > d:
                layers.append(PadLinearTo(d, d_to))
            elif d_to < d:
                layers.append(DropLinearTo(d, d_to))
            d = d_to
            acdc = ACDC(d, d, groups=groups, bias=True)
            bn = nn.BatchNorm1d(d, affine=False)
            riffle = Riffle()
            relu = nn.ReLU()
            layers += [acdc, bn, riffle, relu]
        # remove the last relu
        _ = layers.pop(-1)
        layers.append(DropLinearTo(d, self.out_features))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

if __name__ == '__main__':
    x = torch.Tensor(128,200)
    x.normal_(0,1)
    acdc = ACDC(200,200)
    y = x
    for i in range(10):
        y = acdc(y)
    print(y.mean()) # tends to zero?
    print(torch.sqrt(y.var(1)).mean(0)) # doesn't tend to one? not good

    acdc = StackedACDC(200,400,12, groups=4)
    y = x
    y = acdc(y)
    print(y.mean()) # tends to zero?
    print(torch.sqrt(y.var(1)).mean(0)) # doesn't tend to one? not good
    print(y.size())

    # speed test
    import timeit
    setup = "from __main__ import ACDC; import torch; x = torch.Tensor(1000,4096);  model = %s; model = model.to('cuda').eval(); x = x.to('cuda'); x.normal_(0,1)"
    print("Linear: ", timeit.timeit("_ = model(x)", setup=setup%"torch.nn.Linear(4096,4096)", number=100))
    print("ACDC: ", timeit.timeit("_ = model(x)", setup=setup%"ACDC(4096,4096)", number=100))

