
import math
from functools import reduce

import torch
import torch.nn as nn
import pytorch_acdc as dct

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

        self.riffle = Riffle()

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
        x = self.riffle(x)
        x = dct.idct(x) # inverse DCT
        x = self.unpack(x)
        if self.bias is not None:
            return x + self.bias
        else:
            return x


class BlockDiagonalACDC(nn.Module):
    def __init__(self, in_features, out_features, groups=1, bias=True):
        super(BlockDiagonalACDC, self).__init__()
        self.in_features, self.out_features = in_features, out_features

        self.groups = groups

        assert in_features == out_features, "output size must equal input"
        c = self.in_features
        self.A = nn.Conv1d(c, c, 1, bias=False, groups=groups)
        self.D = nn.Conv1d(c, c, 1, bias=False, groups=groups)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1,out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.riffle = Riffle()

    def reset_parameters(self):
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.out_features)
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        n, d = x.size()
        x = self.A(x.view(n,d,1)) # first block diagonal matrix
        x = dct.dct(x.view(n,d)) # forward DCT
        x = self.D(x.view(n,d,1)) # second block diagonal matrix
        x = dct.idct(x.view(n,d)) # inverse DCT
        x = self.riffle(x)
        if self.bias is not None:
            return x + self.bias
        else:
            return x


class LinearACDC(nn.Linear):
    """Implement an ACDC layer in one matrix multiply (but more matrix
    operations for the parameterisation of the matrix)."""
    def __init__(self, in_features, out_features, bias=False, original=False):
        #assert in_features == out_features, "output size must equal input"
        assert out_features >= in_features, "%i must be greater than %i"%(out_features, in_features)
        assert out_features%in_features == 0
        self.expansion = out_features//in_features
        super(LinearACDC, self).__init__(in_features, out_features, bias=bias)
        self.riffle = Riffle()
        self.original = original # whether to use original parameterisation

    def reset_parameters(self):
        super(LinearACDC, self).reset_parameters()
        # this is probably not a good way to do this
        if 'A' not in self.__dict__.keys():
            self.A = nn.Parameter(torch.Tensor(self.out_features, 1))
            self.D = nn.Parameter(torch.Tensor(self.out_features, 1))
        self.A.data.normal_(1., 1e-2)
        self.D.data.normal_(1., 1e-2)
        # need to have DCT matrices stored for speed
        # they have to be Parameters so they'll be 
        N = self.out_features
        self.dct = dct.dct(torch.eye(N))
        self.idct = dct.idct(torch.eye(N))
        # remove weight Parameter
        del self.weight

    def forward(self, x):
        n, d = x.size()
        if self.expansion > 1:
            x = x.repeat(1, self.expansion)
        self.dct = self.dct.to(self.A.device)
        AC = self.A*self.dct
        self.idct = self.idct.to(self.D.device)
        DC = self.D*self.idct
        if self.original:
            ACDC = torch.matmul(AC,DC)
        else:
            ACDC = torch.matmul(self.riffle(AC),DC)
        self.weight = ACDC.t() # monkey patch
        return super(LinearACDC, self).forward(x)

def kernel_matrix_to_weights(W, c_out, c_in, k):
    """Maps to 4D weight tensor from the kernel matrix used in im2col."""
    assert k == 1 # yeah this function is quite pointless now
    return W.view(c_out, c_in, k, k)

class ConvACDC(nn.Conv2d):
    """Implements an ACDC convolutional layer by replacing the weights in a
    convolutional layer with the effective weights of an ACDC layer. After
    replacing the weights it operates precisely like a convolutional layer."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
            padding=0, dilation=1, groups=1, bias=False, original=False):
        assert out_channels >= in_channels, "channels: %i must be greater than %i"%(out_channels, in_channels)
        assert out_channels%in_channels == 0
        assert bias == False # likely to accidentally set this and break things
        assert groups == 1       
        self.expansion = out_channels//in_channels
        if kernel_size == 1:
            super(ConvACDC, self).__init__(in_channels, out_channels,
                    kernel_size, stride=stride, padding=padding,
                    dilation=dilation, groups=groups, bias=bias)
        elif kernel_size > 1:
            super(ConvACDC, self).__init__(out_channels, out_channels, 1,
                    groups=1, bias=bias)
        if kernel_size > 1:
            self.grouped = nn.Conv2d(in_channels, in_channels, kernel_size,
                    stride=stride, padding=padding, dilation=dilation,
                    groups=in_channels, bias=False)
        self.riffle = Riffle()
        self.original = original

    def reset_parameters(self):
        super(ConvACDC, self).reset_parameters()
        # this is probably not a good way to do this
        assert self.kernel_size[0] == self.kernel_size[1], "%s"%self.kernel_size
        N = self.out_channels
        if 'A' not in self.__dict__.keys():
            self.A = nn.Parameter(torch.Tensor(N, 1))
            self.D = nn.Parameter(torch.Tensor(N, 1))
        self.A.data.normal_(1., 1e-2)
        self.D.data.normal_(1., 1e-2)
        # initialise DCT matrices
        self.dct = dct.dct(torch.eye(N))
        self.idct = dct.idct(torch.eye(N))
        # remove weight Parameter
        del self.weight

    def acdc(self, device):
        k = self.kernel_size[0]
        c_out = self.out_channels
        # check our stored DCT matrices are on the right device
        if self.dct.device != device:
            self.dct = self.dct.to(device)
            self.idct = self.idct.to(device)
        AC = self.A*self.dct
        DC = self.D*self.idct
        if self.original:
            return torch.matmul(AC, DC) 
        else:
            return torch.matmul(self.riffle(AC), DC)

    def forward(self, x):
        if hasattr(self, 'grouped'):
            x = self.grouped(x)
        n, c_in, h, w = x.size()
        k = self.kernel_size[0]
        c_in, c_out = self.in_channels, self.out_channels
        if self.expansion > 1:
            x = x.repeat(1, self.expansion, 1, 1)
        ACDC = self.acdc(x.device)
        self.weight = kernel_matrix_to_weights(ACDC, c_out, c_in, k)
        return super(ConvACDC, self).forward(x)


class Riffle(nn.Module):
    def forward(self, x):
        # based on shufflenet shuffle
        # and https://en.wikipedia.org/wiki/Shuffling#Riffle
        dim = x.dim()
        if dim == 2:
            n, d = x.data.size()
            assert d%2 == 0, "dim must be even, was %i"%d
            groups = d//2
            x = x.view(n, groups, 2).permute(0,2,1).contiguous()
            return x.view(n, d)
        elif dim == 4:
            N,C,H,W = x.size()
            g = 2
            return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)
        else:
            raise ValueError("Shape of x not supported: %s"%x.size())


class Permute(nn.Module):
    """Assuming 2d input, permutes along last dimension using a fixed
    permutation."""
    def __init__(self, d):
        self.d = d
        super(Permute, self).__init__()
        self.reset_parameters()
        
    def reset_parameters(self):
        self.permute_idxs = torch.randperm(self.d)

    def to(self, device):
        self.permute_idxs.to(device)
        super(Permute, self).to(device)

    def forward(self, x):
        return x[:,self.permute_idxs]

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
            acdc = ACDC(d, d, groups=groups, bias=False)
            #bn = nn.BatchNorm1d(d, affine=False)
            riffle = Riffle()
            #relu = nn.ReLU()
            layers += [acdc, riffle]
        # remove the last relu
        #_ = layers.pop(-1)
        if self.out_features < d:
            layers.append(DropLinearTo(d, self.out_features))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class StackedLinearACDC(nn.Module):
    def __init__(self, in_features, out_features, n_layers, bias=False,
            original=False):
        super(StackedLinearACDC, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        assert out_features%in_features == 0
        self.n_layers = n_layers

        layers = []
        d = in_features
        for n in range(n_layers):
            acdc = LinearACDC(d, out_features,
                    bias=False if n < n_layers-1 else bias, original=original)
            d = out_features
            permute = Riffle()
            relu = nn.ReLU()
            layers += [acdc, permute]
        # remove the last relu
        # _ = layers.pop(-1)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class StackedConvACDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_layers, stride=1,
            padding=0, dilation=1, groups=1, bias=True):
        super(StackedConvACDC, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        assert out_channels%in_channels == 0
        self.n_layers = n_layers

        layers = []
        d = in_channels
        for n in range(n_layers):
            acdc = ConvACDC(d, out_channels, kernel_size,
                    stride=stride if n==0 else 1, padding=padding,
                    dilation=dilation, groups=groups, bias=bias)
            d = out_channels
            permute = Riffle()
            relu = nn.ReLU()
            layers += [acdc, permute, relu]
        # remove the last relu
        _ = layers.pop(-1)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ChannelContract(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelContract, self).__init__()
        assert in_channels%out_channels == 0, \
                f"{in_channels} not divisible by {out_channels}"
        self.in_channels = in_channels
        self.out_channels = out_channels
    def forward(self, x):
        n, c, h, w = x.size()
        f = self.in_channels//self.out_channels
        x = x.view(n, c//f, f, h, w)
        return x.sum(2)


class ChannelExpand(nn.Module):
    """Concatenate channels to expand by `c_to_add` channels."""
    def __init__(self, c_to_add):
        super(ChannelExpand, self).__init__()
        self.c = c_to_add
    def forward(self,x):
        x_add = x[:,:self.c,:,:]
        return torch.cat([x,x_add],1)


class FastStackedConvACDC(nn.Conv2d):
    """A Stacked ACDC layer that just combines all of the weight marices of all
    of the layers in the stack before implementing the layer with a
    convolution. This means that there is no ReLUs in it, though, which may
    hinder representational capacity."""
    def __init__(self, in_channels, out_channels, kernel_size, n_layers, stride=1,
            padding=0, dilation=1, groups=1, bias=True, original=False):
        self.n_layers = n_layers
        if kernel_size == 1:
            super(FastStackedConvACDC, self).__init__(in_channels,
                    out_channels, kernel_size, stride=stride, padding=padding,
                    dilation=dilation, groups=groups, bias=bias)
        elif kernel_size > 1:
            assert groups == 1
            super(FastStackedConvACDC, self).__init__(in_channels,
                    out_channels, 1, bias=bias)
            self.grouped = nn.Conv2d(in_channels, in_channels, kernel_size,
                    stride=stride, padding=padding, dilation=dilation,
                    groups=in_channels, bias=False)
        if out_channels > in_channels:
            add_channels = 0
            while out_channels%(in_channels+add_channels) != 0:
                add_channels += 1
            self.expand_channels = ChannelExpand(add_channels)
            self.in_channels += add_channels
            in_channels = self.in_channels
        else:
            self.expand_channels = lambda x: x
        self.expansion = out_channels//in_channels
        layers = []
        for n in range(n_layers):
            channels = max(out_channels, in_channels)
            acdc = ConvACDC(channels, channels, 1, bias=bias,
                    original=original)
            layers += [acdc]
        # remove the last relu
        self.permute = Riffle()
        _ = layers.pop(-1)
        self.layers = nn.Sequential(*layers)
        if out_channels < in_channels:
            self.collapse = ChannelContract(in_channels, out_channels)
        else:
            self.collapse = lambda x: x

    def reset_parameters(self):
        del self.weight

    def forward(self, x):
        if hasattr(self, 'grouped'):
            x = self.grouped(x)
        x = self.expand_channels(x)
        if self.expansion > 1:
            x = x.repeat(1, self.expansion, 1, 1)
        k = self.kernel_size[0]
        c = max(self.out_channels, self.in_channels)
        # gather ACDC matrices from each layer
        acdcs = [layer.acdc(x.device) for layer in self.layers]
        # riffle them all
        acdcs = [self.permute(ACDC) for ACDC in acdcs]
        # and combine them
        ACDC = reduce(torch.matmul, acdcs)
        self.weight = kernel_matrix_to_weights(ACDC, c, c, k)
        return self.collapse(super(FastStackedConvACDC, self).forward(x))


if __name__ == '__main__':
    # check ConvACDC
    x = torch.Tensor(16,128,8,8)
    x.normal_(0,1)
    conv_acdc = ConvACDC(128,128,3)
    assert not hasattr(conv_acdc, 'weight')
    param_names = [n for n,p in conv_acdc.named_parameters()]
    assert 'weight' not in param_names, param_names
    _ = conv_acdc(x)
    param_names = [n for n,p in conv_acdc.named_parameters()]
    assert 'weight' not in param_names, param_names
    assert False

    x = torch.Tensor(128,200)
    x.normal_(0,1)
    acdc = ACDC(200,200,bias=False)
    y = x
    for i in range(10):
        y = acdc(y)
    print(y.mean()) # tends to zero?
    print(torch.sqrt(y.var(1)).mean(0)) # doesn't tend to one? not good

    # check sanity of LinearACDC
    lin_acdc = LinearACDC(200,200)
    lin_acdc.A.data.fill_(1.)
    lin_acdc.D.data.fill_(1.)
    acdc.A.data.fill_(1.)
    acdc.D.data.fill_(1.)
    error = torch.abs(acdc(x) - lin_acdc(x)).max() 
    print("LienarACDC error", error.item())
    assert error < 1e-3

    acdc = StackedACDC(200,400,12, groups=4)
    y = x
    y = acdc(y)
    print(y.mean()) # tends to zero?
    print(torch.sqrt(y.var(1)).mean(0)) # doesn't tend to one? not good
    print(y.size())

    # speed test
    import timeit
    setup = "from __main__ import ACDC,LinearACDC; import torch; x = torch.Tensor(1000,4096);  model = {0}(4096,4096); model = model.to('cuda').eval(); x = x.to('cuda'); x.normal_(0,1)"
    print("Linear: ", timeit.timeit("_ = model(x)", setup=setup.format("torch.nn.Linear"), number=100))
    print("ACDC: ", timeit.timeit("_ = model(x)", setup=setup.format("ACDC"), number=100))
    print("Linear ACDC: ", timeit.timeit("_ = model(x)", setup=setup.format("LinearACDC"), number=100))

    setup = "from __main__ import StackedConvACDC,FastStackedConvACDC; import torch; x = torch.Tensor(100,256,4,4); model = {0}(256,256,1,12,bias=False); model=model.to('cuda').eval(); x = x.to('cuda'); x.normal_(0,1)"
    print("StackedConvACDC: ", timeit.timeit("_ = model(x)", setup=setup.format("StackedConvACDC"), number=100))
    print("FastStackedConvACDC: ", timeit.timeit("_ = model(x)", setup=setup.format("FastStackedConvACDC"), number=100))
