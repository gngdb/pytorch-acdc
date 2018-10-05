import numpy as np
import torch
import torch.nn as nn

def dct1(x):
    """
    Discrete Cosine Transform, Type I

    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.rfft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1)[:, :, 0].view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I

    Our definition if idct1 is such that idct1(dct1(x)) == x

    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    B, N = x.shape
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.rfft(v, 1, onesided=False)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(B,N)

    return V


class FasterDCT(nn.Module):
    """Caches scaling variables for faster computation."""
    def __init__(self, in_features):
        super(FasterDCT, self).__init__()
        N = in_features
        k = - torch.arange(N, dtype=torch.float32)[None, :] * np.pi / (2 * N)
        self.W_r = nn.Parameter(torch.cos(k))
        self.W_i = nn.Parameter(torch.sin(k))
        self.W_r.require_grad, self.W_i.require_grad = False, False
        self.N = N
        self.pad = nn.ConstantPad1d((0,N), 0.)
        #self.reverse_idxs = torch.arange(N//2).flip(0)

    def forward(self, x, norm=None):
        B, N = x.shape
        x = x.contiguous().view(-1, N)
        v = self.pad(x)

        #x = x.view(B, N//2, 2)
        #x[:,:,1] = x[:,:,1].flip(1)
        #v = x.contiguous().view(B, N)

        #v = torch.cat([x[:,:,0], x[:,:,1].flip(1)], dim=1)

        Vc = torch.rfft(v, 1, onesided=False)[:,:self.N]

        V = Vc[:, :, 0] * self.W_r - Vc[:, :, 1] * self.W_i

        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(B,N)

        return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.irfft(V, 1, onesided=False)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_3d(dct_3d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)

def rfft(x):
    return torch.rfft(x, 1, onesided=False)

if __name__ == '__main__':

    import timeit

    print("CPU Speed (100 executions):")
    setup = "from __main__ import %s; import torch; x = torch.Tensor(1000,4096); x.normal_(0,1)"
    for d in ["dct1", "dct", "idct", "rfft"]:
        print("  %s: "%d, timeit.timeit("_ = %s(x)"%d, setup=setup%d, number=100))
    setup = "from __main__ import FasterDCT; import torch; x = torch.Tensor(1000,4096);  model = %s; model = model.eval(); x.normal_(0,1)"
    print("  FasterDCT: ", timeit.timeit("_ = model(x)", setup=setup%"FasterDCT(4096)", number=100))   
    setup = "import torch; x = torch.Tensor(1000,4096);  model = %s; model = model.eval(); x.normal_(0,1)"
    print("  Linear: ", timeit.timeit("_ = model(x)", setup=setup%"torch.nn.Linear(4096,4096)", number=100))   
    print("GPU speed (100 executions):")
    setup = "from __main__ import %s; import torch; x = torch.Tensor(1000,4096); x = x.to('cuda'); x.normal_(0,1)"
    for d in ["dct1", "dct", "idct", "rfft"]:
        print("  %s: "%d, timeit.timeit("_ = %s(x)"%d, setup=setup%d, number=100))
    setup = "from __main__ import FasterDCT; import torch; x = torch.Tensor(1000,4096);  model = %s; model = model.to('cuda').eval(); x = x.to('cuda'); x.normal_(0,1)"
    print("  FasterDCT: ", timeit.timeit("_ = model(x)", setup=setup%"FasterDCT(4096)", number=100))  
    setup = "import torch; x = torch.Tensor(1000,4096);  model = %s; model = model.to('cuda').eval(); x = x.to('cuda'); x.normal_(0,1)"
    print("  Linear: ", timeit.timeit("_ = model(x)", setup=setup%"torch.nn.Linear(4096,4096)", number=100))   
