import torch
from scipy import optimize
import torch.nn.functional as F
import math
import numpy as np
from functools import reduce
from collections import OrderedDict

def branin(x, a=1., b=5.1/(4.*math.pi**2), c=5./math.pi, r=6., s=10.,
    t=1./(8*math.pi)):
    # branin function, global min of 0.397887 at
    # x = (-pi, 12.275), (math.pi, 2.275) and (9.42478, 2.475)
    x1, x2 = x[0], x[1]
    return a*(x1 - b*x2 + c*x1 - r)**2 + s*(1.-t)*torch.cos(x1) + s


class PyTorchObjective(object):
    """PyTorch objective function, wrapped to be called by scipy.optimize."""
    def __init__(self, obj_module):
        self.f = obj_module # some pytorch module, that produces a scalar loss
        # make an x0 from the parameters in this module
        parameters = OrderedDict(obj_module.named_parameters())
        self.param_shapes = {n:parameters[n].size() for n in parameters}
        # ravel and concatenate all parameters to make x0
        self.x0 = np.concatenate([parameters[n].data.cpu().numpy().ravel() 
                                   for n in parameters])

    def unpack_parameters(self, x):
        """optimize.minimize will supply 1D array, chop it up for each parameter."""
        i = 0
        named_parameters = OrderedDict()
        for n in self.param_shapes:
            param_len = reduce(lambda x,y: x*y, self.param_shapes[n])
            # slice out a section of this length
            param = x[i:i+param_len]
            # reshape according to this size, and cast to torch
            param = param.reshape(*self.param_shapes[n])
            named_parameters[n] = torch.from_numpy(param)
            # update index
            i += param_len
        return named_parameters

    def pack_grads(self):
        """pack all the gradients from the parameters in the module into a
        numpy array."""
        grads = []
        for p in self.f.parameters():
            grad = p.grad.data.cpu().numpy()
            grads.append(grad.ravel())
        return np.concatenate(grads)

    def is_new(self, x):
        # if this is the first thing we've seen
        if not hasattr(self, 'cached_x'):
            return True
        else:
            # compare x to cached_x to determine if we've been given a new input
            x, self.cached_x = np.array(x), np.array(self.cached_x)
            error = np.abs(x - self.cached_x)
            return error.max() > 1e-4

    def cache(self, x):
        # unpack x and load into module 
        state_dict = self.unpack_parameters(x)
        self.f.load_state_dict(state_dict)
        # store the raw array as well
        self.cached_x = x
        # zero the gradient
        self.f.zero_grad()
        # use it to calculate the objective
        obj = self.f()
        # backprop the objective
        obj.backward()
        self.cached_f = obj.item()
        self.cached_jac = self.pack_grads()

    def fun(self, x):
        if self.is_new(x):
            self.cache(x)
        return self.cached_f

    def jac(self, x):
        if self.is_new(x):
            self.cache(x)
        return self.cached_jac


if __name__ == '__main__':
    x0 = np.random.rand(2)
    
    obj = PyTorchObjective(branin)

    xL = optimize.minimize(obj.fun, x0, method='BFGS', jac=obj.jac)
    print(x0, xL)
