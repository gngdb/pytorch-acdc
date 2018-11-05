import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from scipy import optimize

from pytorch_acdc.layers import StackedLinearACDC

from obj import PyTorchObjective

from tqdm import tqdm

if __name__ == '__main__':
    # whatever this initialises to is our "true" W
    linear = nn.Linear(32,32)
    linear = linear.eval()

    # input X
    N = 10000
    X = torch.Tensor(N,32)
    X.uniform_(0.,1.) # fill with uniform
    eps = torch.Tensor(N,32)
    eps.normal_(0., 1e-4)

    # output Y
    with torch.no_grad():
        Y = linear(X) #+ eps

    # make module executing the experiment
    class Objective(nn.Module):
        def __init__(self):
            super(Objective, self).__init__()
            # 32 layers
            #self.acdc = StackedLinearACDC(32,32,32)
            self.acdc = nn.Linear(32,32)
            self.acdc = self.acdc.train()
            self.X, self.Y = X, Y

        def forward(self):
            output = self.acdc(self.X)
            return F.mse_loss(output, self.Y).mean()

    objective = Objective()

    def verbose(xk):
        print(xk)
        import time
        time.sleep(0.5)
        return False

    # try to optimize that function with scipy
    obj = PyTorchObjective(objective)
    xL = optimize.minimize(obj.fun, obj.x0, method='BFGS', jac=obj.jac, options={'gtol': 1e-6, 'disp': True})
    #xL = optimize.minimize(obj.fun, obj.x0, method='CG', jac=obj.jac)# , options={'gtol': 1e-2})
    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression()
    linreg.fit(X,Y)
    Y_hat = linreg.predict(X)
    print("By least squares: ", np.square(Y_hat - Y).mean())
