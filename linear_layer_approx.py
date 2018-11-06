import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import OrderedDict
from functools import reduce
from itertools import product

import numpy as np
from scipy import optimize

from pytorch_acdc.layers import StackedLinearACDC
from layers import ShuffleNetLinear

from obj import PyTorchObjective

from tqdm import tqdm

n_replicates = 10
M, N = 32, 100

def sample_experiment(M, N):
    # whatever this initialises to is our "true" W
    linear = nn.Linear(M,M)
    linear = linear.eval()

    # input X
    X = torch.Tensor(N,M)
    X.uniform_(0.,1.) # fill with uniform
    eps = torch.Tensor(N,M)
    eps.normal_(0., 1e-4)

    # output Y
    with torch.no_grad():
        Y = linear(X) #+ eps

    return X,Y

def run_experiment(X,Y,objective):
    maxiter = 1000
    # try to optimize that function with scipy
    obj = PyTorchObjective(objective)
    # got more reliable results with CG than BFGS
    #xL = optimize.minimize(obj.fun, obj.x0, method='BFGS', jac=obj.jac,
    #        callback=verbose, options={'gtol': 1e-6, 'maxiter':maxiter})
    xL = optimize.minimize(obj.fun, obj.x0, method='CG', jac=obj.jac, options={'maxiter':maxiter})
    return xL.fun

def count_params(module):
    n_params = 0
    for p in module.parameters():
        n_params += reduce(lambda x,y: x*y, p.size())
    return n_params

if __name__ == '__main__':
    # open json storing results
    results_file = 'linear_layer_approx_results.json'
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = OrderedDict()

    # Results formatting: dictionary:
    #   Key: string "<layer_type>_<options>"
    #   Value: [(M, N, no. parameters, final MSE), ...]
    # list contains replicates

    with tqdm(total=6*n_replicates) as pbar:
        # ACDC experiments
        #for i in range(6):
        for i in []:
            n_layers = 2**i
            exp_str = 'ACDC_%i'%n_layers
            pbar.set_description(exp_str)
            if exp_str not in results.keys():
                results[exp_str] = []
            # make module executing the experiment
            class Objective(nn.Module):
                def __init__(self, X, Y):
                    super(Objective, self).__init__()
                    # M layers
                    self.acdc = StackedLinearACDC(M,M,n_layers)
                    #self.acdc = nn.Linear(M,M)
                    self.acdc = self.acdc.train()
                    self.X, self.Y = X, Y
                def forward(self):
                    output = self.acdc(self.X)
                    return F.mse_loss(output, self.Y).mean()

            # run experiment n_replicates times
            for n in range(n_replicates):
                X,Y = sample_experiment(M, N)
                objective = Objective(X, Y)
                mse = run_experiment(X,Y,objective) 
                results[exp_str] += [(M, N, count_params(objective), mse)]
                pbar.update(1)

    plan = list(product([int(2**i) for i in range(2)],
                        [int(2**i) for i in range(1,5)]))
    with tqdm(total=len(plan)*n_replicates) as pbar:
        # ShuffleNet experiments
        for n_layers, n_groups in plan:
        #for i in []:
            exp_str = 'ShuffleNet_%i_%i'%(n_layers, n_groups)
            pbar.set_description(exp_str)
            if exp_str not in results.keys():
                results[exp_str] = []
            # make module executing the experiment
            class Objective(nn.Module):
                def __init__(self, X, Y):
                    super(Objective, self).__init__()
                    # M layers
                    self.linapprox = ShuffleNetLinear(M,M,n_layers,n_groups)
                    self.linapprox = self.linapprox.train()
                    self.X, self.Y = X, Y
                def forward(self):
                    output = self.linapprox(self.X)
                    return F.mse_loss(output, self.Y).mean()

            # run experiment n_replicates times
            for n in range(n_replicates):
                X,Y = sample_experiment(M, N)
                objective = Objective(X, Y)
                mse = run_experiment(X,Y,objective) 
                results[exp_str] += [(M, N, count_params(objective), mse)]
                pbar.update(1)

    # write results to file
    with open(results_file, 'w') as f:
        json.dump(results, f)
