# Looking at the properties of he implied weight matrices when using a stacked
# ACDC layer
import math
import torch
from layers import StackedACDC

if __name__ == '__main__':
    def statistics(N):
        x = torch.eye(N).float()
        W = []
        for i in range(10):
            acdc = StackedACDC(N,N,16)
            W.append(acdc(x).view(1,-1))
        W = torch.cat(W, 0)

        print("  Glorot stdv is", 1./math.sqrt(N))
        print("  Effective W stdv", torch.sqrt(W.var(1)).mean().item())
        print("  Effective W mean", W.mean(1).mean().item())

    for N in [2**i for i in range(2,9)]:
        print("Input dim %i"%N)
        statistics(N)
