import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_dct.layers import StackedACDC

from tqdm import tqdm

if __name__ == '__main__':
    # whatever this initialises to is our "true" W
    linear = nn.Linear(32,32)
    linear = linear.eval()

    # input X
    X = torch.Tensor(10000,32)
    X.uniform_(0.,1.) # fill with uniform
    eps = torch.Tensor(10000,32)
    eps.normal_(0., 1e-4)

    # output Y
    with torch.no_grad():
        Y = linear(X) + eps

    # 32 layers should be sufficient
    acdc = StackedACDC(32,32,32)
    acdc = acdc.train()

    # gradient optimiser
    optimizer = optim.SGD(acdc.parameters(), lr=0.1, momentum=0.9)

    # train this for a few thousand iterations
    N = 1000
    with tqdm(total=N) as pbar:
        for i in range(N):
            optimizer.zero_grad()
            output = acdc(X)
            loss = F.mse_loss(output, Y)
            loss.backward()
            optimizer.step()
            pbar.set_description("Loss: %.4f"%loss.item())
            pbar.update(1)
