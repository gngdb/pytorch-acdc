from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from pytorch_acdc.layers import StackedLinearACDC

from obj import PyTorchObjective
from scipy import optimize
from tqdm import tqdm

from layers import ShuffleNetLinear, StackedAFDF, GroupedStackedACDC, GroupedStackedAFDF

class Net(nn.Module):
    def __init__(self, original):
        super(Net, self).__init__()
        #self.model = StackedLinearACDC(784, 784, 32, original=original)
        #self.model = nn.Linear(784,10)
        self.model = ShuffleNetLinear(784, 784, 4, 784//4)

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c*h*w)
        x = self.model(x)
        return F.log_softmax(x[:,:10], dim=1)

def train(args, device, train_loader):
    
    # make an objective as a class
    class Objective(nn.Module):
        def __init__(self): 
            super(Objective, self).__init__()
            self.model = Net(original=False)
            self.model.train()

        def forward(self):
            data, target = train_loader[0][0], train_loader[0][1]
            data, target = data.to(device), target.to(device)
            self.model.to(device)
            output = self.model(data)
            return F.nll_loss(output, target)

    objective = Objective()
    obj = PyTorchObjective(objective)
    maxiter = 1000
    with tqdm(total=maxiter) as pbar:
        def verbose(*args):
            pbar.update(1)
        xL = optimize.minimize(obj.fun, obj.x0, method='CG', jac=obj.jac,
                callback=verbose, options={'maxiter':maxiter})
        #xL = optimize.minimize(obj.fun, obj.x0, method='BFGS', jac=obj.jac,
        #        callback=verbose, options={'maxiter':maxiter})
    # make sure the final value is loaded to the model
    _ = obj.fun(xL.x)
    return obj.f.model

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--original', action='store_true', default=False,
                        help='uses original ACDC parameterisation')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    torch.manual_seed(1)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)
    train_loader = [next(iter(train_loader))] # train on only one minibatch
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True)


    model = train(args, device, train_loader)
    test(args, model, device, test_loader)


if __name__ == '__main__':
    main()
