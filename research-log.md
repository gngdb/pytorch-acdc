
4th October 2018
================

Initial implementation of the ACDC layer, and the stacked ACDC Module. Was
unsure what to use to initialise the diagonal matrices, looking at the code
they used either an identity or uniform noise between -1 and 1. Initially,
I tried the latter.

In between layers in the stacked module, they say they use a shuffle
operation. I wasn't sure what kind of shuffle to use, so I just used a
riffle shuffle as they do in ShuffleNet, because it's simple and seems to
work in that setting. This could be a problem, because it's a systematic
transformation, rather than random and might not have as good guarantees
when mixing.

Another problem was that the input and output size of the layer has to be
equal, because there's no way for a diagonal matrix or a DCT to change the
dimensionality of the input. To get around this, I use a Linear layer to
create new dimensions, then concatenate them on to the end. When reducing
dimensionality I use a Linear layer to create a residual of the desired
size from the "extra" dimensions, then add this to the remaining dimensions
after dropping the "extra" ones.

Finally, the FFT is much faster if the input is always a power of 2, so I
added a provision to group the input into powers of 2 before the DCT, so
the DCT is always acting on a power of 2.

To check it can learn something useful, ran it on MNIST, with the default
settings for the PyTorch MNIST example, using the following model
definition:

```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.acdc = StackedACDC(784, 64, 12, groups=8)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c*h*w)
        x = F.relu(self.acdc(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

It maps down to the output dimensionality in the 12 steps. It was able to
classify a bit, but it has a final linear layer, which can count for a lot
in MNIST.

```
Test set: Average loss: 0.1120, Accuracy: 9657/10000 (97%)
```

Then, after reading the right section of the paper I realised that the
initialisation I was using was wrong, so I changed it to the recommendation
of the paper, which is diagonal normal mean=1 sigma=10^-2. Unfortunately,
that trained a little worse:

```
Test set: Average loss: 0.3903, Accuracy: 9518/10000 (95%)
```

Training appears to be relatively slow. It's a few times slower than the
ConvNet default that comes with the MNIST example. Also, there is *no
regularisation* in this model and yet it does not overfit, which is not a
good thing.

Removing the final Linear layer, and deactivating those used when we drop
dimensions, tried to train the resulting model on MNIST. I wouldn't expect
it to work well necessarily, but it might work a bit. It barely worked:

```
Test set: Average loss: 1.6744, Accuracy: 5541/10000 (55%)
```

It doesn't seem like it is a very easy model component to optimise. That's
partly what we might expect from Figure 3 in the paper.

Finally, I thought the execution speed seemed too slow even for the
multiple layers. Added a line to the testing in `layers.py` to look at the
execution time compared to a `Linear` layer. According to Figure 2 in the
paper, the ACDC layer should be significantly faster with an input
dimensionality of 4096. 100 loops with the packing and unpacking disabled:

```
Linear:  0.007770444266498089
ACDC:  0.5301398802548647
```

Then with packing and unpacking enabled:

```
Linear:  0.005231818184256554
ACDC:  0.529364119283855
```

Must be an issue with the DCT implementation.
