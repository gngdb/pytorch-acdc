
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

Added speed test to the `_dct.py` file:

```
dct1:  0.01097189262509346
dct:  0.2719538463279605
idct:  0.1428453316912055
Linear:  0.00483915489166975
```

So, on the GPU it's slower than a Linear layer every time. The forward DCT
being the worst. Looking at the implementation there's a lot of wacky stuff
going on around the calls to the actual FFT function, which could be
causing the problems, especially given that stuff's not present in `dct1`,
which is much faster (though still slower than Linear).

On CPU, the DCTs are faster than a Linear layer, so it could be that the
CUDA FFT is not optimized well (unlikely), or that something else is
slowing it down:

```
CPU Speed (100 executions):
  dct1:  1.1211627703160048
  dct:  2.0362922428175807
  idct:  3.8709053453058004
  Linear:  4.468587472103536
GPU speed (100 executions):
  dct1:  0.010546403005719185
  dct:  0.2743651457130909
  idct:  0.14596352353692055
  Linear:  0.004489514045417309
```

To check if this is the same in tensorflow, ran the DCT that comes with
tensorflow (running on GPU):

```
DCT:  0.6155939232558012
Linear:  0.9840689264237881
```

This one is faster than the linear layer, but the linear layer is so much
slower than the linear layer in PyTorch. I implemented it as just a matrix
multiply with a random normal weight matrix.

5th October 2018
================

Added a test of the linear problem from Section 6.1 of the paper. This
implementation isn't able to get the loss below 1.0 after 1000 iterations,
and the iterations themselves are extremely slow. 

Trying to figure out why the DCT implementation is so much slower than a
linear layer ended up messing around with the code for the DCT. The FFT on
its own is actually faster than the linear layer (just) on the GPU and CPU,
which is nice.

Thought that it was the generation of `torch.arange` in every pass that
would be increasing the time, so decided to cached that part, as we only
have to know the size of the incoming X. Made a module, and called it
`FasterDCT` to do this. Unfortunately, that didn't affect the speed at all.
Found that the major factor affecting the speed was the `.flip(1)` on the
array before the FFT.

It turned out it was actually faster to use the form of the DCT by DFT
where we just pad the input with zeros using the (relatively efficient) pad
operations built into pytorch. Doing this and the result was slower on the
CPU, but about 10 times faster on the GPU.

Unfortunately, this is still slower than the Linear layer, so finally wrote
an implementation that *is a linear layer*. It's just a linear layer where
the weights are the DCT matrix. This seems to work fine. On the GPU, it's
actually the same speed as the `rfft`:

```
CPU Speed (100 executions):
  dct1:  1.0706987995654345
  dct:  1.982983972877264
  idct:  3.213463810272515
  rfft:  0.631943185813725
  FasterDCT:  3.719810419715941
  Linear:  4.693692773580551
GPU speed (100 executions):
  dct1:  0.010924047790467739
  dct:  0.27390200551599264
  idct:  0.14248866774141788
  rfft:  0.004178566858172417
  FasterDCT:  0.030473709106445312
  Linear:  0.006033767946064472
  Dense DCT:  0.004221102222800255
```
