
While this is a fork of [torch-dct](https://github.com/zh217/torch-dct),
this repository was used to run a small investigation into whether [ACDC
layers][acdc] can be used in convolutional layers.

Initially, this was just because we saw that someone had implemented DCTs
in PyTorch and thought that would make it relatively easy to try it out.

Unfortunately, the answer seems to be that *no*, an ACDC layer doesn't make
a good parameterisation for a convolutional layer. The network won't
optimise as readily as it does with an unconstrained weight matrix.

Is there anything useful here?
==============================

There's some benchmarking of the speed of different DCT implementations in
PyTorch. The fastest implementation on GPU was simply using the DCT matrix
in a Linear layer. This was because the tensor manipulations in order to
use Makhoul's method involving an FFT add much more time than the FFT
itself. 

The ACDC implementations appear to be correct, stable and in line with what
the paper describes, but we've only done a basic replication of section
6.1 in the script [[./linear_layer_approx.py]].

Full details of the investigation can be found in the [[research_log.md]].

[acdc]: https://arxiv.org/abs/1511.05946

