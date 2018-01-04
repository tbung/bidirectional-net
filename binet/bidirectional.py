import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function, Variable

import revnet

CUDA = torch.cuda.is_available()


class InverseRevBlockFunction(Function):
    @staticmethod
    def _grad():
        pass

    @staticmethod
    def forward(ctx, x):
        pass

    @staticmethod
    def backward(ctx, grad_out):
        pass


class BidirectionalBlock(revnet.RevBlock):
    def __init__(self, in_channels, out_channels, activations,
                 no_activation=False):

        super(BidirectionalBlock, self).__init__(
            in_channels,
            out_channels,
            activations,
            stride=1,
            no_activation=False
        )

    def inverse_forward(self, x):
        pass
