import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function, Variable

import revnet
from revnet import RevBlockFunction

CUDA = torch.cuda.is_available()


class InverseRevBlockFunction(Function):
    @staticmethod
    def _grad(x, dy, in_channels, out_channels, training, stride,
              activations, f_params, f_buffs, g_params, g_buffs,
              no_activation=False, storage_hooks=[]):
        dy1, dy2 = Variable.chunk(dy, 2, dim=1)

        x1, x2 = torch.chunk(x, 2, dim=1)

        x1 = Variable(x1, requires_grad=True).contiguous()
        x2 = Variable(x2, requires_grad=True).contiguous()

        if CUDA:
            x1.cuda()
            x2.cuda()

        x1_ = revnet.possible_downsample(x1, in_channels, out_channels, stride)
        x2_ = revnet.possible_downsample(x2, in_channels, out_channels, stride)

        g_x1 = RevBlockFunction.residual(
            x1,
            out_channels,
            out_channels,
            g_params,
            g_buffs,
            training=training
        )

        y2_ = x2_ - g_x1

        f_y2 = RevBlockFunction.residual(
            y2_,
            in_channels,
            out_channels,
            f_params,
            f_buffs,
            training=training,
            stride=stride,
            no_activation=no_activation
        )

        y1_ = x1_ - f_y2

        dd1 = torch.autograd.grad(y1_, (y2_,) + tuple(f_params), dy1,
                                  retain_graph=True)
        dy1_y2 = dd1[0]
        dfw = dd1[1:]
        dy2_plus = -dy1_y2 + dy2
        dd2 = torch.autograd.grad(y2_, (x1, x2) + tuple(g_params), dy2_plus,
                                  retain_graph=True)
        dgw = dd2[2:]

        dx2 = dd2[1]
        dx1 = dd2[0]
        dx1 += torch.autograd.grad(x1_, x1, dy1, retain_graph=True)[0]

        for hook in storage_hooks:
            x = hook(x)

        activations.append(x)

        y1_.detach_()
        y2_.detach_()
        del y1_, y2_
        dx = torch.cat((dx1, dx2), 1)

        return dx, dfw, dgw

    @staticmethod
    def forward(ctx, x, in_channels, out_channels, training, stride,
                no_activation, activations, storage_hooks, *args):
        """Compute forward pass including boilerplate code.

        This should not be called directly, use the apply method of this class.

        Args:
            ctx (Context):                  Context object, see PyTorch docs
            x (Tensor):                     4D input tensor
            in_channels (int):              Number of channels on input
            out_channels (int):             Number of channels on output
            training (bool):                Whethere we are training right now
            stride (int):                   Stride to use for convolutions
            no_activation (bool):           Whether to compute an initial
                                            activation in the residual function
            activations (List):             Activation stack
            storage_hooks (List[Function]): Functions to apply to activations
                                            before storing them
            *args:                          Should contain all the Parameters
                                            of the module
        """

        if not no_activation:
            f_params = [Variable(x) for x in args[:8]]
            g_params = [Variable(x) for x in args[8:16]]
            f_buffs = args[16:20]
            g_buffs = args[20:]
        else:
            f_params = [Variable(x) for x in args[:6]]
            g_params = [Variable(x) for x in args[6:14]]
            f_buffs = args[14:16]
            g_buffs = args[16:]

        if CUDA:
            for var in f_params:
                var.cuda()
            for var in g_params:
                var.cuda()

        # if stride > 1 information is lost and we need to save the input
        if stride > 1 or no_activation:
            activations.append(x)
            ctx.load_input = True
        else:
            ctx.load_input = False

        ctx.save_for_backward(*[x.data for x in f_params],
                              *[x.data for x in g_params])
        ctx.f_buffs = f_buffs
        ctx.g_buffs = g_buffs
        ctx.stride = stride
        ctx.training = training
        ctx.no_activation = no_activation
        ctx.storage_hooks = storage_hooks
        ctx.activations = activations
        ctx.in_channels = in_channels
        ctx.out_channels = out_channels

        y = RevBlockFunction._backward(
            x,
            in_channels,
            out_channels,
            f_params, f_buffs,
            g_params, g_buffs,
            training,
            no_activation=no_activation
        )

        # y_ = y.clone()
        # for hook in storage_hooks:
        #     y_ = hook(y_)

        # activations.append(y_)

        return y

    @staticmethod
    def backward(ctx, grad_out):
        saved_variables = list(ctx.saved_variables)
        if not ctx.no_activation:
            f_params = saved_variables[:8]
            g_params = saved_variables[8:16]
        else:
            f_params = saved_variables[:6]
            g_params = saved_variables[6:14]

        in_channels = ctx.in_channels
        out_channels = ctx.out_channels

        # Load or reconstruct input
        if ctx.load_input:
            ctx.activations.pop()
            x = ctx.activations.pop()
        else:
            output = ctx.activations.pop()
            # print(output.size(), in_channels)
            x = RevBlockFunction._forward(
                output,
                in_channels,
                out_channels,
                ctx.training,
                ctx.stride,
                f_params, ctx.f_buffs,
                g_params, ctx.g_buffs,
                ctx.no_activation
            )

        dx, dfw, dgw = InverseRevBlockFunction._grad(
            x.data,
            grad_out,
            in_channels,
            out_channels,
            ctx.training,
            ctx.stride,
            ctx.activations,
            f_params, ctx.f_buffs,
            g_params, ctx.g_buffs,
            no_activation=ctx.no_activation,
            storage_hooks=ctx.storage_hooks
        )

        num_buffs = 2 if ctx.no_activation else 4

        return ((dx, None, None, None, None, None, None, None) + tuple(dfw) +
                tuple(dgw) + tuple([None]*num_buffs) + tuple([None]*4))


class BidirectionalBlock(revnet.RevBlock):
    def __init__(self, in_channels, out_channels, activations,
                 no_activation=False, storage_hooks=[], inverse_hooks=[]):

        super(BidirectionalBlock, self).__init__(
            in_channels,
            out_channels,
            activations,
            stride=1,
            no_activation=False,
            storage_hooks=storage_hooks
        )

        self.inverse_hooks = inverse_hooks

    def inverse_forward(self, x):
        return InverseRevBlockFunction.apply(
            x,
            self.in_channels,
            self.out_channels,
            self.training,
            self.stride,
            self.no_activation,
            self.activations,
            self.inverse_hooks,
            *self._parameters.values(),
            *self._buffers.values(),
        )
