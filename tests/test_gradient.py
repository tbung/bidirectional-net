import torch

from binet import BidirectionalBlock

def test_grad():
    activations = []
    block = BidirectionalBlock(4, 4, activations)
    block.cuda()
    block.double()

    def forward(x):
        out = block.inverse_forward(x)
        activations.append(out.data)
        return out

    x = torch.autograd.Variable(torch.randn(4,4,4,4).double().cuda(), requires_grad=True)
    c = torch.nn.Conv2d(4, 4, 3)
    c.cuda()
    c.double()
    assert torch.autograd.gradcheck(
        forward,
        (x,)
    )
