import torch

'''pytorch version of gradient reverse function'''

class GradientReverseFunction(torch.autograd.Function):

    '''x = GradientReverseFunction.apply(x)
    This function is used before the discriminator'''

    @staticmethod
    def forward(ctx, input, lambd):
        output = input * 1.0
        ctx.lambd = lambd
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None
