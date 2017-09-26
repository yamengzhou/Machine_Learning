# simple network by using pytorch
import torch
from torch.autograd import Variable

class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Function by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    def forward(self, input):
        """
        In the forward pass we received a Tensor containing the input and return a
        Tensor containing the output. you can cache arbitrary Tensors for use in
        the backward pass using the save_for_backward method
        """
        self.save_for_backward(input)
        return input.clamp(min=0)

    def backward(self, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss with
        respect to the output, and we need to compute the gradient of the loss with respect
        to the input
        """
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


def main():
    dtype = torch.FloatTensor

    # N is batch size
    N = 64
    # D_in is input dimension
    D_in = 1000
    # H is hidden dimension
    H = 100
    # D_out is output dimension
    D_out = 10

    learning_rate = 1e-6

    # Create random Tensors to hold input and outputs, and wrap thm in Variables.
    # Setting requires_grad=False indicates that we do not need to compute gradients
    # with respect to these Variables during the backward pass
    x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
    y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

    # Create random Tensors for weights, and wrap thm in Variables.
    # Setting requires_grad=False indicates that we do not need to compute gradients
    # with respect to these Variables during the backward pass
    w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
    w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

    for t in range(500):
        relu = MyReLU()

        # forward pass
        y_pred = relu(x.mm(w1)).mm(w2)

        # compute loss
        loss = (y_pred - y).pow(2).sum()
        print t, loss.data[0]

        # backpropagation
        loss.backward()

        # update weights
        w1.data -= w1.grad.data * learning_rate
        w2.data -= w2.grad.data * learning_rate

        # manually set
        w1.grad.data.zero_()
        w2.grad.data.zero_()

if __name__ == '__main__':
    main()