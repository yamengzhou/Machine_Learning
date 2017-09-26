# simple network by using pytorch
import torch
from torch.autograd import Variable

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

    learning_rate = 1e-3

    # Create random Tensors to hold input and outputs, and wrap thm in Variables.
    # Setting requires_grad=False indicates that we do not need to compute gradients
    # with respect to these Variables during the backward pass
    x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
    y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

    # Use the nn package to define our model as a sequence of layers. nn.Sequential
    # is a module which contains other Modules, and applies them in sequence to produce
    # its output. Each Linear Module computes output from input using a linear function,
    # and holds internal Variables for its weight and bias
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    )

    # The nn package also contains definition of popular loss functions; in this
    # case we will use Mean Square Error (MSE) as our loss function
    loss_fn = torch.nn.MSELoss(size_average=False)

    for t in range(500):
        # forward pass
        y_pred = model(x)

        # compute loss
        loss = loss_fn(y_pred, y)
        print t, loss.data[0]

        # Zero the gradients before running the backward pass.
        model.zero_grad()

        # backward pass
        loss.backward()

        # update weights
        for param in model.parameters():
            param.data -= learning_rate * param.grad.data


if __name__ == '__main__':
    main()