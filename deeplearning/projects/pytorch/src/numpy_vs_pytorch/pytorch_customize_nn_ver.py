# this script is a sample code for building up customized nn by using pytorch nn module
import torch
from torch.autograd import Variable


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


def main():
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
    x = Variable(torch.randn(N, D_in))
    y = Variable(torch.randn(N, D_out), requires_grad=False)

    # Construct our model by instantiating the class defined above
    model = TwoLayerNet(D_in, H, D_out)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(500):
        # forward pass
        y_pred = model(x)

        # compute loss
        loss = criterion(y_pred, y)
        print t, loss.data[0]

        # Zero gradients, perform a backward pass, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()