# simple network by using pytorch
import torch

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

    x = torch.randn(N, D_in).type(dtype)
    y = torch.randn(N, D_out).type(dtype)

    w1 = torch.randn(D_in, H).type(dtype)
    w2 = torch.randn(H, D_out).type(dtype)

    for t in range(500):
        # forward pass
        h = x.mm(w1)
        h_relu = h.clamp(min=0)
        y_pred = h_relu.mm(w2)

        # compute loss
        loss = (y_pred - y).pow(2).sum()
        print t, loss

        # backpropagation
        grad_y_pred = 2.0 * (y_pred - y)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = x.t().mm(grad_h)

        # update weights
        w1 -= grad_w1 * learning_rate
        w2 -= grad_w2 * learning_rate

if __name__ == '__main__':
    main()