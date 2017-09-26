# numpy version of simple forward and backward propagation

import numpy as np

def main():
    # N is batch size
    N = 64
    # D_in is input dimension
    D_in = 1000
    # H is hidden dimension
    H = 100
    # D_out is output dimension
    D_out = 10

    learning_rate = 1e-6

    # Create place_holder for input and output data
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)

    # Create random weights
    w1 = np.random.randn(D_in, H)
    w2 = np.random.randn(H, D_out)

    for t in range(500):
        # forward pass: compute predict y
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)

        # compute and print loss
        loss = np.square(y_pred - y).sum()
        print t, loss

        # backpropagation manually
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)

        w1 -= grad_w1 * learning_rate
        w2 -= grad_w2 * learning_rate

if __name__ == '__main__':
    main()