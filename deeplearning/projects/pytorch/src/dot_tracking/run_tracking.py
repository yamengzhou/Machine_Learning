import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from projects.pytorch.src.dot_tracking.simple_tracking_cnn import SimpleTrackingCNN


def parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-df', '--data_file', required=True,
                           help='data file for training and validation')
    argparser.add_argument('-tr', '--training_ratio', type=float, default=0.8,
                           help='the portion rate of data used for training')
    argparser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                           help='learning rate for training')
    argparser.add_argument('-o', '--output', help='output to save result')
    return argparser.parse_args()


def main():
    args = parse_arguments()
    print args

    # data preparation
    input_data = np.array(np.load(args.data_file))
    ratio = args.training_ratio
    split_pt = int(input_data.shape[0] * ratio)
    training = input_data[: split_pt]
    testing = input_data[split_pt:]

    # initialize variables and model
    model = SimpleTrackingCNN()

    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    for i in range(1000):
        # training process
        for t in training:
            t_in = np.expand_dims(np.expand_dims(t[0], 0), 0)
            t_out = np.expand_dims(t[1], 0)
            x = Variable(torch.from_numpy(t_in).type(torch.FloatTensor), requires_grad=False)
            y = Variable(torch.from_numpy(np.array(t_out)).type(torch.FloatTensor), requires_grad=False)
            # forward step
            y_pred = model(x)

            # calculate the loss
            loss = criterion(y_pred, y)

            # zero all gradient and backward prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test the MSE on the testing set
        errors = 0.0
        for v in testing:
            t_in = np.expand_dims(np.expand_dims(v[0], 0), 0)
            t_out = np.expand_dims(v[1], 0)
            x = Variable(torch.from_numpy(t_in).type(torch.FloatTensor), requires_grad=False)
            y = Variable(torch.from_numpy(np.array(t_out)).type(torch.FloatTensor), requires_grad=False)
            # forward step
            y_pred = model(x)

            loss = criterion(y_pred, y)
            errors += loss.data[0]
        avg_diff = errors / testing.shape[0]
        print "epoch {}, error: {}".format(i, avg_diff)


if __name__ == '__main__':
    main()
