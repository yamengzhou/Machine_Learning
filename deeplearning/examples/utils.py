import tensorflow as tf
import numpy as np

def max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'):
    return tf.nn.max_pool(x, ksize=ksize,
                          strides=strides, padding=padding)

class Utils:
    def __init__(self):
        pass

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
    @staticmethod
    def max_pool_3x3(x):
        return max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 1, 1, 1],
                        padding='SAME')

    @staticmethod
    def inception(x, channel):
        # inception module item 1x1 conv
        w_1x1 = Utils.weight_variable([1, 1, channel, channel * 2])
        x_1x1 = Utils.conv2d(x, w_1x1)

        # inception module item 3x3 conv
        w_3x3 = Utils.weight_variable([3, 3, channel, channel * 2])
        x_3x3 = Utils.conv2d(x, w_3x3)

        # inception module item 5x5 conv
        w_5x5 = Utils.weight_variable([5, 5, channel, channel * 2])
        x_5x5 = Utils.conv2d(x, w_5x5)

        # inception module item 3x3 max pooling
        x_maxpool_3x3 = Utils.max_pool_3x3(x)

        # concate the result
        res = tf.concat(3, [x_1x1, x_3x3, x_5x5, x_maxpool_3x3]) # 64 + 64 + 64 + 32

        return res
