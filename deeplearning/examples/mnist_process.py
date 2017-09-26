from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from utils import Utils

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# first convolution layer
W_conv1 = Utils.weight_variable([5, 5, 1, 32])
b_conv1 = Utils.bias_variable([32])

h_conv1 = tf.nn.relu(Utils.conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = Utils.max_pool_2x2(h_conv1)

# second convolution layer
# W_conv2 = Utils.weight_variable([5, 5, 32, 64])
# b_conv2 = Utils.bias_variable([64])
#
# h_conv2 = tf.nn.relu(Utils.conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = Utils.max_pool_2x2(h_conv2)

# FIXME: replace a convolution layer with an inception layer
# first inception layer
h_inc1 = tf.nn.relu(Utils.inception(h_pool1, 32))
h_pool2 = Utils.max_pool_2x2(h_inc1)

# first dense connected layer
# W_1 = Utils.weight_variable([7 * 7 * 64, 1024])
W_1 = Utils.weight_variable([7 * 7 * 224, 4096])
b_1 = Utils.bias_variable([4096])

# h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 224])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_1) + b_1)

# dropout mechanism
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# second dense connected layer
W_2 = Utils.weight_variable([4096, 10])
b_2 = Utils.bias_variable([10])

y = tf.nn.relu(tf.matmul(h_fc1_drop, W_2) + b_2)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.InteractiveSession()
sess.run(init)

for step in range(20000):
    batch = mnist.train.next_batch(50)
    if step % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        print("step: {}, accuracy: {}".format(step, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("final accuracy is %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))