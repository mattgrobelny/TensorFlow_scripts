# Import mnist image data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print "done importing image data"

import tensorflow as tf

################################################################################
# 2d tensor with shape None and and 784 dimensions
# 28x28 Image = 784 numbers

# placeholder, a value that we'll input when we ask TensorFlow to run a computation
x = tf.placeholder(tf.float32, [None, 784])

################################################################################

# Variable is a modifiable tensor that lives in TensorFlow's graph of interacting operations.
# It can be used and even modified by the computation.
# For machine learning applications, one generally has the model parameters be Variables

# Weights
W = tf.Variable(tf.zeros([784, 10]))

# Biasies
b = tf.Variable(tf.zeros([10]))

## ^^ just filled with zeros

# Notice that W has a shape of [784, 10] because we want to multiply the
# 784-dimensional image vectors by it to produce 10-dimensional vectors of
# evidence for the difference classes. b has a shape of [10] so we can
# add it to the output.

################################################################################
# Model

# softmax neural net based on 'x' with weights 'W' biased by 'b'
# softmax =A softmax regression has two steps:
#       - first we add up the evidence of our input being in certain classes
#       - then we convert that evidence into probabilities

y = tf.nn.softmax(tf.matmul(x, W) + b)

################################################################################
# Training

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

################################################################################
# Launch session

sess = tf.Session()
sess.run(init)

################################################################################
# Train the model 1000 times

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Each step of the loop, we get a "batch" of one hundred random data points
# from our training set. We run train_step feeding in the batches data to replace the placeholders.

################################################################################
# Model evaulation

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
