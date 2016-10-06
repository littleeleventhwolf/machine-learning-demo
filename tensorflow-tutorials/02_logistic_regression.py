#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

# read MNIST data set
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# create symbolic variables
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# create variables: weights and biases
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# define model
y = tf.nn.softmax(tf.matmul(X, W) + b)

# cross entropy
cross_entropy = -tf.reduce_sum(Y * tf.log(y))

# train step
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# init step
init = tf.initialize_all_variables()

with tf.Session() as sess:
	# run the init op
	sess.run(init)
	# then train
	for i in range(1000):
		batch_trX, batch_trY = mnist.train.next_batch(128)
		sess.run(train_step, feed_dict={X: batch_trX, Y: batch_trY})

	# test and evaluate our model
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})