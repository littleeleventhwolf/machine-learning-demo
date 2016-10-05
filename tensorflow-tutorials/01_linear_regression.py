#!/usr/bin/env python

from __future__ import print_function

import tensorflow as tf
import numpy as np

trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 # create a y value which is approximately linear but with some random noise

X = tf.placeholder(tf.float32) # create symbolic variables
Y = tf.placeholder(tf.float32)

def model(X, w):
	return tf.mul(X, w) # linear regression is just X*w, so this model line is pretty simple

w = tf.Variable(0.0, name="weights") # create a shared variable for the weight matrix
y_model = model(X, w)

cost = tf.square(Y - y_model) # use square error for cost function

# construct an optimizer to minimize cost and fit line to mydata
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# launch the graph in a session
with tf.Session() as sess:
	# you need to initialize variables (in this case just variable w)
	init = tf.initialize_all_variables()
	sess.run(init)

	# train
	for i in range(100):
		for (x, y) in zip(trX, trY):
			sess.run(train_op, feed_dict={X: x, Y: y})

	# print weight
	print(sess.run(w)) # it should be something around 2