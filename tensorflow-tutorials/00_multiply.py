#!/usr/bin/env python

from __future__ import print_function

import tensorflow as tf

a = tf.placeholder(tf.float32) # create a symbolic variable 'a'
b = tf.placeholder(tf.float32) # create a symbolic variable 'b'

y = tf.mul(a, b) # multiply the symbolic variables

with tf.Session() as sess: # create a session to evaluate the symbolic expressions
	#print(sess.run([y], feed_dict={a: [7.], b: [2.]}))
	print(sess.run(y, feed_dict={a: 7., b: 2.})) # eval expressions with parameters for a and b