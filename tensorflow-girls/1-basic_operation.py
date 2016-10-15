# -*- encoding: utf8 -*-

from __future__ import print_function
from __future__ import division

import tensorflow as tf

# basic operation  + - * /
def basic_operation():
	v1 = tf.Variable(10)
	v2 = tf.Variable(5)
	addv = v1 + v2
	print(addv)
	print(type(addv))
	print(type(v1))

	c1 = tf.constant(10)
	c2 = tf.constant(12)
	addc = c1 + c2
	print(addc)
	print(type(addc))
	print(type(c1))

	# session is a runtime
	sess = tf.Session()

	# Variable -> initialize -> valued Tensor
	tf.initialize_all_variables().run(session=sess)

	print("Variable need to be initialized")
	print('variable add operation: (v1 + v2) = ', addv.eval(session=sess))
	print('constant add operation: (v1 + v2) = ', addc.eval(session=sess))

	# first define operation, second execute operation
	# this model is called "Symbolic Programming"

	# tf.Graph.__init__()
	# Creates a new, empty Graph
	graph = tf.Graph()
	with graph.as_default():
		value1 = tf.constant([1, 2])
		value2 = tf.Variable([3, 4])
		mul = value1 * value2

	with tf.Session(graph=graph) as mySess:
		tf.initialize_all_variables().run()
		print('multiply(value1, value2) = ', mySess.run(mul))
		print('multiply(value1, value2) = ', mul.eval())

	# tensor.eval(session=sess)
	# sess.run(tensor)


if __name__ == '__main__':
	basic_operation()