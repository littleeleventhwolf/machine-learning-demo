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

	# session is a runtime
	sess = tf.Session()

	# Variable -> initialize -> valued Tensor
	tf.initialize_all_variables().run(session=sess)

	print("Variable need to be initialized")
	print('add operation: (v1 + v1) = ', addv.eval(session=sess))


if __name__ == '__main__':
	basic_operation()