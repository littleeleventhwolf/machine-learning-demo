# -*- encoding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import tensorflow as tf

def use_placeholder():
	graph = tf.Graph()
	with graph.as_default():
		value1 = tf.placeholder(dtype=tf.float64)
		value2 = tf.Variable([3, 4], dtype=tf.float64)
		mul = value1 * value2

	with tf.Session(graph=graph) as mySess:
		tf.initialize_all_variables().run()
		# imagine the data load from the remote(file or network)
		value = load_from_remote()
		for partialValue in load_partial(value, 2):
			holderValue = {value1: partialValue}
			# runResult = mySess.run(mul, feed_dict=holderValue)
			evalResult = mul.eval(feed_dict=holderValue)
			print('Multiply(value1, value2) = ', evalResult)


def load_from_remote():
	return [-x for x in range(1000)]

# customer iterator
# yield, generator function
def load_partial(value, step):
	index = 0
	while index < len(value):
		yield value[index:index+step]
		index += step
	return

if __name__ == "__main__":
	use_placeholder()