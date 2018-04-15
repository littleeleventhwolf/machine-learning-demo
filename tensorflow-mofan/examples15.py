from __future__ import print_function
import tensorflow as tf

tf.set_random_seed(1) # reproducible

with tf.name_scope("a_name_scope"):
	initializer = tf.constant_initializer(value=1)
	var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
	var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
	var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)
	var22 = tf.Variable(name='var2', initial_value=[2.2], dtype=tf.float32)

with tf.variable_scope("a_variable_scope") as scope:
	initializer = tf.constant_initializer(value=3)
	var3 = tf.get_variable(name='var3', shape=[1], dtype=tf.float32, initializer=initializer)
	scope.reuse_variables()
	var3_reuse = tf.get_variable(name='var3')
	var4 = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)
	var4_reuse = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	print("==========name_scope==========")
	# name_scope
	print(var1.name)
	print(sess.run(var1))
	print(var2.name)
	print(sess.run(var2))
	print(var21.name)
	print(sess.run(var21))
	print(var22.name)
	print(sess.run(var22))

	print("==========variable_scope==========")
	# variable_scope
	print(var3.name)
	print(sess.run(var3))
	print(var3_reuse.name)
	print(sess.run(var3_reuse))
	print(var4.name)
	print(sess.run(var4))
	print(var4_reuse.name)
	print(sess.run(var4_reuse))