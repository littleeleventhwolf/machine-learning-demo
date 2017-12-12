import tensorflow as tf

# define a variable, initialize with 0
v1 = tf.Variable(0, dtype=tf.float32)
# simulate iteration epochs to control decay dunamically
step = tf.Variable(0, trainable=False)

# define an object with decay 0.99 and control-decay variable step
ema = tf.train.ExponentialMovingAverage(0.99, step)
# define an update-moving-average operation. we need a list in this function.
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
	# initialize all variables
	init_op = tf.global_variables_initializer()
	sess.run(init_op)

	# get the moving-average-value by function ema.average(v1).
	# we initialize with 0, so the result is [0.0, 0.0]
	print(sess.run([v1, ema.average(v1)]))		# [0.0, 0.0]

	# update v1 with value 5
	sess.run(tf.assign(v1, 5))
	# update moving-average-value.
	# decay = min{0.99, (1+step)/(10+step)=0.1}=0.1,
	# so the moving-average-value is 0.1*0+0.9*5=4.5
	sess.run(maintain_averages_op)
	print(sess.run([v1, ema.average(v1)]))		# [5.0, 4.5]

	# update step with value 10000
	sess.run(tf.assign(step, 10000))
	# update v1 with value 10
	sess.run(tf.assign(v1, 10))
	# update moving-average-value.
	# decay = min{0.99, (1+step)/(10+step) is approximately equal to 0.999}=0.99
	# so the moving-average-value is 0.99*4.5+0.01*10=4.555
	sess.run(maintain_averages_op)
	print(sess.run([v1, ema.average(v1)]))		# [10.0, 4.5549998]

	# one more time.
	# so the moving-average-value is 0.99*4.555+0.01*10=4.6095
	sess.run(maintain_averages_op)
	print(sess.run([v1, ema.average(v1)]))		# [10.0, 4.6094499]