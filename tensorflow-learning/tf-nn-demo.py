import tensorflow as tf

# simulate data set
from numpy.random import RandomState

# define batch size
batch_size = 8

# define neural networks parameters
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# define training data placeholder
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# define neural network forward-propagation procedure
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# define loss function and back-propagation algorithm
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# use random numbers to simulate dataset
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# define label, when x1+x2<1, the label is 1, otherwise the label is 0.
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# create session to run process
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	# initialize variables
	sess.run(init_op)
	print("Before training, the parameters are:")
	print(sess.run(w1))
	print(sess.run(w2))
	print("------------------------------------")

	# set epochs
	STEPS = 5000
	for i in range(STEPS):
		# use batch size data to train neural network every time
		start = (i * batch_size) % dataset_size
		end = min(start + batch_size, dataset_size)

		# start train and update parameters
		sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

		# print cross entropy every 1000 steps
		if i % 1000 == 0:
			total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
			print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

	print("------------------------------------")
	# After training
	print("After training, the parameters are:")
	print(sess.run(w1))
	print(sess.run(w2))