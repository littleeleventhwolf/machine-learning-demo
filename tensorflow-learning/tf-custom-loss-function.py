import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

# two input node
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
# regression analysis always have one output node
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# define a single layer neural network, in terms of a simple weighted sum
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# define loss function
loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))

# define train optimize step
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# use numpy kit to simulate dataset
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# simulate regression output value by mixing some noise which between -0.05 and 0.05
Y = [[x1 + x2 + rdm.rand()/10.0-0.05] for (x1, x2) in X]

# train the neural network
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	STEPS = 5000
	for i in range(STEPS):
		start = (i * batch_size) % dataset_size
		end = min(start + batch_size, dataset_size)
		sess.run(train_step, feed_dict={x: X[start : end], y_: Y[start : end]})
	print(sess.run(w1))