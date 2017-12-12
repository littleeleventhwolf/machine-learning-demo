import tensorflow as tf

# get weights of single layer, then add L2 regularizer to collection 'losses'
def get_weight(shape, lambda):
	# generate a variable
	var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
	# add_to_collection --- add L2 regularizer loss into collection 'losses'
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda)(var))
	# return this variable
	return var

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
batch_size = 8
# define the number of nodes in each layer
layer_dimension = [2, 10, 10, 10, 1]
# the number of layers
n_layers = len(layer_dimension)

# the front layer (first step is input layer)
cur_layer = x
# the number of nodes in current layer
in_dimension = layer_dimension[0]

# construct 5-layers neural network in a loop
for i in range(1, n_layers):
	# layer_dimension[i] --- the number of node in next layer
	out_dimension = layer_dimension[i]
	# generate weight of current layer, and add L2 regularizer loss into collection
	weight = get_weight([in_dimension, out_dimension], 0.001)
	bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
	# ReLU activation function
	cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
	# update the number of node in current layer before enter next loop
	in_dimension = layer_dimension[i]

# because we add L2 regularizer loss into collection while define the forward propagation process,
# we only calculate the loss of training model.
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
# add mean square error loss into collection
tf.add_to_collection('losses', mse_loss)

# get_collection --- return a list.
# in this example, it contains loss of each layer.
# add all the loss to generate the final loss function.
loss = tf.add_n(tf.get_collection('losses'))