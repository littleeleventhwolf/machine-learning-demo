# # Loading Variables in TensorFlow

import tensorflow as tf
sess = tf.InteractiveSession()

# Create a boolean vector called `spikes` to locate a sudden spike in data.
#
# Since all variables must be initialized, initialize the variable by calling `run()` on its `initializer`
spikes = tf.Variable([False]*8, name='spikes')
saver = tf.train.Saver()

saver.restore(sess, "spikes.cpkt")
print(spikes.eval())

sess.close()