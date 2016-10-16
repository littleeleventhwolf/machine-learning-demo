from __future__ import division
from __future__ import print_function
from six.moves import range

import numpy as np
from sklearn.metrics import confusion_matrix
import csv
import tensorflow as tf

import preprocess_data as load

train_samples, train_labels = load._train_samples, load._train_labels
test_samples, test_labels = load._test_samples, load._test_labels

print('Training set', train_samples.shape, train_labels.shape)
print('    Test set', test_samples.shape, test_labels.shape)

image_size = load.image_size
num_labels = load.num_labels
num_channels = load.num_channels

def get_chunk(samples, labels, chunkSize):
	"""
	Iterator / Generator: get a batch of data
	"""
	if len(samples) != len(labels):
		raise Exception('Length of samples and labels must equal.')
	stepStart = 0 # initial step
	i = 0
	while stepStart < len(samples):
		stepEnd = stepStart + chunkSize
		if stepEnd < len(samples):
			yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
			i += 1
		stepStart = stepEnd

class Network():
	def __init__(self, num_hidden, batch_size):
		"""
		@num_hidden: number of hidden
		@batch_size: size of per batch
		"""
		self.batch_size = batch_size
		self.test_batch_size = 500

		# Hyper Parameters
		self.num_hidden = num_hidden
		
		# Graph Related
		self.graph = tf.Graph()
		self.tf_train_samples = None
		self.tf_train_labels = None
		self.tf_test_samples = None
		self.tf_test_labels = None
		self.tf_test_prediction = None

		# Statistics
		self.merged = None

		self.define_graph()
		self.session = tf.Session(graph=self.graph)
		self.writer = tf.train.SummaryWriter('./board', self.graph)

	def define_graph(self):
		"""
		define the graph
		"""
		with self.graph.as_default():
			# define var in graph
			with tf.name_scope('Input'):
				self.tf_train_samples = tf.placeholder(
					tf.float32, shape=(self.batch_size, image_size, image_size, num_channels), name='tf_train_samples'
					)
				self.tf_train_labels = tf.placeholder(
					tf.float32, shape=(self.batch_size, num_labels), name='tf_train_labels'
					)
				self.tf_test_samples = tf.placeholder(
					tf.float32, shape=(self.test_batch_size, image_size, image_size, num_channels), name='tf_test_samples'
					)

			# fully connected layer 1
			with tf.name_scope('FC1'):
				fc1_weights = tf.Variable(
					tf.truncated_normal([image_size * image_size, self.num_hidden], stddev=0.1), name='fc1_weights'
					)
				fc1_biases = tf.Variable(tf.constant(0.1, shape=[self.num_hidden]), name='fc1_biases')
				tf.histogram_summary('FC1_WEIGHTS', fc1_weights)
				tf.histogram_summary('FC1_BIASES', fc1_biases)

			# fully connected layer 2
			with tf.name_scope('FC2'):
				fc2_weights = tf.Variable(
					tf.truncated_normal([self.num_hidden, num_labels], stddev=0.1), name='fc2_weights'
					)
				fc2_biases = tf.Variable(tf.constant(0.1, shape=[num_labels]), name='fc2_biases')
				tf.histogram_summary('FC2_WEIGHTS', fc2_weights)
				tf.histogram_summary('FC2_BIASES', fc2_biases)

			# define operation in graph
			def model(data):
				# fully connected layer 1
				shape = data.get_shape().as_list()
				reshape = tf.reshape(data, [shape[0], shape[1] * shape[2] * shape[3]])
				with tf.name_scope('FC1_MODEL'):
					fc1_model = tf.matmul(reshape, fc1_weights) + fc1_biases
					hidden = tf.nn.relu(fc1_model)

				# fully connected layer 2
				with tf.name_scope('FC2_MODEL'):
					return tf.matmul(hidden, fc2_weights) + fc2_biases

			# Training computation
			logits = model(self.tf_train_samples)
			with tf.name_scope('LOSS'):
				self.loss = tf.reduce_mean(
					tf.nn.softmax_cross_entropy_with_logits(logits, self.tf_train_labels)
					)
				tf.scalar_summary('Loss', self.loss)

			# Optimizer
			with tf.name_scope('OPTIMIZER'):
				self.optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(self.loss)

			# Predictions for the trainning, validation, and test data
			with tf.name_scope('PREDICTIONS'):
				self.train_prediction = tf.nn.softmax(logits, name='train_prediction')
				self.test_prediction = tf.nn.softmax(model(self.tf_test_samples), name='test_prediction')

			self.merged = tf.merge_all_summaries()

	def run(self):
		"""
		use Session
		"""
		# private function
		def print_confusion_matrix(confusionMatrix):
			print('Confusion Matrix:')
			for i, line in enumerate(confusionMatrix):
				print(line, line[i]/np.sum(line))
			a = 0
			for i, column in enumerate(np.transpose(confusionMatrix, (1, 0))):
				a += (column[i]/np.sum(column))*(np.sum(column)/26000)
				print(column[i]/np.sum(column),)
			print('\n', np.sum(confusionMatrix, a))

		with self.session as session:
			tf.initialize_all_variables().run()

			# Train
			print('Starting Training')
			# batch 100
			for i, samples, labels in get_chunk(train_samples, train_labels, chunkSize=self.batch_size):
				_, l, predictions, summaries = session.run(
					[self.optimizer, self.loss, self.train_prediction, self.merged], 
					feed_dict={self.tf_train_samples: samples, self.tf_train_labels: labels}
					)
				self.writer.add_summary(summaries, i)
				# labels is True Labels
				accuracy, _ = self.accuracy(predictions, labels)
				if i % 50 == 0:
					print('Minibatch loss at step %d: %f' % (i, l))
					print('Minibatch accuracy: %.1f%%' % accuracy)

			# Test
			accuracies = []
			confusionMatrices = []
			for i, samples, labels in get_chunk(test_samples, test_labels, chunkSize=self.test_batch_size):
				result = self.test_prediction.eval(feed_dict={self.tf_test_samples: samples})
				accuracy, cm = self.accuracy(result, labels, need_confusion_matrix=True)
				accuracies.append(accuracy)
				confusionMatrices.append(cm)
				print('Test Accuracy: %.1f%%' % accuracy)
			print('  Average Accuracy:', np.average(accuracies))
			print('Standard Deviation:', np.std(accuracies))
			print_confusion_matrix(np.add.reduce(confusionMatrices))


	def accuracy(self, predictions, labels, need_confusion_matrix=False):
		"""
		accuracy and recall-rate
		@return: accuracy and confusionMatrix as a tuple
		"""
		_predictions = np.argmax(predictions, 1)
		_labels = np.argmax(labels, 1)
		cm = confusion_matrix(_labels, _predictions) if need_confusion_matrix else None
		# == is overloaded for numpy array
		accuracy = (100.0 * np.sum(_predictions == _labels) / predictions.shape[0])
		return accuracy, cm

if __name__ == '__main__':
	net = Network(num_hidden=128, batch_size=100)
	#net.define_graph()
	net.run()