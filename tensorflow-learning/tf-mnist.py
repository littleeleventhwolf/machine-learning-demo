from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

# 55000
print("Training data size: ", mnist.train.num_examples)

# 5000
print("Validating data size: ", mnist.validation.num_examples)

# 10000
print("Testing data size: ", mnist.test.num_examples)

# [0. 0. 0. ... 0.380 0.376 ... 0.]
print("Example training data: ", mnist.train.images[0])

# [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
print("Example training data label: ", mnist.train.labels[0])