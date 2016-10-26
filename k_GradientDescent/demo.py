import network
import mnist_loader

train_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])
net.SGD(train_data, 30, 10, 3.0, test_data=test_data)
