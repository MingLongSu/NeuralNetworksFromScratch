from dnn import NeuralNetwork
from fetch_data import fetch_mnist
import numpy as np

# Used for testing newly designed components of the neural network

(images, labels), (images_2, labels_2) = fetch_mnist()
img = images[0]
lbl = labels[0]

nn = NeuralNetwork()
print(nn.b_h2_o)
nn.train(lr=0.01, epochs=10)
print(nn.b_h2_o)