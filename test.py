from nn import NeuralNetwork
from fetch_data import fetch_mnist
import numpy as np

# Used for testing newly designed components of the neural network

(images, labels), (images_2, labels_2) = fetch_mnist()
img = images[0]

nn = NeuralNetwork()

results = nn.forward_propogate(img)
print(np.argmax(results))
print(results)