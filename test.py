from dnn import NeuralNetwork
from fetch_data import fetch_mnist
import numpy as np

# Used for testing newly designed components of the neural network

(images, labels), (images_2, labels_2) = fetch_mnist()
img = images[0]
lbl = labels[0]

nn = NeuralNetwork()

results = nn.forward_propogate(img)
update = nn.backward_propogation(results, lbl, 0.01, img)
print(update['update_w_h2_o'].shape)
print(update['update_w_h1_h2'].shape)
print(update['update_w_i_h1'].shape)