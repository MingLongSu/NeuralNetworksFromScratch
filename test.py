from nn import NeuralNetwork
import numpy as np

# Used for testing newly designed components of the neural network

nn = NeuralNetwork()

b_i_h1 = np.random.uniform(-0.5, 0.5, (10, 1))
print(nn.ReLU(b_i_h1))