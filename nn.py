import numpy as np

class NeuralNetwork():
    def __init__(self):
        # initialising weights and biases
        self.w_i_h1 = np.random.uniform(-0.5, 0.5, (512, 28 * 28)) # first layer
        self.b_i_h1 = np.random.uniform(-0.5, 0.5, (512, 1))

        self.w_h1_h2 = np.random.uniform(-0.5, 0.5, (128, 512)) # second layer
        self.b_i_h1 = np.random.uniform(-0.5, 0.5, (128, 1))

        self.w_h2_o = np.random.uniform(-0.5, 0.5, (10, 128)) # third layer
        self.w_h2_o = np.random.uniform(-0.5, 0.5, (10, 1))
