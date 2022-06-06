import numpy as np

class NeuralNetwork():
    def __init__(self):
        # initialising weights and biases
        self.w_i_h1 = np.random.uniform(-0.5, 0.5, (512, 28 * 28)) # weight 1
        self.b_i_h1 = np.random.uniform(-0.5, 0.5, (512, 1)) # bias 1

        self.w_h1_h2 = np.random.uniform(-0.5, 0.5, (128, 512)) # weight 2
        self.b_h1_h2 = np.random.uniform(-0.5, 0.5, (128, 1)) # bias 2

        self.w_h2_o = np.random.uniform(-0.5, 0.5, (10, 128)) # weight 3
        self.b_h2_o = np.random.uniform(-0.5, 0.5, (10, 1)) # bias 3

    def ReLU(self, inp, derive=False):
        if derive:
            return inp > 0 
        
        return np.maximum(0, inp)

    def Softmax(self, inp):
        return np.exp(inp) / np.sum(np.exp(inp))

    def forward_propogate(self, image):
        # calculation for input to hidden 1
        pre_result_i_h1 = self.w_i_h1 @ image + self.b_i_h1
        result_i_h1 = self.ReLU(pre_result_i_h1, False) # SHAPE: (512, 1)

        # calculatioon for hidden 1 to hidden 2
        pre_result_h1_h2 = self.w_h1_h2 @ result_i_h1 + self.b_h1_h2
        result_h1_h2 = self.ReLU(pre_result_h1_h2, False) # SHAPE: (128, 1)

        # calculation for hidden 2 to output 
        pre_result_h2_o = self.w_h2_o @ result_h1_h2 + self.b_h2_o
        result_h2_o = self.Softmax(pre_result_h2_o) # SHAPE: (10, 1)
        
        return result_h2_o
