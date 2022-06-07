import numpy as np
from fetch_data import fetch_mnist

class NeuralNetwork():
    def __init__(self):
        # fetching mnist data
        (train_images, train_labels), (test_images, test_labels) = fetch_mnist()
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels

        # initialising weights and biases
        self.w_i_h1 = np.random.uniform(-0.5, 0.5, (50, 28 * 28)) # weight 1
        self.b_i_h1 = np.random.uniform(-0.5, 0.5, (50, 1)) # bias 1

        self.w_h1_h2 = np.random.uniform(-0.5, 0.5, (20, 50)) # weight 2
        self.b_h1_h2 = np.random.uniform(-0.5, 0.5, (20, 1)) # bias 2

        self.w_h2_o = np.random.uniform(-0.5, 0.5, (10, 20)) # weight 3
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

        # storing all calculated results in a dict for easy referencing later
        fp_results = { 
            'pre_result_i_h1': pre_result_i_h1,
            'result_i_h1': result_i_h1,
            'pre_result_h1_h2': pre_result_h1_h2,
            'result_h1_h2': result_h1_h2,
            'pre_result_h2_o': pre_result_h2_o,
            'result_h2_o': result_h2_o
        }
        
        return fp_results

    def backward_propogation(self, fp_results, train_label, lr, image):
        error_1 = (2 / len(fp_results['result_h2_o'])) * (fp_results['result_h2_o'] - train_label) # SHAPE: (10, 1)
        update_w_h2_o = self.w_h2_o - lr * error_1 @ np.transpose(fp_results['result_h1_h2']) # to update the weights of w_h2_o, SHAPE: (10, 128)
        update_b_h2_o = self.b_h2_o - lr * error_1 # to update the biases of b_h2_o, SHAPE: (10, 1)

        error_2 = np.transpose(self.w_h2_o) @ error_1 * self.ReLU(fp_results['pre_result_h1_h2'], derive=True) # SHAPE: (128, 1)
        update_w_h1_h2 = self.w_h1_h2 - lr * error_2 @ np.transpose(fp_results['result_i_h1']) # to update the weights of w_h1_h2, SHAPE: (128, 512)
        update_b_h1_h2 = self.b_h1_h2 - lr * error_2 # to update the biases of b_h1_h2, SHAPE: (128, 1)

        error_3 = np.transpose(self.w_h1_h2) @ error_2 * self.ReLU(fp_results['pre_result_i_h1'], derive=True) # SHAPE: (512, 1)
        update_w_i_h1 = self.w_i_h1 - lr * error_3 @ np.transpose(image) # to update the weights of w_i_h1, SHAPE: (512, 784)
        update_b_i_h1 = self.b_i_h1 - lr * error_3 # to update the biases of b_i_h1, SHAPE (512, 1)

        # dict to store the update values for later use
        updates = {
            'update_w_h2_o': update_w_h2_o,
            'update_b_h2_o': update_b_h2_o,
            'update_w_h1_h2': update_w_h1_h2,
            'update_b_h1_h2': update_b_h1_h2,
            'update_w_i_h1': update_w_i_h1,
            'update_b_i_h1': update_b_i_h1
        }

        return updates

    def update_params(self, updates):
        # updating the weights
        self.w_i_h1 = updates['update_w_i_h1']
        self.w_h1_h2 = updates['update_w_h1_h2']
        self.w_h2_o = updates['update_w_h2_o']
        
        # updating the biases
        self.b_i_h1 = updates['update_b_i_h1']
        self.b_h1_h2 = updates['update_b_h1_h2']
        self.b_h2_o = updates['update_b_h2_o']

    def train(self, lr, epochs=1):
        for epoch in range(epochs):
            for image, label in zip(self.train_images, self.train_labels):
                fp_results = self.forward_propogate(image)
                updates = self.backward_propogation(fp_results=fp_results, train_label=label, lr=lr, image=image)
                self.update_params(updates=updates)
            
            
            #print(f"Epoch { epoch + 1 } / { epochs }, Accuracy: ")


'''
nn = NeuralNetwork()
print(nn.b_h2_o)
nn.train(lr=0.01, epochs=10)
print(nn.b_h2_o)
'''