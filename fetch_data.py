import numpy as np
import matplotlib.pyplot as plt

def fetch_mnist():
    # Loads the mnist data from the mnist.npz file
    with np.load('./assets/mnist.npz') as file_mnist:
        train_images, train_labels = file_mnist['x_train'], file_mnist['y_train']
        test_images, test_labels = file_mnist['x_test'], file_mnist['y_test']

    return (train_images, train_labels), (test_images, test_labels)

(train_images, train_labels), (test_images, test_labels) = fetch_mnist()


'''
plt.imshow(test_images[0], cmap=plt.cm.binary)
plt.show()
'''