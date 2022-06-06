import numpy as np
import matplotlib.pyplot as plt

def fetch_mnist():
    # Loads the mnist data from the mnist.npz file
    with np.load('./assets/mnist.npz') as file_mnist:
        train_images, train_labels = file_mnist['x_train'], file_mnist['y_train']
        test_images, test_labels = file_mnist['x_test'], file_mnist['y_test']
        
    # Preparing the image data
    train_images = train_images.astype('float32')
    train_images = train_images / 255 # normalising pixel values
    test_images = test_images.astype('float32')
    test_images = test_images / 255 # normalising pixel values

    # One hot encoding the training label data
    train_labels = np.eye(10)[train_labels]

    return (train_images, train_labels), (test_images, test_labels)

'''
(train_images, train_labels), (test_images, test_labels) = fetch_mnist()
plt.imshow(test_images[0], cmap=plt.cm.binary)
plt.show()
'''