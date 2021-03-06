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

    train_images = np.reshape(train_images, (train_images.shape[0], train_images.shape[1] * train_images.shape[2], 1)) # FROM: (6000, 28, 28), TO: (6000, 28 * 28, 1)
    test_images = np.reshape(test_images, (test_images.shape[0], test_images.shape[1] * test_images.shape[2], 1)) # FROM: (6000, 28, 28), TO: (6000, 28 * 28, 1)

    # One hot encoding the training and test label data
    train_labels = np.eye(10)[train_labels]
    test_labels = np.eye(10)[test_labels]

    # Reshaping the train labels to appear like vectors
    train_labels = np.reshape(train_labels, (train_labels.shape[0], 10, 1))
    test_labels = np.reshape(test_labels, (test_labels.shape[0], 10, 1))

    return (train_images, train_labels), (test_images, test_labels)

'''
(train_images, train_labels), (test_images, test_labels) = fetch_mnist()
plt.imshow(test_images[0], cmap=plt.cm.binary)
plt.show()
'''