from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths to the MNIST dataset files
images_path = 'C:/Users/abhic/OneDrive/Desktop/Fashion_MNIST_model_training/app/train-images-idx3-ubyte'
labels_path = 'C:/Users/abhic/OneDrive/Desktop/Fashion_MNIST_model_training/app/train-labels-idx1-ubyte'

# Load the dataset
X, y = loadlocal_mnist(images_path=images_path, labels_path=labels_path)

# Check the shape of the data
print(f'Images shape: {X.shape}')
print(f'Labels shape: {y.shape}')

# Directory to save the images
save_dir = 'mnist_images'
os.makedirs(save_dir, exist_ok=True)

# Function to save each image as a separate PNG file
def save_mnist_images(images, labels, n=10, save_dir='mnist_images'):
    for i in range(n):
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'mnist_image_{i}.png'))
        plt.close()

# Save the first 10 images
save_mnist_images(X, y, n=10, save_dir=save_dir)

print(f'First {10} images saved to {save_dir}')
