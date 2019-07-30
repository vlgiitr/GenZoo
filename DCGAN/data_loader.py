from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.data import Dataset

buffer_size_mnist = 60000
buffer_size_cifar = 50000

def load_data_mnist(batch_size):
    '''
    Returns a nested stucture of tensors based on MNIST database.
    Will be divided into (60000/batch_size) batches of (batch_size) each.
    '''
    (images, _), (_, _) = mnist.load_data()

    size = images.shape[0]
    num_batches = size//batch_size

    images = images.reshape(images.shape[0], 28, 28, 1).astype('float32')
    images = (images - 127.5)/127.5

    dataset = Dataset.from_tensor_slices(images).shuffle(buffer_size_mnist).batch(batch_size, drop_remainder = True)
    
    return dataset, num_batches

def load_data_cifar(batch_size):
    '''
    Returns a nested stucture of tensors based on CIFAR-10 database.
    Will be divided into (50000/batch_size) batches of (batch_size) each.
    '''
    (images, _), (_, _) = cifar10.load_data()

    size = images.shape[0]
    num_batches = size//batch_size

    images = images.reshape(images.shape[0], 32, 32, 3).astype('float32')
    images = (images - 127.5)/127.5

    dataset = Dataset.from_tensor_slices(images).shuffle(buffer_size_cifar).batch(batch_size, drop_remainder = True)
    
    return dataset, num_batches
    