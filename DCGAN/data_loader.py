from tensorflow.keras.datasets import mnist
from tensorflow.data import Dataset

buffer_size = 60000

def load_data(batch_size):
    (images, _), (_, _) = mnist.load_data()
    images = images.reshape(images.shape[0], 28, 28, 1).astype('float32')
    images = (images - 127.5)/127.5

    dataset = Dataset.from_tensor_slices(images).shuffle(buffer_size).batch(batch_size, drop_remainder = True)
    
    return dataset