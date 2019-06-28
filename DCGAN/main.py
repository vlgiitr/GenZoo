import tensorflow as tf
tf.enable_eager_execution()

from data_loader import load_data_mnist, load_data_cifar
from model import make_models_mnist, make_models_cifar
from train import train_model, generate_and_save_images

def train_mnist_model_from_scratch(epochs, batch_size):
    dataset = load_data_mnist(batch_size)
    gen_model, disc_model = make_models_mnist()
    train_model(gen_model, disc_model, dataset, epochs, batch_size)
    return gen_model, disc_model

def train_cifar_model_from_scratch(epochs, batch_size):
    dataset = load_data_cifar(batch_size)
    gen_model, disc_model = make_models_cifar()
    train_model(gen_model, disc_model, dataset, epochs, batch_size)
    return gen_model, disc_model

#gen_model, disc_model = train_cifar_model_from_scratch(150, 128)
#                     OR
#gen_model = tf.keras.models.load_model('cifar_gen_model.h5')

#generate 16 sample images from trained model ->
#generate_and_save_images(gen_model)
    
#File will be saved in generated_images folder with name "sample_image_from_epoch_0000.png"