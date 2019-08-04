import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import matplotlib.pyplot as plt
from model import make_models_cifar, make_models_mnist

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', help='Dataset to generate image from (MNIST/CIFAR)')
parser.add_argument('-load_path', help='Directory of checkpoint to load')
parser.add_argument('-grid_size', help='Grid size for generating images. Will generate grid_size*grid_size images')
parser.add_argument('-save_path', help='Path for saving image')
args = parser.parse_args()

grid_size = int(args.grid_size)

if(args.dataset == 'mnist'):
    gen_model, _ = make_models_mnist()

elif(args.dataset == 'cifar'):
    gen_model, _ = make_models_cifar()

ckpt = tf.train.Checkpoint(generator=gen_model)
ckpt.restore(args.load_path)

def image_grid(x, size):
    t = tf.unstack(x[:size * size], num=size*size, axis=0)
    rows = [tf.concat(t[i*size:(i+1)*size], axis=0) 
            for i in range(size)]
    image = tf.concat(rows, axis=1)
    return image

test_vector = tf.random.normal([grid_size*grid_size, 100])
predictions = gen_model(test_vector, training=False)
images = predictions*0.5 + 0.5
plt.figure(figsize=(grid_size, grid_size), dpi=100)
plt.axis('off')
plt.title('Generated Images')
if(images.shape[3]==1):
    plt.imshow(image_grid(images, grid_size)[:, :, 0], cmap='gray')
else:
    plt.imshow(image_grid(images, grid_size))

if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)
plt.savefig(args.save_path+'/generated_image_{}.png'.format(args.dataset))
plt.show()
