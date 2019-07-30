import logging
import os
from os import mkdir
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import webbrowser
from datetime import datetime

import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import configparser
import argparse
from tensorboard import program

from data_loader import load_data_mnist, load_data_cifar
from model import make_models_mnist, make_models_cifar

parser = argparse.ArgumentParser()
parser.add_argument('-config', default='default_mnist.ini', help='Name of config file stored in configs folder')
args = parser.parse_args()

path_to_config = args.config

config = configparser.ConfigParser()
config.read(path_to_config)

exp_name = config['MODEL_VARIABLES']['exp_name']
lr = float(config['MODEL_PARAMETERS']['learning_rate'])
b1 = float(config['MODEL_PARAMETERS']['beta_1'])
EPOCHS = int(config['MODEL_PARAMETERS']['epochs'])
BATCH_SIZE = int(config['MODEL_PARAMETERS']['batch_size'])
dataset_in_use = config['MODEL_VARIABLES']['dataset']
log_frequency = int(config['SAVE_PARAMETERS']['log_frequency'])
model_save_frequency= int(config['SAVE_PARAMETERS']['model_save_frequency'])
grid_size=int(config['SAVE_PARAMETERS']['grid_size'])

exp_path = "experiments/" + exp_name

if(dataset_in_use == "cifar"):
    gen_model, disc_model = make_models_cifar()
    dataset = load_data_cifar(BATCH_SIZE)
    
elif(dataset_in_use == "mnist"):
    gen_model, disc_model = make_models_mnist()
    dataset = load_data_mnist(BATCH_SIZE)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.2)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate = lr, beta_1 = b1)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = lr, beta_1 = b1)

noise_dim = 100
num_examples = grid_size*grid_size
random_vector = tf.random.normal([num_examples, noise_dim])

logs_path = exp_path + "/loss_graph_logs"

writer = tf.summary.create_file_writer(logs_path)

checkpoint_dir = exp_path + '/training_checkpoints/'
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=gen_model,
                                 discriminator=disc_model)

def generator_loss(generated_output):
    return cross_entropy(tf.zeros_like(generated_output), generated_output)

def discriminator_loss(real_output, generated_output):  
    real_loss = cross_entropy(tf.zeros_like(real_output), real_output)
    generated_loss = cross_entropy(tf.ones_like(generated_output), generated_output)

    total_loss = real_loss + generated_loss
    return total_loss

@tf.function
def train_step(images, batch_size):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen_model(noise, training=True)
        real_output = disc_model(images, training=True)
        generated_output = disc_model(generated_images, training=True)
        gen_loss = generator_loss(generated_output)
        disc_loss = discriminator_loss(real_output, generated_output)

    gen_grads = gen_tape.gradient(gen_loss, gen_model.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, disc_model.trainable_variables)

    generator_optimizer.apply_gradients(zip(gen_grads, gen_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_grads, disc_model.trainable_variables))
    return gen_loss, disc_loss

def train_model(dataset, epochs, batch_size):
    step = 0
    for epoch in range(epochs):
        for images in dataset:
            gen_loss, disc_loss = train_step(images, batch_size)
            if (step+1) % log_frequency == 0:
                with writer.as_default():
                    tf.summary.scalar("Generator_Loss", gen_loss, step)
                    tf.summary.scalar("Discriminator_Loss", disc_loss, step)
            step = step+1
            
        if (epoch + 1) % model_save_frequency == 0:
            current_time_as_string = f"{datetime.now():%Y-%m-%d_%H:%M:%S}"
            checkpoint_name = checkpoint_dir + "ckpt_epoch_" + str(epoch + 1)
            checkpoint.write(file_prefix = checkpoint_name)
            print("Checkpoint saved :{}".format(checkpoint_name))

        predictions=gen_model(random_vector, training=False)
        images=predictions*0.5 + 0.5
        generate_and_save_images(epoch+1, images)
        with writer.as_default():
            tf.summary.image("Generated images", images, step, max_outputs=num_examples)

        print("Epoch {} done".format(epoch+1))

image_save_directory = exp_path +  "/generated_images"
if not os.path.exists(image_save_directory):
    mkdir(image_save_directory)

def image_grid(x, size=grid_size):
    t = tf.unstack(x[:size * size], num=size*size, axis=0)
    rows = [tf.concat(t[i*size:(i+1)*size], axis=0) 
            for i in range(size)]
    image = tf.concat(rows, axis=1)
    return image

def generate_and_save_images(epoch, images):
    plt.axis('off')
    plt.title('Generated Images')
    if(images.shape[3]==1):
        plt.imshow(image_grid(images)[:, :, 0], cmap='gray')
    else:
        plt.imshow(image_grid(images))

    plt.savefig(image_save_directory + "/sample_image_from_epoch_{:04d}.png".format(epoch))
    plt.close()

log = logging.getLogger('werkzeug').setLevel(logging.ERROR)
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', logs_path])
url = tb.launch()
print('TensorBoard at {}'.format(url))
webbrowser.open('http://localhost:6006')
train_model(dataset, EPOCHS, BATCH_SIZE)
