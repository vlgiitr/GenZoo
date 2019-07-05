import tensorflow as tf
import matplotlib.pyplot as plt
import configparser
import argparse
import os
from os import mkdir

from data_loader import load_data_mnist, load_data_cifar
from model import make_models_mnist, make_models_cifar

parser = argparse.ArgumentParser()
parser.add_argument('-config', default='default.ini', help='Name of config file stored in configs folder')
args = parser.parse_args()

path_to_config = args.config

config = configparser.ConfigParser()
config.read(path_to_config)

exp_name = config['DEFAULT']['exp_name']
lr = float(config['DEFAULT']['learning_rate'])
b1 = float(config['DEFAULT']['beta_1'])
EPOCHS = int(config['DEFAULT']['epochs'])
BATCH_SIZE = int(config['DEFAULT']['batch_size'])
dataset_in_use = config['DEFAULT']['dataset']

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
num_examples = 16
random_vector = tf.random.normal([num_examples, noise_dim])

logs_path = exp_path + "/loss_graph_logs"

writer = tf.summary.create_file_writer(logs_path)

checkpoint_dir = exp_path + '/training_checkpoints/'
checkpoint_prefix = checkpoint_dir + '/ckpt'
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
            with writer.as_default():
                tf.summary.scalar("Generator_Loss", gen_loss, step)
                tf.summary.scalar("Discriminator_Loss", disc_loss, step)
            step = step+1
            
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        generate_and_save_images(epoch+1, random_vector)

        print("Epoch {} done".format(epoch+1))

image_save_directory = exp_path +  "/generated_images"
if not os.path.exists(image_save_directory):
    mkdir(image_save_directory)

def image_grid(x, size=4):
    t = tf.unstack(x[:size * size], num=size*size, axis=0)
    rows = [tf.concat(t[i*size:(i+1)*size], axis=0) 
            for i in range(size)]
    image = tf.concat(rows, axis=1)
    return image

def generate_and_save_images(epoch=0, test_input=tf.random.normal([16,100])):
    predictions = gen_model(test_input, training=False)
    images = predictions*0.5 + 0.5

    plt.axis('off')
    plt.title('Generated Images')
    plt.imshow(image_grid(images))

    plt.savefig(image_save_directory + "/sample_image_from_epoch_{:04d}.png".format(epoch))
    plt.close()


train_model(dataset, EPOCHS, BATCH_SIZE)