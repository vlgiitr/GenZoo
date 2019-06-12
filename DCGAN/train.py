import tensorflow as tf
import matplotlib.pyplot as plt

def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)

def discriminator_loss(real_output, generated_output):
    real_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(real_output), real_output)
    generated_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(generated_output), generated_output)

    total_loss = real_loss + generated_loss

    return total_loss

generator_optimizer = tf.train.AdamOptimizer(1e-4)
discriminator_optimizer = tf.train.AdamOptimizer(1e-4)

noise_dim = 100
num_examples = 16
random_vector = tf.random.normal([num_examples, noise_dim])

def train_step(images, gen_model, disc_model):
    noise = tf.random.normal([images.shape[0], noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen_model(noise, training=True)
        real_output = disc_model(images, training=True)
        generated_output = disc_model(generated_images, training=True)
        gen_loss = generator_loss(generated_output)
        disc_loss = discriminator_loss(real_output, generated_output)

    gen_grads = gen_tape.gradient(gen_loss, gen_model.variables)
    disc_grads = disc_tape.gradient(disc_loss, disc_model.variables)

    generator_optimizer.apply_gradients(zip(gen_loss, gen_model.variables))
    discriminator_optimizer.apply_gradients(zip(disc_loss, disc_model.variables))

train_step = tf.contrib.eager.defun(train_step)

def train_model(gen_model, disc_model, dataset, epochs):
    for epoch in range(epochs):
        print("Started")
        i = 1
        for images in dataset:
            train_step(images, gen_model, disc_model)
            print("Minibatch {} Done!".format(i))
            i = i+1
        
        if epoch%5 == 0:
            generate_and_save_images(gen_model, epoch+1, random_vector)

        print("Epoch {} done".format(epoch+1))

def generate_and_save_images(gen_model, epoch, test_input):
    predictions = gen_model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4,4,i)
        plt.imshow(predictions[i,:,:,0]*127.5+127.5, cmap='grey')
        plt.axis('off')
    
    plt.savefig('generated_images\sample_image_from_epoch_{:04d}'.format(epoch))