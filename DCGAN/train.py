import tensorflow as tf
import matplotlib.pyplot as plt

def generator_loss(generated_output):
    gen_loss = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)
    tf.contrib.summary.scalar("Generator Loss", gen_loss)
    return gen_loss

def discriminator_loss(real_output, generated_output):  
    with tf.name_scope("Discriminator_Loss") as scope:
        real_loss = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(real_output), real_output)
        tf.contrib.summary.scalar("Discriminator Loss (Real)", real_loss)
        generated_loss = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(generated_output), generated_output)
        tf.contrib.summary.scalar("Discriminator Loss (Generated)", generated_loss)

        total_loss = real_loss + generated_loss
        tf.contrib.summary.scalar("Discriminator Loss (Total)", total_loss)

    return total_loss

generator_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)

noise_dim = 100
num_examples = 16
random_vector = tf.random.normal([num_examples, noise_dim])

global_step = tf.compat.v1.train.get_or_create_global_step()
writer = tf.contrib.summary.create_file_writer('./')

def train_step(images, gen_model, disc_model, batch_size):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen_model(noise, training=True)
        real_output = disc_model(images, training=True)
        generated_output = disc_model(generated_images, training=True)
        gen_loss = generator_loss(generated_output)
        disc_loss = discriminator_loss(real_output, generated_output)

    gen_grads = gen_tape.gradient(gen_loss, gen_model.variables)
    disc_grads = disc_tape.gradient(disc_loss, disc_model.variables)

    generator_optimizer.apply_gradients(zip(gen_grads, gen_model.variables))
    discriminator_optimizer.apply_gradients(zip(disc_grads, disc_model.variables), global_step=global_step)

train_step = tf.contrib.eager.defun(train_step)

def train_model(gen_model, disc_model, dataset, epochs, batch_size):
    with writer.as_default(), tf.contrib.summary.always_record_summaries():
        for epoch in range(epochs): 
            for images in dataset:
                train_step(images, gen_model, disc_model, batch_size)

            generate_and_save_images(gen_model, epoch+1, random_vector)

            print("Epoch {} done".format(epoch+1))

def generate_and_save_images(gen_model, epoch=0, test_input=random_vector):
    predictions = gen_model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow(predictions[i,:,:,0]*127.5+127.5, cmap='gray')
        plt.axis('off')
    
    plt.savefig('generated_images\sample_image_from_epoch_{:04d}'.format(epoch))