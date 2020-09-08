from tensorflow.python.keras.api._v2.keras.models import Sequential
from tensorflow.python.keras.api._v2.keras.layers import Conv2D, Conv2DTranspose, Reshape, LeakyReLU, Dropout, Flatten, Dense, BatchNormalization

def make_models_mnist():
    '''
    Returns 2 models -> gen_model, disc_model
    1.) Generator Model
    Takes 100 dimensional noise as input to produce a 28x28x1 image resembling MNIST database

    2.) Discriminator Model
    Takes a 28x28x1 image and labels it as Real(1) or Fake/Generated(0)
    '''
    gen_model = Sequential()
    gen_model.add(Dense(7*7*256, use_bias=False, input_shape=(100,)))
    gen_model.add(BatchNormalization())
    gen_model.add(LeakyReLU(alpha=0.2))

    gen_model.add(Reshape((7,7,256)))
    gen_model.add(Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False))
    gen_model.add(BatchNormalization())
    gen_model.add(LeakyReLU(alpha=0.2))

    gen_model.add(Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    gen_model.add(BatchNormalization())
    gen_model.add(LeakyReLU(alpha=0.2))

    gen_model.add(Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))

    disc_model = Sequential()
    disc_model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    disc_model.add(LeakyReLU(alpha=0.2))
    disc_model.add(Dropout(0.5))
  
    disc_model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    disc_model.add(LeakyReLU(alpha=0.2))
    disc_model.add(Dropout(0.5))
  
    disc_model.add(Flatten())
    disc_model.add(Dense(1))

    return gen_model, disc_model

def make_models_cifar():
    gen_model = Sequential()
    gen_model.add(Dense(4*4*512, use_bias=False, input_shape=(100,)))
    gen_model.add(BatchNormalization())
    gen_model.add(LeakyReLU(alpha=0.2))

    gen_model.add(Reshape((4,4,512)))
    gen_model.add(Conv2DTranspose(256,(5,5), strides=(2,2), padding='same', use_bias=False))
    gen_model.add(BatchNormalization())
    gen_model.add(LeakyReLU(alpha=0.2))

    gen_model.add(Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False))
    gen_model.add(BatchNormalization())
    gen_model.add(LeakyReLU(alpha=0.2))

    gen_model.add(Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
    
    disc_model = Sequential()
    disc_model.add(Conv2D(128, (5,5), strides=(2,2), padding='same', use_bias=False))
    disc_model.add(LeakyReLU(alpha=0.2))
    disc_model.add(Dropout(0.5))

    disc_model.add(Conv2D(256, (5,5), strides=(2,2), padding='same', use_bias=False))
    disc_model.add(LeakyReLU(alpha=0.2))
    disc_model.add(Dropout(0.5))

    disc_model.add(Conv2D(512, (5,5), strides=(2,2), padding='same', use_bias=False))
    disc_model.add(LeakyReLU(alpha=0.2))
    disc_model.add(Dropout(0.5))

    disc_model.add(Flatten())
    disc_model.add(Dense(1))

    return gen_model, disc_model
    
