from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Reshape, LeakyReLU, Dropout, Flatten, Dense, BatchNormalization

def make_models():
    '''
    Returns 2 models -> gen_model, disc_model
    1.) Generator Model
    Takes 100 dimentional noise as input to produce a 28x28x1 image resembling MNIST database

    2.) Discriminator Model
    Takes a 28x28x1 image and labels it as Real(1) or Fake/Generated(0)
    '''
    gen_model = Sequential()
    gen_model.add(Dense(7*7*256, use_bias=False, input_shape=(100,)))
    gen_model.add(BatchNormalization())
    gen_model.add(LeakyReLU())

    gen_model.add(Reshape((7,7,256)))
    gen_model.add(Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False))
    gen_model.add(BatchNormalization())
    gen_model.add(LeakyReLU())

    gen_model.add(Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    gen_model.add(BatchNormalization())
    gen_model.add(LeakyReLU())

    gen_model.add(Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False))

    disc_model = Sequential()
    disc_model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    disc_model.add(LeakyReLU())
    disc_model.add(Dropout(0.3))
  
    disc_model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    disc_model.add(LeakyReLU())
    disc_model.add(Dropout(0.3))
  
    disc_model.add(Flatten())
    disc_model.add(Dense(1))

    return gen_model, disc_model