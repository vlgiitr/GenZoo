import tensorflow as tf
tf.enable_eager_execution()

from data_loader import load_data
from model import make_models
from train import train_model, generate_and_save_images

EPOCHS = 50
BATCH_SIZE = 32
dataset = load_data(BATCH_SIZE)

gen_model, disc_model = make_models()

#gen_model.load_model(gen_model.h5) -> load saved gen_model.h5

#train_model(gen_model, disc_model, dataset, EPOCHS, BATCH_SIZE) -> train the model

#generate_and_save_images(gen_model) -> generate 16 random samples and save them