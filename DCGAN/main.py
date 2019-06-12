import tensorflow as tf
tf.enable_eager_execution()

from data_loader import load_data
from model import make_models
from train import train_model

EPOCHS = 1
BATCH_SIZE = 32
dataset = load_data(BATCH_SIZE)

gen_model, disc_model = make_models()

train_model(gen_model, disc_model, dataset, EPOCHS, BATCH_SIZE)