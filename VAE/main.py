import numpy as np
import torch
import helper
import torch.utils.data
from torchvision import datasets, transforms
from torch import nn, optim
import model as VAE
import torch.nn.functional as F
import data_loader as load
from tensorboardX import SummaryWriter
import train as train_model
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import configparser
import argparse
import os
from os import mkdir
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="the configuration file of the model", default='./configs/mnist.ini')  # for
# python main.py --config ./configs/mnist2.ini (example of terminal command)
args = parser.parse_args()  # create a parser to take input from command line

print(args.config)
config = configparser.ConfigParser()  # parses a .ini file made specially for configs
config.read(args.config)

lr = float(config['MODEL_PARAMETERS']['lr'])
batch_size = int(config['MODEL_PARAMETERS']['batch_size'])
epochs = int(config['MODEL_PARAMETERS']['epochs'])
save_frequency = int(config['SAVE_PARAMETERS']['model_save_frequency'])
exp_name = config['MODEL_VARIABLES']['exp_name']
# pathlib.Path('./experiments').mkdir(parents=True, exist_ok=True)
exp_path = "./experiments/" + exp_name
pathlib.Path(exp_path).mkdir(parents=True, exist_ok=True)
image_save_directory = exp_path + "/generated_images"
checkpoint_dir = exp_path + '/training_checkpoints/'
logs_path = exp_path + "/loss_graph_logs"

pathlib.Path(image_save_directory).mkdir(parents=True, exist_ok=True)
pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(logs_path).mkdir(parents=True, exist_ok=True)

trainloader = load.load_mnist(batch_size)  # load training data


# x, _ = next(iter(trainloader))
# plt.imshow(x[0].view(28, 28), cmap='gray')
# plt.show(block=True)

def t_sne(size_datapoints):
    model.to('cpu')
    model.eval()
    with torch.no_grad():
        dataloader = load.load_mnist(size_datapoints)
        data = iter(dataloader)
        images, labels = data.next()
        mean, logvar = model.encoder(images)
        z = model.reparameterize(mean, logvar)  # latent space representation of dataset

        tsne = TSNE(n_components=2, random_state=0)
        z_2d = tsne.fit_transform(z)  # Apply t-sne to convert to 2D (n_components) dimensions

        target_ids = range(0, 9)
        y = labels.detach().numpy()  # need to detach labels from computation graph to use .numpy()

        plt.figure(figsize=(6, 5))
        colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
        for i, c in zip(target_ids, colors):  # zip creates iterator
            ind = np.where(y == i)  # returns indices where y==i
            plt.scatter(z_2d[ind, 0], z_2d[ind, 1], c=c)  # plt.scatter(x-coordinate , y-coordinate , color)

        plt.show()


model = VAE.make_model()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

train_model.train(trainloader, epochs, optimizer, model, device,
                  batch_size=batch_size, print_every=5, save_frequency=save_frequency,
                  image_save_directory=image_save_directory, checkpoint_dir=checkpoint_dir)

t_sne(1000)  # using t-sne visualisation of latent space (using 1000 points)

z = torch.zeros(105, 8)
for i in range(105):
    z[i] = torch.randn([1, 8])  # creates random gaussian inputs

x_out = model.decoder(z)
train_model.display_grid(grid_size=10, digit_size=28, images=x_out)  # display the image constructed from random
# gaussian inputs
