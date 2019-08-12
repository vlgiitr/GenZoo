import numpy as np
import torch
import helper
import torch.utils.data
# from torchvision import datasets, transforms
from torch import nn, optim
import model as VAE
# import torch.nn.functional as F
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
import future
from torchvision.utils import save_image

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
image_save_directory = exp_path + "/training_images"
checkpoint_dir = exp_path + '/training_checkpoints/'
logs_path = exp_path + "/loss_graph_logs"
transit_image_directory = exp_path + '/digit_transit/'
t_sne_save_directory = exp_path + '/t_sne/'
pathlib.Path(image_save_directory).mkdir(parents=True, exist_ok=True)
pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(logs_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(transit_image_directory).mkdir(parents=True, exist_ok=True)
pathlib.Path(t_sne_save_directory).mkdir(parents=True, exist_ok=True)

trainloader = load.load_mnist(batch_size)  # load training data


# x, _ = next(iter(trainloader))
# plt.imshow(x[0].view(28, 28), cmap='gray')
# plt.show(block=True)

def display_transit():
    model.eval()
    model.to('cpu')
    images1, labels1 = iter(trainloader).next()
    y1 = labels1.detach().numpy()
    x1 = images1[0]
    i = 0
    while y1[i] == y1[0]:
        i += 1
    x2 = images1[i]
    x1 = x1.view(1, 1, 28, 28)
    x2 = x2.view(1, 1, 28, 28)
    mean1, logvar1 = model.encoder(x1)
    mean2, logvar2 = model.encoder(x2)
    z1 = model.reparameterize(mean1, logvar1)
    z2 = model.reparameterize(mean2, logvar2)

    grid_size = 15
    z_transit = torch.zeros([grid_size * grid_size, 20])

    for i, _ in enumerate(z_transit):
        z_transit[i] = z1 + ((z2 - z1) / (grid_size * grid_size)) * i

    img = model.decoder(z_transit)
    figure = torch.from_numpy(train_model.display_grid(grid_size=15, digit_size=28, images=img)).float()
    save_image(figure, transit_image_directory + 'digit_transit.png')


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
        plt.savefig(t_sne_save_directory + 't_sne_visualization.png')

        plt.show()


model = VAE.make_model()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
epochs = 10
train_model.train(trainloader, epochs, optimizer, model, device,
                  batch_size=batch_size, print_every=5, save_frequency=save_frequency,
                  image_save_directory=image_save_directory, checkpoint_dir=checkpoint_dir)

t_sne(1000)  # using t-sne visualisation of latent space (using 1000 points)

state_dict = torch.load('./experiments/mnist/training_checkpoints/checkpoint6.pth',
                        map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict)

display_transit()
