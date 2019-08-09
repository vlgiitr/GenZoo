import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
import model as VAE
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt


def loss_function(out, target, mean, logvar, batch_size):
    bce = F.binary_cross_entropy(out.view(-1, 784), target.view(-1, 784))
    kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return bce + (kld / (784 * batch_size))


def display_grid(grid_size, digit_size, images):
    figure = np.zeros((digit_size * grid_size, digit_size * grid_size))

    # Construct grid of latent variable values
    grid_x = norm.ppf(np.linspace(0, 0.9, grid_size))
    grid_y = norm.ppf(np.linspace(0, 0.9, grid_size))

    counter = 0
    # decode for each square in the grid
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            digit = images[counter].squeeze().cpu().detach().numpy()
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit
            counter += 1

    plt.figure(figsize=(grid_size, grid_size))
    plt.imshow(figure, cmap='bone')
    plt.show()


def train(trainloader, num_epoch, optimiser, model, device, batch_size, print_every):
    count = 0
    for epoch in range(num_epoch):
        model.train()
        model.to(device)
        for i, (images, _) in enumerate(trainloader):
            images = images.to(device)
            optimiser.zero_grad()
            x_output, mean, logvar = model(images)
            loss = loss_function(x_output, images, mean, logvar, batch_size)
            loss.backward()
            optimiser.step()
           

        if count % print_every == 0:
            model.eval()
            display_grid(10, 28, x_output)

        count += 1
