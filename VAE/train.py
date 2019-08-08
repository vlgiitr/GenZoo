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


batch_size = 100


def loss_function(out, target, mean, logvar):
    bce = F.binary_cross_entropy(x_output.view(-1, 784), x.view(-1, 784))
    kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return bce + (kld / (784 * batch_size))


def train(trainloader, num_epoch, optimiser, model, device, print_every):
    for epoch in range(num_epoch):
        model.train()
        model.to(device)
        for i, (images, _) in enumerate(trainloader):
            images = images.to(device)
            optimiser.zero_grad()
            x_output, mean, logvar = model(images)
            loss = loss_function(x_output, images, mean, logvar)
            loss.backward()
            optimiser.step()

        model.to('cpu')
        model.eval()
        out, _, _ = model(images)

        # Display a 2D manifold of the digits
        n = 10  # figure with 20x20 digits
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))

        # Construct grid of latent variable values
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

        counter = 0
        # decode for each square in the grid
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                digit = out[counter].squeeze().detach().numpy()
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit
                counter += 1

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='bone')
        plt.show()

