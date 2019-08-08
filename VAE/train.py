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


def loss_function(x_output, x, mean, logvar):
    bce = F.binary_cross_entropy(x_output.view(-1, 784), x.view(-1, 784))
    kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return (bce + kld) / (784 * batch_size)


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

        print(x_output.shape)
        model.eval()
        plt.imshow(x_output[0].squeeze().detach().numpy(), cmap='gray')
        plt.show(block=True)
