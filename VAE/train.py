import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
import model as VAE

batch_size = 20


def loss_function(x_output, x, mean, logvar):
    bce = F.binary_cross_entropy(x_output, x.view(-1, 784))
    kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return (bce + kld) / (784 * batch_size)


def train(trainloader, epoch, optimiser, model, device):
    for i in range(epoch):
        model.train()
        model.to(device)
        for images, _ in trainloader:
            images = images.to(device)
            optimiser.zero_grad()
            x_output, mean, logvar = model(images)
            loss = loss_function(x_output, images, mean, logvar)
            loss.backward()
            optimiser.step()
