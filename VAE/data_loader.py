import torch
import torch.utils.data
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F


def load_mnist(batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/home/ayushtues/VAE/GenZoo/VAE/MNIST', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    return train_loader
