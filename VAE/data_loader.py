import torch
import torch.utils.data
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import pathlib

pathlib.Path('./MNIST_dataset').mkdir(parents=True, exist_ok=True)


def load_mnist(batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./MNIST_dataset', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    return train_loader
