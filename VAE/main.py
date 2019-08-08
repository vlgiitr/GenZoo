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

trainloader = load.load_mnist(batch_size=100)
x, _ = next(iter(trainloader))
print(x.numpy)
plt.imshow(x[0].view(28, 28), cmap='gray')
plt.show(block=True)

model = VAE.make_model()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
device ='cpu'
epoch = 10
print_every = 10
train_model.train(trainloader, epoch, optimizer, model, device, print_every)
