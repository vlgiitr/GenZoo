import torch
import torch.utils.data
from torchvision import datasets, transforms
from torch import nn, optim
import model as VAE
import torch.nn.functional as F
import data_loader as load
from tensorboardX import SummaryWriter
import  train.py as train_model
import matplotlib
import matplotlib.pyplot as plt


trainloader = load.load_mnist(batch_size=60)
x = trainloader.next()
plt.imshow(x.numpy()[0], cmap='gray')

model = VAE.make_model()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
device = ('cuda' if torch.cuda.is_available() else 'cpu')
epoch = 10

train_model.train(trainloader, epoch, optimizer, model, device)



