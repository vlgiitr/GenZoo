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

trainloader = load.load_mnist(batch_size=100)


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
        z = model.reparameterize(mean, logvar)

        tsne = TSNE(n_components=2, random_state=0)
        z_2d = tsne.fit_transform(z)

        target_ids = range(0, 9)
        y = labels.detach().numpy()

        plt.figure(figsize=(6, 5))
        colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
        for i, c in zip(target_ids, colors):
            ind = np.where(y == i)
            plt.scatter(z_2d[ind, 0], z_2d[ind, 1], c=c)

        plt.show()


model = VAE.make_model()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epoch = 50
print_every = 10
train_model.train(trainloader, epoch, optimizer, model, device, batch_size=100, print_every=5)
t_sne(1000)

z = torch.zeros(105, 8)
for i in range(105):
    z[i] = torch.randn([1, 8])

x_out = model.decoder(z)
train_model.display_grid(grid_size=10, digit_size=28, images=x_out)
