import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, z_dim=20, keep_prob=0.2):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 5, 3, padding=1), nn.LeakyReLU(), nn.MaxPool2d(2, 2), nn.BatchNorm2d(5),
                                  nn.Conv2d(5, 10, 3, padding=1), nn.LeakyReLU(), nn.MaxPool2d(2, 2),
                                  nn.BatchNorm2d(10), nn.Conv2d(10, 16, 3, padding=1))
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(z_dim, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 784)
        self.decode = nn.Sequential(nn.ConvTranspose2d(16, 16, 3, padding=1),
                                    nn.BatchNorm2d(16),
                                    nn.ConvTranspose2d(16, 3, 8),
                                    nn.BatchNorm2d(3),
                                    nn.ConvTranspose2d(3, 1, 15))

        self.mean = nn.Linear(256, z_dim)
        self.logvar = nn.Linear(256, z_dim)
        self.drop = nn.Dropout(keep_prob)
        self.bn1 = nn.BatchNorm1d(num_features=256)

    def encoder(self, x):
        x = self.conv(x)
        x = x.view([-1, 784])
        x = self.drop(nn.LeakyReLU(self.fc1(x)))
        x = self.drop(self.bn1(nn.LeakyReLU(self.fc2(x))))

        mean = self.mean(x)
        logvar = self.logvar(x)

        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return mean + eps * std

    def decoder(self, z):
        z = self.fc3(z)
        z = self.fc4(z)
        z = self.fc5(z)
        z = z.view([-1, 16, 7, 7])

