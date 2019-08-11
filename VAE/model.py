import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, z_dim=8, keep_prob=0.2):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 5, 3, padding=1), nn.LeakyReLU(), nn.MaxPool2d(2, 2), nn.BatchNorm2d(5),
                                  nn.Conv2d(5, 10, 3, padding=1), nn.LeakyReLU(), nn.MaxPool2d(2, 2),
                                  nn.BatchNorm2d(10), nn.Conv2d(10, 16, 3, padding=1))
        # conv model sort of based on  AlexNet  using leaky ReLU
        self.fc1 = nn.Linear(784, 512)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        self.fc2 = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.fc3 = nn.Linear(z_dim, 256)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        self.fc4 = nn.Linear(256, 512)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)
        self.fc5 = nn.Linear(512, 784)
        nn.init.xavier_normal_(self.fc5.weight)
        nn.init.zeros_(self.fc5.bias)
        # a bunch of linear layers with Xavier Initialisation
        self.decode = nn.Sequential(nn.ConvTranspose2d(16, 10, 3, padding=1),
                                    nn.BatchNorm2d(10),
                                    nn.ConvTranspose2d(10, 5, 8),
                                    nn.BatchNorm2d(5),
                                    nn.ConvTranspose2d(5, 1, 15))
        # decoder made of transpose convolutions (deconv) for upsampling and batchnorm
        self.mean = nn.Linear(256, z_dim)
        nn.init.xavier_normal_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        self.logvar = nn.Linear(256, z_dim)
        nn.init.xavier_normal_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)
        self.drop = nn.Dropout(keep_prob)
        self.bn1 = nn.BatchNorm1d(num_features=256)

        self.decode.apply(self.init_weights)
        self.conv.apply(self.init_weights)
        # apply Xavier Initialisation to Convolution layers

    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def encoder(self, x):
        x = self.conv(x)
        x = x.view([-1, 784])
        x = self.drop(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.drop(x) # applying dropout in encoder but not in decoder
        x = F.leaky_relu(self.fc2(x))
        x = self.drop(x)
        x = self.bn1(x)
        mean = self.mean(x)
        logvar = self.logvar(x)

        return mean, logvar  # mean and log-variance of the approximation of the latent variable distribution

    def reparameterize(self, mean, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return mean + eps * std
            # The re-parametrization trick to allow back-propagation
            # (separates the non differentiable sampling operation from the computation graph)
        else:
            return mean
            # return mean if testing reconstruction of input images as its most probable

    def decoder(self, z):
        z = self.fc3(z)
        z = self.fc4(z)
        z = self.fc5(z) # no dropout in decoder
        z = z.view([-1, 16, 7, 7])  # reshape to fit the input size of the ConvTranspose layer
        z = self.decode(z)
        z = torch.sigmoid(z)  # hence we can apply the BCE loss for reconstruction error
        return z

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)  # sample z through re-parametrization trick
        x_output = self.decoder(z)  # obtain the reconstructed image
        return x_output, mean, logvar


def make_model():
    model = VAE()
    return model
