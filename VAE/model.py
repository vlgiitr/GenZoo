import torch
import torch.nn as nn
import torch.nn.functional as F

__name__ = "model.py"


class VAE(nn.Module):
    def __init__(self, z_dim=8, keep_prob=0.2):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 5, 3, padding=1), nn.LeakyReLU(), nn.MaxPool2d(2, 2), nn.BatchNorm2d(5),
                                  nn.Conv2d(5, 10, 3, padding=1), nn.LeakyReLU(), nn.MaxPool2d(2, 2),
                                  nn.BatchNorm2d(10), nn.Conv2d(10, 16, 3, padding=1))
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
        self.decode = nn.Sequential(nn.ConvTranspose2d(16, 10, 3, padding=1),
                                    nn.BatchNorm2d(10),
                                    nn.ConvTranspose2d(10, 5, 8),
                                    nn.BatchNorm2d(5),
                                    nn.ConvTranspose2d(5, 1, 15))

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

    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def encoder(self, x):
        x = self.conv(x)
        x = x.view([-1, 784])
        x = self.drop(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.drop(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.drop(x)
        x = self.bn1(x)
        mean = self.mean(x)
        logvar = self.logvar(x)

        return mean, logvar

    def reparameterize(self, mean, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean

    def decoder(self, z):
        z = self.fc3(z)
        z = self.fc4(z)
        z = self.fc5(z)
        z = z.view([-1, 16, 7, 7])
        z = self.decode(z)
        z = torch.sigmoid(z)
        return z

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_output = self.decoder(z)
        return x_output, mean, logvar


def make_model():
    model = VAE()
    return model
