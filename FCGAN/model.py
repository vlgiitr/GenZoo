import torch
import torch.nn as nn
import torch.functional as F
import numpy as np



class Generator(nn.Module):

    def __init__(self,model_type,z_dim=100):
        super(Generator, self).__init__()

        self.model_type = model_type
        self.image_shape = {'mnist':(1,28,28),
                            'cifar10':(3,32,32)
                        }
        self.models = nn.ModuleDict({
            'mnist': nn.Sequential(
                        nn.Linear(z_dim,128,bias=True),
                        nn.LeakyReLU(0.2,inplace=True),
                        nn.Linear(128,256,bias=True),
                        nn.LeakyReLU(0.2,inplace=True),
                        nn.Linear(256,512,bias=True),
                        nn.LeakyReLU(0.2,inplace=True),
                        nn.Linear(512,1024,bias=True),
                        nn.LeakyReLU(0.2,inplace=True),
                        nn.Linear(1024,int(np.prod(self.image_shape[model_type]))),
                        nn.Tanh()
                    ),
            'cifar10':nn.Sequential(
                        nn.Linear(z_dim,128,bias=True),
                        nn.LeakyReLU(0.2,inplace=True),
                        nn.Linear(128,256,bias=True),
                        nn.LeakyReLU(0.2,inplace=True),
                        nn.Linear(256,512,bias=True),
                        nn.LeakyReLU(0.2,inplace=True),
                        nn.Linear(512,1024,bias=True),
                        nn.LeakyReLU(0.2,inplace=True),
                        nn.Linear(1024,2048,bias=True),
                        nn.LeakyReLU(0.2,inplace=True),
                        nn.Linear(2048,int(np.prod(self.image_shape[model_type]))),
                        nn.Tanh()
                    )
                })

    def forward(self, z):
        img = self.models[self.model_type](z)
        img = img.view(img.size(0), *self.image_shape[self.model_type])
        return img


class Discriminator(nn.Module):

    def __init__(self,model_type):
        super(Discriminator, self).__init__()

        self.model_type = model_type
        self.image_shape = {'mnist':(1,28,28),
                            'cifar10':(3,32,32)
                        }
        self.models = nn.ModuleDict({
                'mnist':nn.Sequential(
                            nn.Linear(int(np.prod(self.image_shape[model_type])), 512),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(512, 256),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(256, 1),
                            nn.Sigmoid(),
                    ),
                'cifar10':nn.Sequential(
                            nn.Linear(int(np.prod(self.image_shape[model_type])), 1024),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(1024,512),
                            nn.LeakyReLU(0.2,inplace=True),
                            nn.Linear(512, 256),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(256, 1),
                            nn.Sigmoid(),
                    )
                })

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        output = self.models[self.model_type](img_flat)
        return output
