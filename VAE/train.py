import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
import model as VAE
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm


def loss_function(out, target, mean, logvar, batch_size):
    bce = F.binary_cross_entropy(out.view(-1, 784), target.view(-1, 784))
    # BCE loss to maximize likelihood of training data
    kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    # KL Divergence loss to measure the difference
    # between a standard gaussian and our gaussian approximation of the latent variables
    return bce, bce + (kld / (784 * batch_size))
    # scale KL Divergence loss by (dimensions*number_of_examples) to ensure proper weightage
    # between reconstruction (BCE) and standard gaussian distribution (KLD)


def display_grid(grid_size, digit_size, images):
    figure = np.zeros((digit_size * grid_size, digit_size * grid_size))
    # we create a very big image and divide it into different sub-images
    grid_x = norm.ppf(np.linspace(0, 0.9, grid_size))
    grid_y = norm.ppf(np.linspace(0, 0.9, grid_size))

    counter = 0
    # decode for each square in the grid
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            digit = images[counter].squeeze().cpu().detach().numpy()
            # squeeze to reduce from [1 ,28 ,28] to [28,28]
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit
            # assign a particular square to a image
            counter += 1

    plt.figure(figsize=(grid_size, grid_size))
    plt.imshow(figure, cmap='bone')
    plt.show()

    return figure


def train(trainloader, num_epoch, optimiser, model, device, batch_size, print_every,
          save_frequency, image_save_directory, checkpoint_dir):
    writer = SummaryWriter()

    count = 0
    for epoch in range(num_epoch):
        model.train()
        model.to(device)
        reconstruction_loss = 0
        pbar = tqdm(total=len(trainloader), unit='batch', ncols=80, desc=f'Epoch {epoch}: ')
        for i, (images, _) in enumerate(trainloader):
            images = images.to(device)
            optimiser.zero_grad()
            x_output, mean, logvar = model(images)
            # get the reconstructed output and the mean and logvar of the latent vairables
            bce, loss = loss_function(x_output, images, mean, logvar, batch_size)
            loss.backward()
            reconstruction_loss += bce
            optimiser.step()
            writer.add_scalar('loss', reconstruction_loss, epoch)
            pbar.update(1)

        pbar.close()

        if count % print_every == 0:
            model.eval()
            figure = torch.from_numpy(display_grid(5, 28, x_output)).float()
            save_image(figure, image_save_directory + '/img_from_epoch' + str(count) + '.png')
            # display the reconstructed output for the input training images

        if count % save_frequency == 0:
            torch.save(model.state_dict(), checkpoint_dir + '/checkpoint' +
                       str(int(count / save_frequency) + 1) + '.pth')

        count += 1
