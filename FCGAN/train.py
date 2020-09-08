import torch
from data_loader import DataLoad
from model import *
import torch.nn as nn
from torch.utils import tensorboard
from torch.autograd import Variable
from torchvision.utils import save_image,make_grid
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cpu'
k = 4
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128,type=int,help='Enter the batch size')
parser.add_argument('--total_epochs',default=100,type=int,help='Enter the total number of epochs')
parser.add_argument('--dataset',default='mnist',help='Enter the dataset you want the model to train on')
parser.add_argument('--model_save_frequency',default=20,type=int,help='How often do you want to save the model state')
parser.add_argument('--image_sample_frequency',default=20,type=int,help='How often do you want to sample images ')
parser.add_argument('--learning_rate',default=0.0002,type=int)
parser.add_argument('--beta1',default=0.5,type=int,help='beta1 parameter for adam optimizer')
parser.add_argument('--beta2',default=0.999,type=int,help='beta2 parameter for adam optimizer')
parser.add_argument('--z_dim',default=100,type=int,help='Enter the dimension of the noise vector')
parser.add_argument('--exp_name',default='default-mnist',help='Enter the name of the experiment')
args = parser.parse_args()

fixed_noise = torch.randn(16,args.z_dim,device=device)


#Create the experiment folder
if not os.path.exists(args.exp_name):
    os.makedirs(args.exp_name)

def load_data(use_data):
    # Initialize the data loader object
    data_loader = DataLoad()
    # Load training data into the dataloader
    if use_data == 'mnist':
        train_loader = data_loader.load_data_mnist(batch_size=args.batch_size)
    elif use_data == 'cifar10':
        train_loader = data_loader.load_data_cifar10(batch_size=args.batch_size)
    # Return the data loader for the training set
    return train_loader

def save_checkpoint(state,dirpath, epoch):
    #Save the model in the specified folder
    folder_path = dirpath+'/training_checkpoints'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filename = '{}-checkpoint-{}.ckpt'.format(args.dataset,epoch)
    checkpoint_path = os.path.join(folder_path, filename)
    torch.save(state, checkpoint_path)
    print(' checkpoint saved to {} '.format(checkpoint_path))

def generate_image(fakes,image_folder):
    #Function to generate image grid and save
    image_grid = make_grid(fakes.to(device),padding=2,nrow=4,normalize=True)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    save_image(image_grid,filename='{}/img_{}.png'.format(image_folder,epoch))

# Loss function
criterion = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator(args.dataset)
discriminator = Discriminator(args.dataset)

if cuda:
    generator.cuda()
    discriminator.cuda()
    criterion.cuda()
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(args.beta1,args.beta2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(args.beta1,args.beta2))

# Establish convention for real and fake labels during training
real_label = float(1)
fake_label = float(0)

# Load training data
train_loader = load_data(args.dataset)

# Training Loop
# Lists to keep track of progress
# Create the runs directory if it does not exist
if not os.path.exists(args.exp_name+'/tensorboard_logs'):
    os.makedirs(args.exp_name+'/tensorboard_logs')
writer = tensorboard.SummaryWriter(log_dir=args.exp_name+'/tensorboard_logs')
print("Starting Training Loop...")
steps = 0
# For each epoch
for epoch in range(args.total_epochs):
    # Update the discriminator k times before updating generator as specified in the paper
    for i, (imgs, _) in enumerate(train_loader):

        ############################
        # (1) Update discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        # Format batch
        imgs = imgs.to(device)
        # Adversarial ground truths
        valid = Variable(torch.Tensor(imgs.size(0),1).fill_(real_label), requires_grad=False).to(device)
        fake = Variable(torch.Tensor(imgs.size(0),1).fill_(fake_label), requires_grad=False).to(device)
        optimizer_D.zero_grad()
        # Calculate loss on all-real batch
        real_loss = criterion(discriminator(imgs), valid)
        # Generate batch of latent vectors
        noise = Variable(torch.Tensor(np.random.normal(0, 1, (imgs.shape[0], args.z_dim)))).to(device)
        # Generate fake image batch with generator
        gen_imgs = generator(noise)
        # Classify all fake batch with D
        # Calculate D's loss on the all-fake batch
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
        # Add the gradients from the all-real and all-fake batches
        loss_D = real_loss + fake_loss
        # Calculate the gradients
        loss_D.backward()
        #Update D
        optimizer_D.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        # Optimize the generator network only after k steps of optimizing discriminator as
        # specified in the paper. This is done to ensure that the discriminator is being maintained
        # near its optimal solution as long as generator changes slowly enough.
        # Go through the Adversarial nets section in the paper
        # for detailed explanation (https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
        ###########################
        if (epoch+1)%k == 0:

            optimizer_G.zero_grad()
            # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            gen_imgs = generator(noise)
            output = discriminator(gen_imgs)
            # Calculate the probability of the discriminator to classify fake images as real.
            # If the  value of this probability is close to 0, then it means that the generator has
            # successfully learnt to fool the discriminator777
            D_x = output.mean().item()
            # Calculate G's loss based on this output
            loss_G = criterion(output, valid)
            # Calculate gradients for G
            loss_G.backward()
            # Update G
            optimizer_G.step()
            # Output training stats
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\t'
                    % (epoch+1, args.total_epochs, i+1, len(train_loader),
                        loss_D.item(), loss_G.item(), D_x))

            writer.add_scalar('D_x',D_x,steps)
            writer.add_scalar('Discriminator_loss',loss_D,steps)
            writer.add_scalar('Generator_loss',loss_G,steps)
            steps+=1

    if (epoch+1) % args.model_save_frequency == 0:
    # Saved the model and optimizer states
        save_checkpoint({
            'epoch': epoch + 1,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_G' : optimizer_G.state_dict(),
            'optimizer_D' : optimizer_D.state_dict(),
        }, args.exp_name, epoch + 1)
    # Generate images from the generator network
    if epoch % args.image_sample_frequency == 0:
        with torch.no_grad():
            fakes = generator(fixed_noise)
            image_folder = args.exp_name + '/genereated_images'
            generate_images(fakes,image_folder)
writer.close()
