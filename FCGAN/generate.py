import torch
from torchvision.utils import save_image,make_grid
from data_loader import DataLoad
from model import *
import argparse




parser = argparse.ArgumentParser()
parser.add_argument('--dataset',help='dataset to generate image from [mnist/cifar10]')
parser.add_argument('--load_path', help='Directory of checkpoint to load')
parser.add_argument('--grid_size',default=4, help='Grid size for generating images. Will generate grid_size*grid_size images')
parser.add_argument('--save_path', help='Path for saving image')
parser.add_argument('--z_dim',default=100,help='The noise dimension as used in training')
args = parser.parse_args()
grid_size = int(args.grid_size)
#Initialize the model
model = Generator(args.dataset)
#Load the checkpoint
checkpoint = torch.load(args.load_path,map_location='cpu')
# Load the generator model to generate new images
model.load_state_dict(checkpoint['generator'])
#model.eval() to set dropout and batch normalization layers to evaluation mode
#before running inference. Failing to do this will yield inconsistent inference results.
model.eval()
noise = torch.randn(grid_size*grid_size,args.z_dim)

gen_images = model(noise)
image_grid = make_grid(gen_images,padding=2,nrow=grid_size,normalize=True)
# Image saved in the specified folder
save_image(image_grid,filename='{}/generated_image_{}.png'.format(args.save_path,args.dataset))
