import argparse
import pathlib
from torchvision.utils import save_image
import model
import torch.utils
import train
import data_loader as load
import train as train_model
import numpy as np
import torch.distributions as tdist

testloader = load.load_mnist_test()
data2 = iter(testloader)


def generate_digit(no_datapoints, grid_size, digit):
    images2, labels2 = data2.next()
    images2 = images2[0:no_datapoints]
    labels2 = labels2[0:no_datapoints]
    mean, logvar = model_mnist.encoder(images2)
    var = logvar.mul(0.5).exp()
    y = labels2.detach().numpy()
    ind = np.where(y == digit)
    mean_average = torch.mean(mean[ind], 0)
    std_average = torch.sqrt(torch.mean(var[ind], 0))
    dist = tdist.Normal(mean_average, std_average)
    t = tuple([grid_size * grid_size]) + tuple(mean_average.size())
    z2 = torch.zeros(t)

    for i in range(z2.shape[0]):
        z2[i] = dist.sample()
    print(z2.shape)
    x_output = model_mnist.decoder(z2)

    figure2 = torch.from_numpy(train_model.display_grid(grid_size=grid_size, digit_size=28, images=x_output)).float()
    save_image(figure2, transit_image_directory + 'digit_generated.png')


def display_transit():
    model_mnist.eval()
    model_mnist.to('cpu')
    images1, labels1 = data.next()
    y1 = labels1.detach().numpy()
    x1 = images1[0]
    i = 0
    while y1[i] == y1[0]:
        i += 1
    x2 = images1[i]
    x1 = x1.view(1, 1, 28, 28)
    x2 = x2.view(1, 1, 28, 28)
    mean1, logvar1 = model_mnist.encoder(x1)
    mean2, logvar2 = model_mnist.encoder(x2)
    z1 = model_mnist.reparameterize(mean1, logvar1)
    z2 = model_mnist.reparameterize(mean2, logvar2)

    grid_size1 = 15
    z_transit = torch.zeros([grid_size1 * grid_size1, z_dim])

    for i, _ in enumerate(z_transit):
        z_transit[i] = z1 + ((z2 - z1) / (grid_size1 * grid_size1)) * i

    img = model_mnist.decoder(z_transit)
    figure1 = torch.from_numpy(train_model.display_grid(grid_size=15, digit_size=28, images=img)).float()
    save_image(figure1, transit_image_directory + 'digit_transit.png')


dataloader = load.load_mnist(50)
data = iter(dataloader)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="the dataset to generate from (Only MNIST currently)", default='MNIST')
parser.add_argument("--model_path", help="the path to the pre-trained model",
                    default='./experiments/mnist/training_checkpoints/checkpoint10.pth')
parser.add_argument("--grid_size", help="the size of the grid to make (a grid_size*grid_size grid will be made )",
                    default='8')
parser.add_argument("--save_path", help="the path to save image at ", default='./experiments/generated_images/')
parser.add_argument("--z_dims", help="the size of the latent space ", default='20')
parser.add_argument("--grid_size2", help="the grid size  of the digit generating image ", default='8')
parser.add_argument("--no_datapoints", help="the number of  test data-points used for estimating mean and var for "
                                            "generation ", default='1000')
parser.add_argument("--digit", help="the digit to generate", default='9')

args = parser.parse_args()

grid_size = int(args.grid_size)
z_dim = int(args.z_dims)
digit = int(args.digit)
no_datapoints = int(args.no_datapoints)
grid_size2 = int(args.grid_size2)

pathlib.Path(args.save_path).mkdir(parents=True, exist_ok=True)

model_mnist = model.make_model(z_dim)
# state_dict = torch.load('./experiments/mnist/training_checkpoints/checkpoint6.pth',
#                         map_location=lambda storage, loc: storage)
# model.load_state_dict(state_dict)

state_dict = torch.load(args.model_path, map_location=lambda storage, loc: storage)
model_mnist.load_state_dict(state_dict)

z = torch.zeros(grid_size * grid_size, z_dim)
for i in range(z.shape[0]):
    z[i] = torch.randn([1, z_dim])

x_out = model_mnist.decoder(z)
figure = torch.from_numpy(train.display_grid(grid_size=grid_size, digit_size=28, images=x_out)).float()
save_image(figure, args.save_path + '/generated_image.png')
transit_image_directory = args.save_path + '/digit_transit/'
pathlib.Path(transit_image_directory).mkdir(parents=True, exist_ok=True)

display_transit()
generate_digit(no_datapoints=no_datapoints, grid_size=grid_size, digit=digit)
