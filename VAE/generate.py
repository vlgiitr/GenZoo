import argparse
import pathlib
from torchvision.utils import save_image
import model
import torch.utils
import train

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="the dataset to generate from (Only MNIST currently)", default='MNIST')
parser.add_argument("--model_path", help="the path to the pretrained model",
                    default='./experiments/mnist/training_checkpoints/checkpoint6.pth')
parser.add_argument("--grid_size", help="the size of the grid to make (a grid_size*grid_size grid will be made )",
                    default='8')
parser.add_argument("--save_path", help="the path to save image at ", default='./experiments/user_generated_images/')
args = parser.parse_args()

grid_size = int(args.grid_size)
pathlib.Path(args.save_path).mkdir(parents=True, exist_ok=True)

model_mnist = model.make_model()
state_dict = torch.load(args.model_path)
model_mnist.load_state_dict(state_dict)


z = torch.zeros(grid_size*grid_size, 20)
for i in range(z.shape[0]):
    z[i] = torch.randn([1, 20])

x_out = model_mnist.decoder(z)
figure = torch.from_numpy(train.display_grid(grid_size=grid_size, digit_size=28, images=x_out)).float()
save_image(figure, args.save_path + 'user_generated_image.png')


