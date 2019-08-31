from torchvision.datasets import  mnist,cifar
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class DataLoad():

    def __init__(self):
        pass


    def load_data_mnist(self,batch_size=128):
        '''
        Returns a nested structure of tensors based on MNIST database.
        Will be divided into (60000/batch_size) batches of (batch_size) each.
        '''
        mnist_data = mnist.MNIST(root='./data/mnist',train=True,download=True,transform=transforms.Compose(
                                                 [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]))
        mnist_loader = DataLoader(mnist_data,batch_size=batch_size,shuffle=True)
        return mnist_loader

    def load_data_cifar10(self,batch_size=128):
        '''
        Returns a nested structure of tensors based on CIFAR10 database.
        Will be divided into (60000/batch_size) batches of (batch_size) each.
        '''
        cifar_data = cifar.CIFAR10(root='./data/cifar10',train=True,download=True,transform=transforms.Compose(
                                                 [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]))
        cifar_loader = DataLoader(cifar_data,batch_size=batch_size,shuffle=True)
        return cifar_loader
