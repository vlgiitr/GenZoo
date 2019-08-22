# Variational Auto Encoder (VAE) in Pytorch
The repository consists of a variational autoencoder implemented in pytorch and trained on MNIST dataset . It can be used to generate images and also for generating cool T-SNE visulaisations of the latent space .

## VAE : Overview

Variational autoencoders at first glance seems like another autoencoder . An autoencoder basically consists of an encoder and a decoder . **The encoder converts the input into another dimension space , generally of a smaller size**  and then tries to reconstruct the input from this representation . This kind of forces the network to filter out the not so useful features and only stores useful features .  So this is sometimes used to get a lower dimension representation of our data .

<img src='readme_images/autoencoder.png' style="max-width:100%">


Now whats so special about Variational autoencoders .

Well this is not a tutorial for VAE so let's just get an overview .

Now VAE is a generative model ..meaning it can be used to generate new data . Now why can't we use a standard autoencoder to do this . The problem **with the standard auto-encoder is that the latent space repesentation of the data follows some very complext distribution which is not known to us** . So we can't sample new latent variables from that distribution and decode them into something that looks like an image .

So that's were **VAE are different ..they constraint the latent space representation to be of that of a unit gaussian** which we can easily sample from and use to create new samples . 

Now this is done using ..well a lot of complicated maths ..something called variational inference . I guess its easier to explain it using the loss function . 

**Loss = Ez∼Q(z|x)[logP(x|z)]−KL[Q(z|x)||P(z)]**

<img src='readme_images/vae.png' style="max-width:100%">

The first term is basically maximising the likelihood of the input data and is simply said the reconstruction loss . The second term is a KL divergence loss and it measures the similarity of Q(z|x) and P(z) . P(z) is what the distribution of the latent variables should be (ie . unit gaussian) and Q(z|x) is our approximator of P(z) using the encoder neural network ( Its also a gaussian but with mean and variance output by the encoder)



So basically the loss has two opposing functions ..the reconstruction loss which tries to recreate the input as such not caring about the latent variable distribution and the KL divergence term which forces the distribution to be gaussian .

## Architecture

<img src='readme_images/VAE_architecture.png' style="max-width:100%">

The architecture is basically divided into two parts  an encoder and a decoder .
The encoder first has a bunch of convultional layers with LeakyRelu activation function and max pooling and batchnorm . The last conv layer also has dropout. Then there are a bunch of Fully connected layers with leakyRelu activation and dropout . Finally a fully connected layer gives us the mean and logvar respectively.

Then we sample from the distribution using the reparameterisation trick . With z = (std*eps)+mean , where eps = N (0,I) .

The decoder consists of a bunch of fully conncected layers followed by Transpose Convolutional layers and finally a sigmoid function which gives the output images . 

## Training images
Image from 0th epoch  

<img src='readme_images/trainiing_images/img_from_epoch0.png' style="max-width:100%">

Image from 20th epoch  

<img src='readme_images/trainiing_images/img_from_epoch20.png' style="max-width:100%">

Image from 40th epoch  

<img src='readme_images/trainiing_images/img_from_epoch40.png' style="max-width:100%">

Image from 60th epoch  

<img src='readme_images/trainiing_images/img_from_epoch60.png' style="max-width:100%">

Image from 80th epoch  

<img src='readme_images/trainiing_images/img_from_epoch80.png' style="max-width:100%">

Image from 95th epoch  
<img src='readme_images/trainiing_images/img_from_epoch95.png' style="max-width:100%">

## T-sne Visualization

After 100 epochs the t-sne visualisation is     

<img src='readme_images/t_sne_visualization.png' style="max-width:100%">

## Image generated from random gaussian input 

The following image was generated after passing a randomly sampled z from the unit gaussian and then passed through the decoder       

<img src='readme_images/user_generated_image.png' style="max-width:100%">

## Smooth transition between two digits

The following shows images formed when the latent variable z of one image was uniformly changed into that of another image and was passed through the decoder 

<img src='readme_images/digit_transit.png' style="max-width:100%">
