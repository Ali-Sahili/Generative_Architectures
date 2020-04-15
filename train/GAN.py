import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

import os
path = os.getcwd()
os.chdir("../")

from models.GAN import Generator, Discriminator

from Param import lr, b1, b2
from Param import Tensor, device

os.chdir(path)

# --------------
#  Training GAN
# --------------

def fitGAN(pp, generator, discriminator, optimizer_G, optimizer_D, adversarial_loss, latent_dim, n_epochs, nb_patches, Tensor):
    for epoch in range(n_epochs):
        for q in range(nb_patches):
            data = pp[:,q,:,:]
        

            # Adversarial ground truths
            valid = Variable(Tensor(data.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(data.size(0), 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_data = Variable(data.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (data.shape[0], latent_dim))))

            # Generate a batch of images
            gen_data = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_data), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_data), valid)
            fake_loss = adversarial_loss(discriminator(gen_data.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()


            if (epoch%10 == 0):
                print(
                      "[Epoch %d/%d] [Patch %d] [D loss: %f] [G loss: %f]"
                      % (epoch, n_epochs, q+1, d_loss.item(), g_loss.item())
                     )

    return gen_data




def generate_model(pp, latent_dim, n_epochs, nb_patches):



    """ Loss function """
    adversarial_loss = torch.nn.BCELoss().to(device)

    """ Initialize generator and discriminator """
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)


    """ Optimizers """
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    gen_data = fitGAN(pp, generator, discriminator, optimizer_G, optimizer_D, adversarial_loss, latent_dim, n_epochs, nb_patches, Tensor)

    return gen_data
