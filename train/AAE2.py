import numpy as np
import itertools

import torch
import torch.nn as nn
from torch.autograd import Variable

import os
path = os.getcwd()
os.chdir("../")

from models.AAE2 import Encoder, Decoder, Discriminator

from Param import lr, b1, b2
from Param import X_dim, z_dim, train_batch_size
from Param import nb_epochs, device, Tensor

from Utils.utils import Covariance_Correlation

os.chdir(path)

def generate_model_AAE2(train_data):

    # Use binary cross-entropy loss
    adversarial_loss = torch.nn.BCELoss().to(device)
    #pixelwise_loss = torch.nn.L1Loss().to(device)
    pixelwise_loss = torch.nn.MSELoss().to(device)


    # Initialize generator and discriminator
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    discriminator = Discriminator().to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam( itertools.chain(encoder.parameters(), decoder.parameters()), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))


# ----------
#  Training
# ----------

    for epoch in range(nb_epochs):
        #for i in range(train_data.shape[0]):
        for X in train_data:
            #X = train_data[i]
            # Adversarial ground truths
            valid = Variable(Tensor(train_batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(train_batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(X.type(Tensor)).to(device)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            encoded_imgs = encoder(real_imgs)
            decoded_imgs = decoder(encoded_imgs)

            # Loss measures generator's ability to fool the discriminator
            g_loss_1 = 0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(decoded_imgs, real_imgs)

            # loss measures the difference between the Correlations of real and generated images

            # use the covariance in the loss
            #Cov_real, Cor_real = Covariance_Correlation(real_imgs)
            #Cov_gen, Cor_real = Covariance_Correlation(decoded_imgs)
            #g_loss_2 = torch.mean(torch.abs(Cov_gen - Cov_real))

            # use the correlation in the loss
            #Cor_desired = torch.ones((train_batch_size,train_batch_size), dtype = torch.float32)
            #Cov_gen, Cor_gen = Covariance_Correlation(decoded_imgs)

            #g_loss_2 = torch.mean(torch.abs(Cor_gen.to(device) - Cor_desired.to(device)))  


            # to estimate shifting between images
            Tensor_tmp = torch.zeros((real_imgs.shape[0],real_imgs.shape[1]), dtype = torch.float32)
            Tensor_tmp[1:] = decoded_imgs[1:]
            Tensor_tmp[real_imgs.shape[0]-1] = decoded_imgs[0]

            g_loss_2 = pixelwise_loss(Tensor_tmp.to(device), real_imgs.to(device))


            k = 0.5
            g_loss = (1-k)*g_loss_1 + k*g_loss_2

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as discriminator ground truth
            z = Variable(Tensor(np.random.normal(0, 1, (train_batch_size, z_dim)))).to(device)

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(z), valid)
            fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

        print(  "[Epoch %d/%d] [D loss: %f] [G loss: %f] [G2 loss: %f]"
                % (epoch, nb_epochs, d_loss.item(), g_loss.item(), g_loss_2.item()))

    return encoder, decoder, encoded_imgs



