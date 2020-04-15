import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import os
path = os.getcwd()
os.chdir("../")

from Param import X_dim, N, z_dim, train_batch_size
from Param import gen_lr, reg_lr, nb_epochs, device

from models.AAE import Q_net, P_net, D_net_gauss
from Utils.utils import report_loss

os.chdir(path)

# Train procedure for one epoch
def train(P, Q, D_gauss, P_decoder, Q_encoder, Q_generator, D_gauss_solver, train_data):

    TINY = 1e-15
    # Set the networks in train mode (apply dropout when needed)
    Q.train()
    P.train()
    D_gauss.train()

    # Loop through the labeled and unlabeled dataset getting one batch of samples from each
    # The batch size has to be a divisor of the size of the dataset or it will return
    # invalid samples
    for X in train_data:

        # Load batch and normalize samples to be between 0 and 1
        X = X * 0.3081 + 0.1307
        X.resize_(train_batch_size, X_dim)
        X = Variable(X).to(device)

        # Init gradients
        P.zero_grad()
        Q.zero_grad()
        D_gauss.zero_grad()

        #----------------------
        # Reconstruction phase
        #----------------------
        z_sample = Q(X)
        X_sample = P(z_sample)
        recon_loss = F.binary_cross_entropy(X_sample + TINY, X.resize(train_batch_size, X_dim) + TINY)

        recon_loss.backward()
        P_decoder.step()
        Q_encoder.step()

        P.zero_grad()
        Q.zero_grad()
        D_gauss.zero_grad()

        #----------------------
        # Regularization phase
        #----------------------
        # Discriminator
        Q.eval()
        z_real_gauss = Variable(torch.randn(train_batch_size, z_dim) * 5.).to(device)

        z_fake_gauss = Q(X)

        D_real_gauss = D_gauss(z_real_gauss)
        D_fake_gauss = D_gauss(z_fake_gauss)

        D_loss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))

        D_loss.backward()
        D_gauss_solver.step()

        P.zero_grad()
        Q.zero_grad()
        D_gauss.zero_grad()

        # Generator
        Q.train()
        z_fake_gauss = Q(X)

        D_fake_gauss = D_gauss(z_fake_gauss)
        G_loss = -torch.mean(torch.log(D_fake_gauss + TINY))

        G_loss.backward()
        Q_generator.step()

        P.zero_grad()
        Q.zero_grad()
        D_gauss.zero_grad()

    return D_loss, G_loss, recon_loss, X_sample


def generate_model_AAE(train_data):

    torch.manual_seed(10)

    Q = Q_net().to(device)
    P = P_net().to(device)
    D_gauss = D_net_gauss().to(device)

    # Set optimizators
    P_decoder = optim.Adam(P.parameters(), lr=gen_lr)
    Q_encoder = optim.Adam(Q.parameters(), lr=gen_lr)# to have independence in the optimization 
                                                     # procedure for the encoder, we take 2 optimizers
    Q_generator = optim.Adam(Q.parameters(), lr=reg_lr)
    D_gauss_solver = optim.Adam(D_gauss.parameters(), lr=reg_lr)

    for epoch in range(nb_epochs):
        D_loss_gauss, G_loss, recon_loss, X_sample = train(P, Q, D_gauss, P_decoder, Q_encoder,
                                                 Q_generator,
                                                 D_gauss_solver,
                                                 train_data)
        if epoch % 10 == 0:
            report_loss(epoch, D_loss_gauss, G_loss, recon_loss)



    return Q, P, X_sample

