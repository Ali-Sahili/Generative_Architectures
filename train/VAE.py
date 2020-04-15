import torch
from torch.autograd import Variable

import os
path = os.getcwd()
os.chdir("../")

from models.VAE import Variational_AE

from Param import nb_epochs, lr_AE, device
from Loss_functions import loss_function

os.chdir(path)


def fit_VAE(train_data):

    model = Variational_AE().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_AE)

    for epoch in range(nb_epochs):
        model.train()
        train_loss = 0
        
        for i in range(train_data.shape[0]):
            img = Variable(train_data[i]).to(device)

        optimizer.zero_grad()

        recon_batch, mu, logvar = model(img)

        loss = loss_function(recon_batch, img, mu, logvar)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / train_data.shape[0]))

    return model
