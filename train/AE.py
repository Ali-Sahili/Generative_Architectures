import torch
from torch import nn
from torch.autograd import Variable

import numpy as np

import os
path = os.getcwd()
os.chdir("../")

from models.AE import autoencoder

from Param import nb_epochs, lr_AE, device

os.chdir(path)

def fit_AE(train_data):

    model = autoencoder().to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_AE, weight_decay=1e-5)

    for epoch in range(nb_epochs):
        total_loss = 0
        for i in range(train_data.shape[0]):
            
            img = Variable(train_data[i])
            average = torch.mean(img, dim=1)
            sigma = torch.var(img, dim=1)

            img = img.to(device)

            average = average.cpu()
            sigma = sigma.cpu()
            label = torch.from_numpy(np.array([average.detach().numpy(),sigma.detach().numpy()]).T).float()
            label = label.to(device)
            
            output, encod_out = model(img)
            #encod_out = encod_out.view(average.shape[0])

            #loss = 0.1*criterion(output, img) + 0.8*criterion(encod_out[:,0], average.to(device)) + 0.1*criterion(encod_out[:,1], sigma.to(device))

            # Used to reconstruct the average
            #loss = criterion(encod_out, average.to(device))

            # Used to recontruct image
            loss = criterion(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        total_loss = total_loss/train_data.shape[0]
        #print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, nb_epochs, loss.item()))
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, nb_epochs, total_loss))

    return model


















