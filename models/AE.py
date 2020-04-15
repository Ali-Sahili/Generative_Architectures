from torch import nn
import torch.nn.functional as F

from Param import patch_size

latent_dim = 20

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.fc1 = nn.Linear(patch_size*patch_size, 400)
        self.fc2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, latent_dim)
 
        self.fc4 = nn.Linear(latent_dim, 20)
        self.fc5 = nn.Linear(20, 400)
        self.fc6 = nn.Linear(400, patch_size*patch_size)

    def encoder(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = self.fc3(h2)
        return h3

    def decoder(self, x):
        h1 = F.relu(self.fc4(x))
        h2 = F.relu(self.fc5(h1))
        h3 = self.fc6(h2)
        return h3

    def forward(self, x):
        y = self.encoder(x)
        x = self.decoder(y)
        return x, y






"""
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5,5), stride=1, padding=0), 
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  
            nn.Conv2d(16, 8, kernel_size=(5,5), stride=1, padding=0), 
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 1, kernel_size=(3,3), stride=2, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 8, 3, stride=2), 
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        y = self.encoder(x)
        #x = self.decoder(y)
        #return x, y
        return y

"""
