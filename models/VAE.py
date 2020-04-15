import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from Param import patch_size, device

class Variational_AE(nn.Module):
    def __init__(self):
        super(Variational_AE, self).__init__()

        self.fc1 = nn.Linear(patch_size*patch_size, 400)
        self.bn1 = nn.BatchNorm1d(400)

        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.bn2 = nn.BatchNorm1d(20)

        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, patch_size*patch_size)
        self.bn3 = nn.BatchNorm1d(patch_size*patch_size)

    def encode(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        h1 = F.relu(x)
        return self.bn2(self.fc21(h1)), self.bn2(self.fc22(h1))

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()

        eps = torch.FloatTensor(std.size()).normal_().to(device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.fc3(z)
        z = self.bn1(z)
        h3 = F.relu(z)
        return F.sigmoid(self.bn3(self.fc4(h3)))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

