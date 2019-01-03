import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary

class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()

        self.fc1 = nn.Linear(784,400)
        self.fc1_bn = nn.BatchNorm1d(400)
        self.fc2_1 = nn.Linear(400,20)
        self.fc2_2 = nn.Linear(400,20)
        self.fc3 = nn.Linear(20,400)
        self.fc3_bn = nn.BatchNorm1d(400)
        self.fc4 = nn.Linear(400,784)

    def encoder(self,x):
        h1 = self.fc1_bn(F.relu(self.fc1(x)))
        return self.fc2_1(h1),self.fc2_2(h1)

    def reparameterizer(self,mu, logvar):
        standard_deviation = torch.exp(logvar/2)
        eps = torch.randn_like(standard_deviation)
        return mu + standard_deviation * eps

    def decoder(self,x):
        h3 = self.fc3_bn(F.relu(self.fc3(x)))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encoder(x)
        result = self.decoder(self.reparameterizer(mu, logvar))
        return result,mu,logvar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE().to(device)
summary(model,(784,))