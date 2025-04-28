import torch
from torch import nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.fc1 = nn.Linear(100, 64 * 7 * 7)
        
        self.conv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

        self.d1 = nn.Dropout(p=0.2)

        
    def forward(self, z):

        x = F.leaky_relu(self.fc1(z))
        x = x.view(-1, 64, 7, 7)

        
        x = F.leaky_relu(self.conv1(x))

        x = self.d1(x)

        x = F.leaky_relu(self.conv2(x))
        x = x.view(-1, 1, 28, 28)
        x = torch.sigmoid(x)
        return x
