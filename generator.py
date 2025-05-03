import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import crop

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256 * 4 * 4)
        
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

        self.no1 = nn.BatchNorm2d(32)
        self.no2 = nn.BatchNorm2d(32)
        
    def forward(self, z):

        x = F.relu(self.fc1(z))
        x = x.view(-1, 256, 4, 4)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        x = x.view(-1, 1, 32, 32)
        x = F.tanh(x)
        x = crop(x, 2, 2, 28, 28)
        return x

