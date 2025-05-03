import torch
from torch import nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, k=1):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1) 
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)
        self.fc = nn.Linear(2 * 2 * 256, 1)
        # self.d1 = nn.Dropout(p=0.2)

    def forward(self, img):
        x = F.pad(img, (2, 2, 2, 2))
        x = F.leaky_relu(F.layer_norm(self.conv1(x), [32, 16, 16]), negative_slope=0.02)
        x = F.leaky_relu(F.layer_norm(self.conv2(x), [ 64, 8, 8]), negative_slope=0.02)
        x = F.leaky_relu(F.layer_norm(self.conv3(x), [128, 4, 4]), negative_slope=0.02)
        x = F.leaky_relu(F.layer_norm(self.conv4(x), [256, 2, 2]), negative_slope=0.02)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
