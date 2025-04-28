import torch
from torch import nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, k=1):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1) 
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.fc = nn.Linear(64 * 7 * 7, k)
        self.d1 = nn.Dropout(p=0.2)
        

    def forward(self, img):
        x = F.leaky_relu(self.conv1(img))
        x = self.d1(x)

        x = F.leaky_relu(self.conv2(x))

        x = x.view(x.size(0), -1)

        validity = torch.sigmoid(self.fc(x))
        
        return validity
