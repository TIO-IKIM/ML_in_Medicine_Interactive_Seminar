# Author: Prometheus9920

import torch
import torch.nn as nn
import torch.nn.functional as F

class Alexnet(nn.Module):

    def __init__(self):

        super().__init__()

        # Define all constituent modules
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(3,3),padding=1)
        self.lrp1=nn.LocalResponseNorm(4, alpha=0.0001, beta=0.75, k=1.0)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=(3,3),padding=1)
        self.lrp2=nn.LocalResponseNorm(4, alpha=0.0001, beta=0.75, k=1.0)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=(3,3),padding=1)
        self.lrp3=nn.LocalResponseNorm(4, alpha=0.0001, beta=0.75, k=1.0)
        self.dr1=nn.Dropout(p=0.5, inplace=False)
        self.fc1 = nn.Linear(8 * 32 * 32, 64)
        self.dr2=nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(64,32)
        self.dr3=nn.Dropout(p=0.5, inplace=False)
        self.fc3 = nn.Linear(32, 3)
        
 
    def forward(self, x):

        # Chain all constituent modules
        out = F.max_pool2d(self.lrp1(F.relu(self.conv1(x))),kernel_size=(2,2),stride= 2,padding=0)
        out = F.max_pool2d(self.lrp2(F.relu(self.conv2(out))),kernel_size=(2,2),stride= 2,padding=0)
        out = F.max_pool2d(self.lrp3(F.relu(self.conv2(out))),kernel_size=(2,2),stride= 2,padding=0)
        out = out.view(-1, 8 * 32 * 32)
        out = self.dr1(out)
        out = F.relu(self.fc1(out))
        out = self.dr2(out)
        out = F.relu(self.fc2(out))
        out = self.dr3(out)
        out = self.fc3(out)

        return out
    
if __name__ == "__main__":

    # Build the model and count its parameters.
    model = Alexnet()
    numel_list = [p.numel() for p in model.parameters()]
    print(sum(numel_list))