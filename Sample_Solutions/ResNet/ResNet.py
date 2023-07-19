# Author: obakumenko

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlockBottleneck(nn.Module):

    """
    Constructs a block [conv -> batch_norm -> activation]*3, which we will stack in the Resnet.
    Input:      int: n_chans_in
                int: n_chans_between
                int: n_chans_out
                boolean: downsample = False, set True if first block
                int: stride = 1, set 2 if want to downsample
    Output:     nn.Sequential() block
    """

    def __init__(self, n_chans_in,n_chans_between,n_chans_out, downsample = False, stride = 1):

        super().__init__()

        self.conv1 = nn.Conv2d(n_chans_in, n_chans_between, kernel_size=1, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(num_features=n_chans_between)
        self.relu = torch.nn.ReLU()
        self.conv2 = nn.Conv2d(n_chans_between, n_chans_between, kernel_size=3, stride= stride, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(num_features=n_chans_between)
        self.relu = torch.nn.ReLU()
        self.conv3 = nn.Conv2d(n_chans_between, n_chans_out, kernel_size=1, padding=0, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(num_features=n_chans_out)
        self.relu = torch.nn.ReLU()

        # Manual initialization to kaiming normal distributed values for BatchNorm parameters
        torch.nn.init.kaiming_normal_(self.conv1.weight,
                                      nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.conv2.weight,
                                      nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.conv3.weight,
                                      nonlinearity='relu')

        torch.nn.init.constant_(self.batch_norm1.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm1.bias)

        torch.nn.init.constant_(self.batch_norm2.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm2.bias)

        torch.nn.init.constant_(self.batch_norm3.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm3.bias)

        # If the output dimension is different from the input dimension, 
        # downsample the skip connection using a 1x1 convolution.
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(n_chans_in, n_chans_out, kernel_size=1, padding=0, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=n_chans_out),
                nn.ReLU()
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):

        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.batch_norm3(out)
        out = self.relu(out)

        return out + self.downsample(x)

class ResNet50(torch.nn.Module):

    """
    Construct a ResNet50 from ResBlockBottleneck modules.
    The output to three classes is hardcoded for the LiTS task.

    For the forward method, we have
    Input:    Tensor: [Batch, 1, Height, Width]
    Output:   Tensor: [Batch, 3]
    """

    def __init__(self):

        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=64)
        self.relu = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Now we stack residual blocks.
        # Like in the paper, we want 3, 4, 6 and 3 blocks for the ResNet-50,
        # so we stack 1 first block and 2 regular blocks, then 1+3, 1+5, and 1+2 again.
        self.resblocks2 = nn.Sequential(
            ResBlockBottleneck(n_chans_in=64, n_chans_between=64, n_chans_out=256, downsample=True),
            *(2 * [ResBlockBottleneck(n_chans_in=256, n_chans_between=64, n_chans_out=256)]))
        self.resblocks3 = nn.Sequential(
            ResBlockBottleneck(n_chans_in=256, n_chans_between=128, n_chans_out=512, downsample=True, stride=2),
            *(3 * [ResBlockBottleneck(n_chans_in=512, n_chans_between=128, n_chans_out=512)]))
        self.resblocks4 = nn.Sequential(
            ResBlockBottleneck(n_chans_in=512, n_chans_between=256, n_chans_out=1024, downsample=True, stride=2),
            *(5 * [ResBlockBottleneck(n_chans_in=1024, n_chans_between=256, n_chans_out=1024)]))
        self.resblocks5 = nn.Sequential(
            ResBlockBottleneck(n_chans_in=1024, n_chans_between=512, n_chans_out=2048, downsample=True, stride=2),
            *(2 * [ResBlockBottleneck(n_chans_in=2048, n_chans_between=512, n_chans_out=2048)]))
        self.avgpool6 = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(in_features=2048, out_features=3, bias=True)

    def forward(self, x):

        out_1 = self.conv1(x)
        out_1 = self.batch_norm1(out_1)
        out_1 = self.relu(out_1)
        out_1 = self.pool2(out_1)
        out_2 = self.resblocks2(out_1)
        out_3 = self.resblocks3(out_2)
        out_4 = self.resblocks4(out_3)
        out_5 = self.resblocks5(out_4)
        out_6 = self.avgpool6(out_5)
        out_6 = self.fc(torch.flatten(out_6, start_dim=1))

        return out_6

if __name__ == "__main__":

    # Build the model and count its parameters.
    model = ResNet50()
    numel_list = [p.numel() for p in model.parameters()]
    print(sum(numel_list))