import torch
import torch.nn as nn
import torch.nn.functional as nnf
from typing import Union

# This is a simplified and symmetrical U-Net, not the original U-Net.
# The code is inspired by https://github.com/milesial/Pytorch-UNet,
# although our implementation comes with many simplifications and a few changes.

class BasicBlock(nn.Module):

    """
    This is the basic building block of the U-Net, performing Convolution -> BatchNorm -> ReLU twice.
    """
    
    def __init__(self, in_channels: int, out_channels: int):

        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = "same", bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = "same", bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x: torch.Tensor):

        return self.block(x)

class DownBlock(nn.Module):

    """
    This is the building block of the downward (encoder) path of the U-Net.
    Each downward block performs the operation of one basic block and uses MaxPool2d for downsampling the image to a smaller size.
    """

    def __init__(self, in_channels: int, out_channels: int):

        super(DownBlock, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            BasicBlock(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor):

        x = self.down(x)

        return x

class UpBlock(nn.Module):

    """
    This is the building block of the upward (decoder) path of the U-Net.
    Each upward block performs the operation of one basic block and uses ConvTranspose2d for upsampling the image to a larger size.
    """

    def __init__(self, in_channels: int, out_channels: int):

        super(UpBlock, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2)
        self.conv = BasicBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, r: torch.Tensor):

        """
        The tensor x is our forwarded tensor, the tensor r is a residual from before,
        the side connection going from an encoder to a decoder part, skipping the rest of the U-Net.
        """

        x = self.up(x)
        x = torch.cat([r, x], dim = 1)
        x = self.conv(x)

        return x

class FinalConv(nn.Module):

    """
    This is the final convolution to map from the end of our U-Net to the number of classes we actually have.
    """

    def __init__(self, in_channels: int, out_channels: int):

        super(FinalConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):

        x = self.conv(x)

        return x

class Example_UNet(nn.Module):

    def __init__(self, in_channels: int, out_classes: int):

        super(Example_UNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_classes = out_classes

        self.in_conv = BasicBlock(in_channels, 8)
        self.down1 = DownBlock(8, 16)
        self.down2 = DownBlock(16, 32)
        self.down3 = DownBlock(32, 64)
        self.down4 = DownBlock(64, 128)
        self.up4 = UpBlock(128, 64)
        self.up3 = UpBlock(64, 32)
        self.up2 = UpBlock(32, 16)
        self.up1 = UpBlock(16, 8)
        self.out_conv = FinalConv(8, out_classes)

    def forward(self, x):

        r1 = self.in_conv(x) # Along every step of the way, we save our residuals (r1, r2, ...)
        r2 = self.down1(r1)
        r3 = self.down2(r2)
        r4 = self.down3(r3)
        r5 = self.down4(r4)
        x = self.up4(r5, r4) # And now, we add each residual on top of the current tensor
        x = self.up3(x, r3)
        x = self.up2(x, r2)
        x = self.up1(x, r1)
        x = self.out_conv(x)

        return x # Logits, NOT (pseudo-)probabilities!