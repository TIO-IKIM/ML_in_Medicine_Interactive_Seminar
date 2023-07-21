# Author: fjonske

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Alexnet(nn.Module):

    """
    Note that the cross-GPU talk that exists in the original AlexNet is not happening in the modernized version, where splitting across GPUs is automatic. The "extra" channels that would exist on the second GPU are instead simply added to the model width.
    Note also that we forego the modern AdaptiveAveragePool and ReLUs, because those were not in the original paper. 
    """

    def __init__(self):

        super().__init__()

        # Define all constituent modules
        self.network = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2), # 3 for regular images, 1 for LiTS
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 3),
        )
 
    def forward(self, x):

        x = self.network(x)
        x = x.view(x.size()[0], -1) # This functions somewhat like flatten(), but is faster.
        x = self.head(x)

        return x
    
if __name__ == "__main__":

    # Build the model and count its parameters.
    model = Alexnet()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_parameters = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Our AlexNet: {num_parameters} parameters")

    """
    Note that this number necessarily deviates from the pytorch version and original paper version:
    The convolution layers in pytorch have the wrong channel sizes, and the original paper expects
    a hardcoded input of 227x227 pixel images, while this sample solution has a hardcoded 256x256
    expectation which the first fully connected layer relies on. The modern PyTorch implementation
    uses AdaptiveAveragePool here, which is generally recommended, unless you do archaeology like
    we are doing right now.
    """