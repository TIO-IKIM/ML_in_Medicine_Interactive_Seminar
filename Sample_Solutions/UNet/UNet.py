# Author: tsimo123

import torch
import torch.nn as nn


class Conv2d_Block(nn.Module):

    """
    A block that contains two Convolutions and an activation function.
    It gets used in the Encoder to increase the amount of channel and in the Decoder to decrease the amount of channels
    """

    def __init__(self, input_channel, output_channel):

        super().__init__()

        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding = 1)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding = 1)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x

class Encoder(nn.Module):

    """
    This class combines multiple Conv2d_Block's into the Encoder. It does this by saving the output after the Conv2d_Block and passing the output afterwards trough a Max pooling layer.
    """

    def __init__(self, channel = (1,64,128,256,512)): 

        """
        The variable 'channel' describes how the channel dimension evolves in the Encoder.
        (1,64,[...]) means that the first Conv2d_Block takes the input channel (image has only one colourchannel) and creates 64 channels.
        ([...],64,128,[...]) means that the 64 channels become 128 in the next Conv2d_Block.
        The max depth here is 512.
        """
        super().__init__()

        # creates all the Conv2d_Block in one list
        self.encoder_blocks = nn.ModuleList([Conv2d_Block(channel[i], channel[i+1]) for i in range(len(channel)-1)])
        self.pooling2d = nn.MaxPool2d(2)

    def forward(self, x):

        saved_images = []
        for conv_block in self.encoder_blocks:
            x = conv_block(x)           # Applies the Conv2d_Block.
            saved_images.append(x)      # Saves the output of the Conv2d_Block.
            x = self.pooling2d(x)       # Applies the pooling layer.

        return saved_images

class Decoder(nn.Module):

    """
    This class combines multiple Conv2d_Block's into the Decoder.
    It does this by applying a upconvolution than it takes the output and combines it with the saved image at the end it applies the Conv2d_Block.
    """ 

    def __init__(self, channel=(512, 256, 128, 64)):
        
        super().__init__()

        # The variable 'channel' describes how the channel dimension evolves in the Decoder. It works like the Encoder class.
        self.channel = channel
        # Creates all the upconvolutions in one list.
        self.upconvolution = nn.ModuleList([nn.ConvTranspose2d(channel[i],channel[i+1], 2, 2) for i in range(len(channel)-1)])
        # Creates all the Conv2d_Block in one list.
        self.decoder_blocks = nn.ModuleList([Conv2d_Block(channel[i],channel[i+1]) for i in range(len(channel)-1)])

    def forward(self, x, saved_images):

        for i in range(len(self.channel)-1):
            x = self.upconvolution[i](x)                # Applies the upconvolution.
            saved_image = saved_images[i]               # Loads the saved image.
            x = torch.cat([x, saved_image], dim = 1)    # Combines the loaded image with the output of the upconvolution
            x = self.decoder_blocks[i](x)               # Applies the Conv2d_Block.
        
        return x

class UNet(torch.nn.Module):

    """
    This class combines every submodule into thecomplete UNet.
    """

    def __init__(self, encoder_channel=(1,64,128,256,512), decoder_channel=(512, 256, 128, 64), num_class=3):
        
        super().__init__()

        self.encoder     = Encoder(encoder_channel)                         
        self.decoder     = Decoder(decoder_channel)
        self.head        = nn.Conv2d(decoder_channel[-1], num_class,1) 
        # This head-convolution adjusts the number of output channel from the decoder to the number of classes. 

    def forward(self, x):

        saved_images = self.encoder(x)                                          
        # Applies the Encoder.
        
        out      = self.decoder(saved_images[::-1][0], saved_images[::-1][1:])
        # Applies the Decoder. 
        # saved_images[::-1][0] is the last saved images basically the output  of the Encoder.
        # saved_images[::-1][1:] is the saved images in reverse without the last image.

        out      = self.head(out)
        # Applies the head-convolution. 

        return out
    
if __name__ == "__main__":

    # Build the model and count its parameters.
    model = UNet()
    numel_list = [p.numel() for p in model.parameters()]
    print(sum(numel_list))