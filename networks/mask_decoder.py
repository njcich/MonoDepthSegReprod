
import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from utils.torch_utils import *


class MaskDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_output_channels=5, use_skips=True):
        super(MaskDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.convs = OrderedDict()
        
        # TODO: Change last channel to be K masks
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        
        self.convs[("maskconv", 0)] = Conv3x3(self.num_ch_dec[0], self.num_output_channels)
        
        self.softmax = nn.Softmax(dim=1)

        self.decoder = nn.ModuleList(list(self.convs.values()))        
    
    
    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            
        if i == 0 :
            # needs to be fed through the masking process not the sigmoid 
            masks = self.convs[("maskconv", i)](x)
            
            for j in range(self.num_output_channels):
                masks[:, j, :, :] *= j + 1
            
            masks = self.softmax(masks)
            self.outputs[("masks", i)] = masks
                

        # Shape (Batch, num_masks, img_width, img_height)
        # torch.Size([12, 5, 192, 640])
        


        return self.outputs
