# ========================= Python imports ========================
from __future__ import annotations
from typing import Type, Union, Tuple, Optional

from enum import Enum

# ========================= Lib imports ========================
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================================================
# ======================================================================================
# ======================================================================================


class UpSample2d(nn.Module):

    class SampleType(Enum):
        NEAREST_NEIGHBOR = 1,
        SUBPIXEL = 2,
        CONV_TRANPOSE = 3,
        BILINEAR = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 scaleFactor: int,
                 kernel_size: int = 1,                 
                 padding: int = 0,
                 dilation: int = 1,
                 normType: Type[nn.Module] = nn.BatchNorm2d,                 
                 modeType: SampleType = SampleType.NEAREST_NEIGHBOR,
                 dtype=None):
        super(UpSample2d, self).__init__()

        super().__init__()
                
        self.scaleFactor = scaleFactor
        self.modeType = modeType

        if (self.modeType == UpSample2d.SampleType.SUBPIXEL):                      

            # this calculated more channels and with pixel shuffle combine
            # multiple channels to form a block of pixels in s single image

            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, 
                          out_channels * scaleFactor * scaleFactor, 
                          kernel_size=kernel_size,
                          stride=1, 
                          padding=padding, 
                          dilation=dilation,
                          dtype=dtype),
                nn.PixelShuffle(scaleFactor)
            ])
            
        elif (self.modeType == UpSample2d.SampleType.CONV_TRANPOSE):
            # in this case, scale factor is not used directly
            # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html            

            # out = [(in - 1) * stride] - [2 * padding] + [dilation * (kernel_size - 1)] + outPadding + 1
            # input = 8x8
            # expected out = 16x16 = [(8 - 1) * 2] - [2 * 1] + [1 * (4 - 1)] + 0 + 1
            # (one possible combination !!!)
            #  kernel = 4
            #  stride = 2 (default: 1)
            #  padding = 1 (default: 0)
            #  dilation = 1 (default: 1)
            #  outPadding = 0 (default: 0)
            # (for example if kernel is 3, outPadding should be 1 to get the same size)
            
            self.conv = nn.ConvTranspose2d(in_channels, 
                                           out_channels, 
                                           kernel_size, 
                                           stride=self.scaleFactor, 
                                           padding=padding,
                                           dilation=dilation,
                                           output_padding=0,
                                           dtype=dtype)
            
        else:
            #https://github.com/julianstastny/VAE-ResNet18-PyTorch/blob/master/model.py            

            #this is accompanied by F.interpolate of the x input in forward

            self.conv = nn.Conv2d(in_channels, 
                              out_channels, 
                              kernel_size, 
                              stride=1, 
                              padding=padding,
                              dilation=dilation,
                              dtype=dtype)
        #endif

    # =======================================================================

    def forward(self, x):
        if (self.modeType == UpSample2d.SampleType.NEAREST_NEIGHBOR):
            x = F.interpolate(x, scale_factor=self.scaleFactor, mode='nearest')
        elif (self.modeType == UpSample2d.SampleType.BILINEAR):
            x = F.interpolate(x, scale_factor=self.scaleFactor, mode='bilinear')
            
        x = self.conv(x)
        return x