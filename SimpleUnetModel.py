#========================= Python imports ========================
from __future__ import annotations
from typing import Optional, Tuple, Union

from typing import Tuple

#========================= Lib imports ========================
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

#========================= My imports ========================

from UpSample2D import UpSample2d

#======================================================================================
#======================================================================================
#======================================================================================


#https://amaarora.github.io/posts/2020-09-13-unet.html
#https://github.com/usuyama/pytorch-unet
#https://github.com/jvanvugt/pytorch-unet/blob/master/README.md

#https://medium.com/@arvindwaskarthik/pedestrian-segmentation-a-study-of-influence-of-parameters-and-datasets-with-unet-716162bac05f
#https://github.com/groupprojectdelft/Segmentation-using-UNet-Architecture/


#======================================================================================
#======================================================================================
#======================================================================================



class SimpleUNetBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 use_padding: bool=False,
                 dropout: Optional[float]=None):

        nn.Module.__init__(self)

        padding = 0
        padding_mode="zeros"
        if (use_padding):
            padding = 1
            padding_mode="replicate"

        self.conv1 = nn.Conv2d(in_ch, 
                               out_ch, 
                               kernel_size=3, 
                               padding=padding, 
                               padding_mode=padding_mode)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, 
                               out_ch, 
                               kernel_size=3, 
                               padding=padding, 
                               padding_mode=padding_mode)
        if (dropout is not None):
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.conv2(x)

#======================================================================================
#======================================================================================
#======================================================================================


class Encoder(nn.Module):
    def __init__(self, 
                 inputChannels: int,
                 chs=(64,128,256,512,1024),
                 use_padding: bool=False):

        nn.Module.__init__(self)

        self.enc_blocks = nn.ModuleList([SimpleUNetBlock(inputChannels, chs[0], use_padding)])
        for i in range(0, len(chs) - 1):
            self.enc_blocks.append(SimpleUNetBlock(chs[i], chs[i+1], use_padding))

        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

#======================================================================================
#======================================================================================
#======================================================================================


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64), use_padding: bool=False):
        nn.Module.__init__(self)

        self.chs = chs
        
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for i in range(0, len(chs) - 1):

            self.upconvs.append(UpSample2d(chs[i], chs[i + 1], scaleFactor=2))
            self.dec_blocks.append(SimpleUNetBlock(chs[i], chs[i+1], use_padding))
            
            
    def crop(self, img: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _, _, h, w = target.shape
        _, _, Hx, Wx = img.shape

        if ((Hx == h) and (Wx == w)):
            #no need to crop
            return img

        #version 1
        #diffY = h - Hx
        #diffX = w - Wx
        #
        #c = F.pad(x, [diffX // 2, diffX - diffX // 2,
        #              diffY // 2, diffY - diffY // 2])

        #version 2
        #diffY = (Hx - h)
        #diffX = (Wx - w)
        #startY = diffY // 2
        #startX = diffX // 2
        #endY = diffY - startY
        #endX = diffX - startX
        #c = x[:, :, startY:-endY, startX:-endX]

        c = transforms.CenterCrop([h, w])(img)
        return c
            

    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x
    
    

#======================================================================================
#======================================================================================
#======================================================================================


class SimpleUNetModel(nn.Module):
    
    def __init__(self,                 
                 channels_in: int,
                 channels_out: int,
                 out_w: int,
                 out_h: int,                
                 enc_chs=(64,128,256,512,1024), 
                 dec_chs=(1024, 512, 256, 128, 64),
                 sigmoidOutput: bool = False
                 ):       
        super().__init__()
                  
        self.encoder     = Encoder(channels_in, enc_chs, True)
        self.decoder     = Decoder(dec_chs, True)
        self.head        = nn.Conv2d(dec_chs[-1], channels_out, 1)
        
        if (sigmoidOutput):
            self.outputActivation = torch.nn.Sigmoid()
        else:
            self.outputActivation = torch.nn.Identity()

        self.out_w = out_w
        self.out_h = out_h


    def forward(self, x: torch.Tensor) -> torch.Tensor:
               
        enc_ftrs = self.encoder(x)
        outDec   = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(outDec)

        needInterpolate = ((out.shape[2] != self.out_h) or 
                           (out.shape[3] != self.out_w))
            
        if (needInterpolate):
            out = F.interpolate(out, (self.out_w, self.out_h))       
        
        return self.outputActivation(out)
       

    


