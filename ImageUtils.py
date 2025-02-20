#========================= Python imports ========================
from __future__ import annotations
from typing import Tuple, Optional, Union


from enum import Enum

import math
import io
import base64

#========================= Lib imports ========================

import torch
import torch.nn.functional as F

import numpy as np

from PIL import Image, ImageFile
from torchvision import transforms
import torchvision

#========================= My imports ========================


#=================================================================
#=================================================================
#=================================================================

class ImageLoader:
    
    @staticmethod
    def loadImage(path: str) -> Optional[Image.Image]:
        try:           
            ImageFile.LOAD_TRUNCATED_IMAGES = True #allow to load corrupted images  
            img = Image.open(path)
        except Exception as err:
            return None

        return img

    @staticmethod
    def loadImageFromBase64(data: bytearray) -> Optional[Image.Image]:
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True  # allow to load corrupted images
            img = base64.b64decode(data.decode())
            img = Image.open(io.BytesIO(img))
        except Exception as err:
            return None

        return img

#=================================================================

class PILToHSV:
    """
    Convert a ``PIL Image`` from RGB to HSV. 
    This transform does not support torchscript.

    Converts a PIL Image (H x W x C).
    """
    
    def __call__(self, pic: Image.Image) -> Image.Image:       
        return pic.convert('HSV')

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

#=================================================================

class PILGetSingleChannel:
    """
    Get a single channel from PIL image
    This transform does not support torchscript.

    Converts a PIL Image (H x W x C) to (H x W x 1).
    """
    
    def __init__(self, channelIndex: int):
        self.ci = channelIndex

    def __call__(self, pic: Image.Image) -> Image.Image:       
        return pic.getchannel(self.ci)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

#=================================================================

class ToRGB:
    """
    Take input tensor and duplicate channels to
    sattisfy RGB
    """
      
    def __call__(self, pic: Union[torch.Tensor, Image.Image]) -> Union[torch.Tensor, Image.Image]:  
        if ((isinstance(pic, torch.Tensor)) or (isinstance(pic, np.ndarray))):
            #repeat method exist for tensor and ndarray

            if (len(pic.shape) == 4):
                #first dim is batch
                if (pic.shape[1] == 3):
                    return pic
                else:
                    return pic.repeat(1, 3, 1, 1)

            if (pic.shape[0] == 3):
                return pic
            else:
                return pic.repeat(3, 1, 1)
        
        else:        
            if (pic.mode == "RGB"):
                return pic
            return pic.convert('RGB')

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

#=================================================================

class MyToTensor:
        
    def __call__(self, pic: Union[torch.Tensor, Image.Image, np.ndarray]) -> torch.Tensor:
        if (isinstance(pic, torch.Tensor)):
            return pic
        return torchvision.transforms.functional.to_tensor(pic)        

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

#=================================================================

class ImageUtils:
    
    class ImageFormat(Enum):
        C_W_H = 1,
        C_H_W = 2,
        H_W_C = 3,
        W_H_C = 4


    class SequenceFormat(Enum):
        B = 1, #batch only
        B_S = 2, #batch_sequence
        S_B = 3  #sequence_batch

    #=============================================================

    def __init__(self):                
        return
    
    @staticmethod
    def disablePilInfo():
        import logging
        logging.getLogger("PIL").setLevel(logging.WARNING)
    
    # =============================================================

    @staticmethod
    def cropImage(x: torch.Tensor, w: int, h: int) -> torch.Tensor:

        _, _, Hx, Wx = x.shape
        if ((Hx == h) and (Wx == w)):
            #no need to crop
            return x
        
        #which version is fastest? todo

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

        #version 3
        c = transforms.CenterCrop([h, w])(x)

        return c

    #=============================================================

    @staticmethod
    def loadImageAsTensor(imgPath: str, 
                          imgFormat: ImageUtils.ImageFormat,
                          transform = None) -> Optional[torch.Tensor]:

        if (transform is None):
            transform = transforms.ToTensor()
        
        img = ImageLoader.loadImage(imgPath)
        if (img is None):
            return None

        return ImageUtils.convertImageToTensor(img, imgFormat, transform)

    # =============================================================

    @staticmethod
    def convertImageToTensor(img: Image.Image,
                             imgFormat: ImageUtils.ImageFormat,
                             transform=None) -> Optional[torch.Tensor]:

        imgTensor = transform(img)
        
        if (isinstance(imgTensor, torch.Tensor) is False):
            transform = transforms.ToTensor()
            imgTensor = transform(imgTensor)

        if (imgFormat == ImageUtils.ImageFormat.C_H_W):
            return imgTensor
        
        if (imgFormat == ImageUtils.ImageFormat.H_W_C):
            imgTensor = imgTensor.permute(1, 2, 0)             
        elif (imgFormat == ImageUtils.ImageFormat.W_H_C):
            imgTensor = imgTensor.permute(2, 1, 0)             
        elif (imgFormat == ImageUtils.ImageFormat.C_W_H):
            imgTensor = imgTensor.permute(0, 2, 1)             
                                                  
        return imgTensor
      
    #=============================================================

    @staticmethod
    def getEmptyImage(c: int, w: int, h: int,
                      format: ImageUtils.ImageFormat, 
                      dtype=np.float32) -> np.ndarray:

        if (format == ImageUtils.ImageFormat.C_W_H):
            x = np.zeros((c, w, h), dtype=dtype)
        elif (format == ImageUtils.ImageFormat.C_H_W):
            x = np.zeros((c, h, w), dtype=dtype)
        elif (format == ImageUtils.ImageFormat.W_H_C):
            x = np.zeros((w, h, c), dtype=dtype)
        elif (format == ImageUtils.ImageFormat.H_W_C):
            x = np.zeros((h, w, c), dtype=dtype)
        else:
            x = np.zeros((c, h, w), dtype=dtype)

        return x
    
    #=============================================================

    @staticmethod
    def getImageDimensions(t: torch.Tensor) -> Tuple[int, int, int]:
        if (len(t.shape) == 3):
            chanCount = t.shape[0]
        else:
            chanCount = 1 
            
        if (len(t.shape) == 3):
            h = t.shape[1]
        else:
            h = t.shape[0]

        if (len(t.shape) == 3):
            w = t.shape[2]
        else:
            w = t.shape[1]

        return chanCount, w, h
       
    #=============================================================

    @staticmethod
    def tensorToImage(t: Union[torch.Tensor, np.ndarray], 
                      chanCount: int = -1,
                      w: int = -1,
                      h: int = -1,
                      intervalMapping: bool = True) -> Image.Image:

        """
        torch.Tensor of size (c, h, w) or (1, c, h, w)
        """
        
        if (len(t.shape) == 4):
            #remove batch of size 1
            t = t.squeeze(0)
        
        #==================================
        # try to get dimensions automatically
        #==================================
        
        if (chanCount == -1):
            if (len(t.shape) == 3):
                chanCount = t.shape[0]
            else:
                chanCount = 1            
        
        if (h == -1):
            if (len(t.shape) == 3):
                h = t.shape[1]
            else:
                h = t.shape[0]
                
        if (w == -1):
            if (len(t.shape) == 3):
                w = t.shape[2]
            else:
                w = t.shape[1]
        #==================================

        mode = "RGB"
        if (chanCount == 1):
            mode = "L"
        elif (chanCount == 4):
            mode = "RGBA"

        if isinstance(t, torch.Tensor):
            flat = t.cpu()
        else:
            flat = t
                        
        flat = np.array(flat.flatten().tolist())

        if (intervalMapping):    
            #map to [0, 1]
            maxVal = np.max(flat)
            minVal = np.min(flat)

            diff = (maxVal - minVal)
            if (diff == 0):                
                #min and max val are same
                #check if both values are outside interval [0, 1]
                #if not, keep the value instead of mapping

                if ((minVal < 0) or (maxVal > 1)):               
                    #they are outisde interval => apply mapping
                    #this lead to a list full of 0
                    flat = (flat - minVal)
                else:
                    flat = np.array(flat)
            else:
                flat = (flat - minVal) / diff
        else:
            #clamp to [0, 1], no interval mapping
            flat = np.clip(flat, a_min=0, a_max=1)

        flat = 255 * flat 

        flatImg = []

        if (chanCount == 1):
            for i in range(0, len(flat)):
                v = flat[i]
                #if (math.isnan(v)):
                #    print("?")
                flatImg.append((int)(v))

        elif (chanCount == 3):            
            for i in range(0, w * h):                           
                r = flat[i]
                g = flat[i + 1 * w * h]
                b = flat[i + 2 * w * h]
                flatImg.append(((int)(r), (int)(g), (int)(b)))

        elif (chanCount == 4):
            for i in range(0, w * h):                           
                r = flat[i]
                g = flat[i + 1 * w * h]
                b = flat[i + 2 * w * h]
                a = flat[i + 3 * w * h]
                flatImg.append(((int)(r), (int)(g), (int)(b), (int)(a)))

                    
        im = Image.new(mode, (w, h))
        im.putdata(flatImg)

        #im.save("D://im.png")

        return im

    #=============================================================

    @staticmethod
    def tensorsToImage(t: Union[list, np.ndarray, torch.Tensor], 
                      seqFromat: SequenceFormat = SequenceFormat.B_S,
                      chanCount: int = -1,
                      w: int = -1,
                      h: int = -1,
                      borderSize: int = 0,
                      backgroundValue: Image._Color = 255,
                      intervalMapping: bool = True) -> Image.Image:
        """
        Convert input t to image
        t can be:
        * torch.Tensor of size (c, h, w)
          single image
        * torch.Tensor of size (batch, c, h, w)
          images are put in one line
        * torch.Tensor of size (batch, seq, c, h, w)
          images are put in seq lines
        * list[torch.Tensor]
          list of torch.Tensors, each tensor is (c, h, w)
          images are put in one line
        * list[list[torch.Tensor]]
          [y] = batch
          [x] = sequence
          2D matrix [y][x] of torch.Tensors, each tensor is (c, h, w)          
          images in [y] are put in one line

        Each batch = one line


        Note:
        Usually, dimmensions are set to -1. 
        However, when we need to "remap" image to a different size, dimensions
        can be provided manually. This si because data are flatten to 1D array
        and it is processed based on dimensions

        Parameters
        ----------
        t:
            input data
        chanCount (int): 
            number of channels (if -1 => auto calculated)
        w (int): 
            width (if -1 => auto calculated)
        h (int): 
            height (if -1 => auto calculated)
        borderSize (int): 
            size of border between images
        backgroundValue (int):
            value of bacground (visible only if borders are used) or images are not set
        intervalMapping (bool): 
            map image interval from [min, max] to 0 - 255

        Returns
        ----------
        PIL image
        """

        if (isinstance(t, torch.Tensor)):

            if (len(t.shape) == 3):
                return ImageUtils.tensorToImage(t, chanCount, w, h, intervalMapping)
            
            elif (len(t.shape) == 4):

                #input is tensor (batch, c, h, w)
                #convert it to list of (c, h, w)
                tmpBatch = []
                for i in range(0, t.shape[0]):
                    tens = t[i]                
                    tmpBatch.append(tens)
                t = tmpBatch

            elif (len(t.shape) == 5):
                #input is tensor (batch, seq, c, h, w)
               
                if (seqFromat == ImageUtils.SequenceFormat.S_B):
                    t = t.permute(1, 0, 2, 3, 4)

                #convert it to list of list of (c, h, w)
                tmpBatch = []
                for i in range(0, t.shape[0]):
                    batch = t[i]   
                    tmpSeq = []
                    for j in range(0, batch.shape[0]):
                        tmpSeq.append(batch[j])
                    tmpBatch.append(tmpSeq)
                t = tmpBatch

        if (ImageUtils.isIndexable(t)):           
            if (ImageUtils.isIndexable(t[0])):  
                
                if (seqFromat == ImageUtils.SequenceFormat.S_B):
                    #change from S_B to B_S
                    t = list(map(list, zip(*t)))
                    
                cAuto, wAuto, hAuto = ImageUtils.getImageDimensions(t[0][0])
            else:
                cAuto, wAuto, hAuto = ImageUtils.getImageDimensions(t[0])        
        else:
            raise TypeError("input t is not a list") 

        if (chanCount == -1):
            chanCount = cAuto       
        if (w == -1):
            w = wAuto
        if (h == -1):
            h = hAuto

        mode = "RGB"
        if (chanCount == 1):
            mode = "L"
        elif (chanCount == 4):
            mode = "RGBA"


        totalW: int = w
        totalH: int = h

        if (ImageUtils.isIndexable(t)):           
            if (ImageUtils.isIndexable(t[0])):                
                maxW = 0
                for y in range(0, len(t)):                    
                    s = (w + borderSize) * len(t[y]) + borderSize 
                    if (s > maxW):
                        maxW =s

                totalW = maxW
                totalH = (h + borderSize) * len(t) + borderSize

            else:
                totalW = (w + borderSize) * len(t) + borderSize 
                totalH = h + 2 * borderSize
        #endif

        newImg = Image.new(mode, (totalW, totalH), backgroundValue)

        if (ImageUtils.isIndexable(t[0])):            
            offsetY = borderSize
            for y in range(0, len(t)):
                offsetX = borderSize
                for x in range(0, len(t[y])):
                    img = ImageUtils.tensorToImage(t[y][x], chanCount, w, h, intervalMapping)

                    newImg.paste(img, (offsetX, offsetY))
                    offsetX += w + borderSize 

                offsetY += h + borderSize
        else:            
            offsetY = borderSize            
            offsetX = borderSize
            for x in range(0, len(t)):
                img = ImageUtils.tensorToImage(t[x], chanCount, w, h, intervalMapping)

                newImg.paste(img, (offsetX, offsetY))
                offsetX += w + borderSize 

                    
        return newImg

