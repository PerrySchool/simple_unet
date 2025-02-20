#========================= Python imports ========================
from __future__ import annotations
from typing import Tuple, Optional, Union, List, Any

import os
import random

from enum import Enum

#========================= Lib imports ========================

import torch
from torch.utils.data import DataLoader

from torchvision import transforms

import numpy as np

from PIL import Image, ImageStat

#========================= My imports ========================

from Settings import Settings

from ImageUtils import ImageUtils, ImageLoader, ToRGB, MyToTensor

from SimpleDataset import SimpleDataset

#======================================================================================
#======================================================================================
#======================================================================================

class LoaderType(Enum):
    TRAIN = 1,
    VALID = 2,
    TEST = 3,
    INFERENCE = 4

#================================================================================

class SegmentationInputLoader:
        
    def __init__(self, 
                 s: Settings,     
                 loaderType: LoaderType,
                 subsetSize: Optional[int]=None):
           
        self.sets = s

        self.channelsCount = s.channelsCount
        self.imgW = s.imgW
        self.imgH = s.imgH
        self.imgFormat = ImageUtils.ImageFormat.C_H_W

        self.loaderType = loaderType
        self.shuffleSeed = None
        
        self.trainRatio = s.trainRatio
        self.valRation = 0.0
        self.testRatio = 1.0 - self.trainRatio

        self.subsetSize = subsetSize

        self.dataRoot = s.datasetPath              
        self.fileIds = []
        self.fileNames = []        
                      
        self.transforms = self._buildTransforms()

        #mask cannot have custom transforms
        self.transformsMask = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((self.imgH, self.imgW)),
            transforms.ToTensor()])


        self.load()
      
    # ======================================================================================

    def _buildTransforms(self):

        t = []
        
        if (self.channelsCount == 1):            
            t.append(transforms.Grayscale())
        elif (self.channelsCount == 3): 
            t.append(ToRGB())

        t.append(transforms.Resize(size=(self.imgH, self.imgW), antialias=True))
               
        t.append(MyToTensor())
               
        return transforms.Compose(t)

    #======================================================================================

    def load(self):

        self.fileIds = []
        self.fileNames = []

        tmpFileNames = []
        tmpFileIds = []
        
        for root, dirs, files in os.walk(self.dataRoot, topdown=False):  
            
            fileId = os.path.basename(root)
            isCorrect = fileId.isnumeric()
            if (isCorrect is False):                
                continue

            filesFiltered = files
            
            
            tmpFileNames += [os.path.join(root, s) for s in filesFiltered]
            
            tmpFileIds += [fileId] * len(filesFiltered)
        
        #endfor

        self.fileNames, self.fileIds = self._buildSplits(tmpFileNames, tmpFileIds)
        
    #======================================================================================

    def _buildSplits(self, *inputs) -> list[list[Any]]:
        """
        Build splits of input lists           
        and get list of output arrays
        """

        totalFilesCount = len(inputs[0])

        offsetIndex = 0
        filesCount = 0
        if (self.loaderType == LoaderType.TRAIN):
            filesCount = int(self.trainRatio * totalFilesCount)
        elif (self.loaderType == LoaderType.VALID):            
            filesCount = int(self.valRatio * totalFilesCount)
            offsetIndex = int(self.trainRatio * totalFilesCount)
        else:
            if (self.testRatio is None):
                filesCount = totalFilesCount - int((self.valRatio + self.trainRatio) * totalFilesCount)
                offsetIndex = totalFilesCount - filesCount
            else:
                filesCount = int(self.testRatio * totalFilesCount)
                offsetIndex = int(self.testRatio * totalFilesCount)
          
        if (filesCount == 0):
            return [[]]

        if (self.subsetSize is not None):
            if (filesCount > self.subsetSize):
                filesCount = self.subsetSize

        #shuffle mmust have same seed for train and test
        #the train / test to contain unique files
        #if we use different seed for train / test
        #sets will contain same files
        if (self.shuffleSeed is not None):
            rnd = random.Random(self.shuffleSeed)

            c = list(zip(*inputs))
            rnd.shuffle(c)            
            inputs = list(zip(*c))
            
        
        outputCount = len(inputs)
        output = [None] * outputCount

        for i in range(0, len(output)):
            if (self.loaderType == LoaderType.TRAIN):
                output[i] = list(inputs[i][0: filesCount])
            elif (self.loaderType == LoaderType.VALID):
                output[i] = list(inputs[i][offsetIndex: offsetIndex + filesCount])
            else:
                output[i] = list(inputs[i][-filesCount:])
                       
        return output


    #======================================================================================
    
    def getSize(self) -> int:
        return len(self.fileNames)


    #======================================================================================

    def getFileNames(self, index) -> Union[str, List[str]]:

        maskName = self.fileIds[index]
        if (maskName[-1] != 'g'):
            #mask name is not already a file, but a fileId
            #create mask file name manually
            maskName = f"{self.dataRoot}/{self.fileIds[index]}.png"

        return self.fileNames[index], maskName

    #======================================================================================

    def getData(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        
        fileName, maskName = self.getFileNames(index)
        
        imgTensor = ImageUtils.loadImageAsTensor(fileName, self.imgFormat, self.transforms)              
        mask = ImageUtils.loadImageAsTensor(maskName, self.imgFormat, self.transformsMask)

        if (imgTensor is None):
            c = 3
            if (self.channelsCount == 1):
                c = 1

            imgTensor = ImageUtils.getEmptyImage(c, self.imgW, self.imgH, self.imgFormat)
            imgTensor = torch.as_tensor(imgTensor)
            mask = None #clear mask, since image failed to load

        if (mask is None):            
            mask = ImageUtils.getEmptyImage(1, self.imgW, self.imgH, self.imgFormat)
                                            
            mask = torch.as_tensor(mask)
        else:
            mask[mask > 0.5] = 1.
            mask[mask <= 0.5] = 0.

                
        return imgTensor, mask
 
    #======================================================================================

    def buildDataLoader(self, shuffle: bool=False) -> torch.utils.data.DataLoader:
        """
        Create torch DataLoader class instance
        """
        ds = SimpleDataset(self)
        return DataLoader(ds, 
                          num_workers=self.sets.numWorkers, 
                          shuffle=shuffle, 
                          batch_size=self.sets.batchSize,
                          pin_memory=True)

