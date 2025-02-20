#========================= Python imports ========================
from __future__ import annotations
from typing import Tuple, Optional

#========================= Lib imports ========================

import torch
from torch.utils.data import Dataset



#=================================================================
#=================================================================
#=================================================================


class SimpleDataset(Dataset):
         
    def __init__(self, inputLoader):
       
        self._i = 0   
        self._curIndex = -1
                
        self.inputLoader = inputLoader
        
    #=================================================================

    def getSize(self) -> int:
        return self.inputLoader.getSize()
       
    #=================================================================
   
    def getProgress(self):
        return self._i / len(self)

    #=================================================================

    def __len__(self):
        return self.getSize()

    #=================================================================

    def __getitem__(self, index):
        
        self._i += 1
        self._curIndex = index
       
        res = self.inputLoader.getData(index)
                                    
        return res
