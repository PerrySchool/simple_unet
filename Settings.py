from __future__ import annotations
from dataclasses import dataclass
import os

import torch

@dataclass
class Settings:
    currentDir: str = os.path.join(os.getcwd(), os.path.dirname(__file__))
    datasetPath: str = os.path.join(currentDir, "dataset")
    trainRatio: float = 0.8
    channelsCount: int = 3
    imgW: int = 256
    imgH: int = 256    
    
    maxTrainSize: int | None = None
    maxTestSize: int | None = 10

    numWorkers: int = 4 #0 for single "thread" processing and/or if > 0 is crashing :-)
    batchSize: int = 10
    epochCount: int = 50

    saveEveryNthEpoch: int = 20
    loadModelCheckpoint: str | None = None
                                      #os.path.join(currentDir, "checkpoints", "model_chackpoint_40.pth")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")