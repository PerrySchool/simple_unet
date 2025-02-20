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
    
    maxTrainSize: int | None = 100
    maxTestSize: int | None = 10

    numWorkers: int = 0
    batchSize: int = 2
    epochCount: int = 100

    saveEveryNthEpoch: int = 20
    loadModelCheckpoint: str | None = None

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")