# ========================= Python imports ========================
from __future__ import annotations
from typing import Tuple, Optional, Union, Type

# ========================= Lib imports ========================
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================= My imports ========================


# ======================================================================================
# ======================================================================================
# ======================================================================================

class DiceLoss(nn.Module):

    def __init__(self, smooth: float = 1.0):
        super().__init__()

        self.smooth = smooth

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:

        intersection = (pred * target).sum(dim=2).sum(dim=2)

        loss = (1 - ((2. * intersection + self.smooth) / (
                    pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)))

        return loss.mean()

# ======================================================================================
# ======================================================================================
# ======================================================================================

class BceDiceLoss(nn.Module):

    def __init__(self, smooth: float = 1.0, bce_weight: float = 0.5):
        super().__init__()

        self.bce_weight = bce_weight
        self.diceLoss = DiceLoss(smooth)

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:

        bce = F.binary_cross_entropy_with_logits(pred, target)

        pred = torch.sigmoid(pred)
        dice = self.diceLoss(pred, target)

        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)

        return loss