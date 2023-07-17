import numpy as np
import torch, torch.nn as nn, torch.nn.functional as nnf
import torchvision
from typing import List

class ExampleSegmentationLoss(torch.nn.Module):

    """
    Return a loss module for the segmentation tasks. Supports XE+DICE, but DICE is disabled by default (and not even calculated), because it has a universal tendency to degrade the learning process.
    This class guarantees that there are no overlapping target masks, and constructs a background at runtime If you already had one, this constructed background is going to be empty).

    'weights' should be None (equal weighting) or a tensor containing class weights.
    'classes' should be the number of classes for which you have targets masks.
    If 'loss_for_background' is True, class 0 will be the computed background, and a loss will be calculated for it. If False, class 0 will instead be whatever class the first target mask is for.
    If 'allow_multiclass' is False (the default), any overlapping segmentations are resolved by preferring the higher class index. For example, if class 1 is 'liver' and class 2 is 'lesion', and the segmentation masks say that a pixel is both in the liver and in a lesion, the new 'truth' in this case would be that the pixel is only in class 2.
    """

    def __init__(
        self, 
        classes: int, 
        weights: torch.Tensor = None,
        on_the_fly_background: bool = True,
        allow_multiclass = False):

        super(ExampleSegmentationLoss, self).__init__()
        self.classes = classes
        self.on_the_fly_background = on_the_fly_background
        if weights is None: # if no weights given, equal weights by default
            if self.on_the_fly_background is True:
                weights = torch.Tensor([1 for c in range(self.classes+1)])
            else:
                weights = torch.Tensor([1 for c in range(self.classes)])
        self.weights = weights
        self.xe_module = nn.CrossEntropyLoss(weight = self.weights, reduction = "mean", ignore_index = -1)
        self.allow_multiclass = allow_multiclass

    def forward(self, predictions: torch.Tensor, targets: List[torch.Tensor,]):

        # De facto number of classes (including background computed at runtime)
        nc = (self.classes + 1 if self.on_the_fly_background is True else self.classes)

        # Sanity check for tensor shape
        if nc == predictions.size()[1]:
            pass
        else:
            raise ValueError(f"The amount of de facto used classes ({nc}) must be equal to the number of provided predictions ({predictions.size()[1]}). If loss_for_background ({self.loss_for_background}) is True, the number of provided predictions should be one greater.")

        if self.allow_multiclass is False:
            # Add a background target mask based on the other targets
            all_targets = [torch.ones_like(targets[0]).to(targets[0].device)]
            all_targets.extend(targets)
            
            # Last class index always has priority if two masks match in one location
            c_targets = torch.squeeze(self.classes - torch.argmax(torch.stack(tensors = all_targets[::-1], dim = 1), dim = 1), dim = 1)

            # Convert to one-hot encoding
            oh_targets = torch.nn.functional.one_hot(c_targets, num_classes = nc)
            oh_targets = torch.moveaxis(oh_targets, -1, 1).to(torch.float32)
        
        else:
            # Alternatively, if we don't care about class overlap, 
            # we just call every pixel that is not something else background, and leave it at that
            background = torch.clamp(torch.ones_like(targets[0]) - (targets[0]+targets[1]), min = 0)
            # and simply glue the tensors together
            oh_targets = torch.hstack([background, targets[0], targets[1]])
        
        # Compute CrossEntropy loss (since this class is just for testing, we outsource the math)
        xe_loss = self.xe_module.forward(predictions, oh_targets)
        return xe_loss 


# The below loss was adapted from:
# https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py

class BinaryDiceLoss(nn.Module):
    """
    Binary Dice Loss. Targets must be one-hot encoded.
    """
    def __init__(self, reduction: str = 'mean'):
        super(BinaryDiceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):

        intersection = torch.sum(prediction * target, dim = (1, 2))
        cardinality = torch.sum(prediction, dim = (1, 2)) + torch.sum(target, dim = (1, 2))

        loss = 1 - 2 * (intersection / cardinality)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise NotImplementedError

class DiceLoss(nn.Module):
    """
    Dice Loss. Targets must be one-hot encoded.
    """
    def __init__(self, classes: int, weights: torch.Tensor = None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.classes = classes
        if weights is None:
            self.weights = torch.Tensor([1. if c == 0 else 3. for c in range(self.classes)])
        else:
            self.weights = weights

    def forward(self, predictions: torch.Tensor, oh_targets: torch.Tensor):
        BinaryDice = BinaryDiceLoss(**self.kwargs)
        predictions = nnf.softmax(predictions, dim=1)
        nc = oh_targets.size()[1]

        total_loss = 0
        for i in range(nc):
            dice_loss = BinaryDice(predictions[:, i, :, :], oh_targets[:, i, :, :])
            if self.weights is not None:
                dice_loss *= self.weights[i]
            total_loss += dice_loss

        return total_loss / nc

class ExampleDiceCELoss(torch.nn.Module):

    """
    Return a loss module for the segmentation tasks.
    Accepts a weight tensor w_l to weight classes differently.
    The background mask is constructed at runtime for the entire batch at once.
    """

    def __init__(self, classes: int, l_xe: float = 1, l_dice: float = 1, w_l: torch.Tensor = None):
        super(ExampleDiceCELoss, self).__init__()
        self.classes = classes

        # Default to 1 to 1 weighting of classes
        if w_l is None:
            w_l = torch.Tensor([1 for c in range(self.classes)])
        
        self.XE = nn.CrossEntropyLoss(weight = w_l, reduction = "mean")
        self.DICE = DiceLoss(classes = classes, weights = w_l)
        self.l_xe = l_xe
        self.l_dice = l_dice

    def forward(self, predictions: torch.Tensor, targets: List[torch.Tensor,]):
        # For background, add a target mask based on the other targets which has background areas at c = 0
        all_targets = [torch.ones_like(targets[0]).to(targets[0].device)]
        all_targets.extend(targets)
        # Convert to onehot, last class index always has priority if two masks match in one location
        c_targets = torch.squeeze(self.classes - torch.argmax(torch.stack(tensors = all_targets[::-1], dim = 1), dim = 1) - 1)
        oh_targets = nnf.one_hot(c_targets, num_classes = self.classes).moveaxis(-1, 1)
        dice_loss = self.DICE.forward(predictions, oh_targets)
        xe_loss = self.XE.forward(
            torch.moveaxis(predictions.squeeze(), 1, -1).flatten(end_dim = -2), 
            c_targets.flatten()
            )
        return (self.l_xe * xe_loss + self.l_dice * dice_loss)/(self.l_xe + self.l_dice)