import numpy as np
import torch, torch.nn as nn, torch.nn.functional as nnf
import torchvision
from typing import List

class ExampleSegmentationLoss(torch.nn.Module):
    """
    Custom CrossEntropyLoss. Weights losses by weights tensor.
    """
    def __init__(self, classes: int, weights: List[float, ] = None):

        super(ExampleSegmentationLoss, self).__init__()

        self.classes = classes
        self.weights = weights
        if self.weights is None:
            self.weights = torch.tensor([1.] * self.classes)
        elif isinstance(self.weights, list):
            self.weights = torch.tensor(self.weights, dtype = torch.float32)
        elif torch.is_tensor(self.weights):
            pass
        else:
            raise NotImplementedError

    def forward(self, predictions: torch.tensor, targets: List[torch.tensor, ], allow_multiclass: bool = False, norm_for_weights: bool = False):

        # Create background tensor
        if allow_multiclass is False:
            # If we want to make sure there is no overlap between classes, we first make a background of all 1s ...
            all_targets = [torch.ones_like(targets[0]).to(targets[0].device)]
            # We stack all targets on top of another ...
            all_targets.extend(targets)
            # We switch the order, and use argmax - this finds the first match, meaning we jot down the class
            # index of whatever target we have. If there is more than one, argmax simply returns the first one,
            # which, in our case, is the highest class index. This also neatly resolves all conflicts between
            # the all-1s background tensor.
            c_targets = torch.squeeze(self.classes - torch.argmax(torch.stack(tensors = all_targets[::-1], dim = 1), dim = 1) - 1)
            # Finally, we convert to a one-hot representation.
            # Since the new dimension is added at dim = -1, we move it
            oh_targets = nn.functional.one_hot(c_targets, num_classes = self.classes).moveaxis(-1, 1)
        else:
            # Alternatively, if we don't care about class overlap, 
            # we just call every pixel that is not something else background, and leave it at that
            background = torch.clamp(torch.ones_like(targets[0]) - (targets[0]+targets[1]), min = 0)
            # and simply glue the tensors together
            oh_targets = torch.hstack([background, targets[0], targets[1]])

        # Calculate cross entropy
        B, C, X, Y = predictions.size()
        eps = 1e-9
        # Be careful - we must multiply elementwise here (* or torch.mul), NOT with a matrix product (@ or torch.matmul)!
        xe = -1 * torch.log(nn.functional.softmax(predictions, dim = 1) + eps) - oh_targets.to(dtype = predictions.dtype)

        # Multiply with weight tensor (which we expand to be the same size as our softmaxed predictions)
        # Note that this implementation is slightly different from that of PyTorch in how weights are applied.
        # (In the special case of equal weights, and norming for weights, the two are equivalent)
        w = self.weights.to(device = predictions.device).view(1, -1, 1, 1).expand(B, C, X, Y)
        xe_sum = torch.sum(xe * w)

        # Normalize by dividing by B * X * Y (and normalize to account for custom weights, if desired)
        w_sum = (1 if norm_for_weights is False else torch.sum(self.weights))
        norm_sum = xe_sum / (B * w_sum * X * Y)
        
        return norm_sum

# Adapted from:
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