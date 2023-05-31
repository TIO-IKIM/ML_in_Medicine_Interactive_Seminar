from typing import List
from collections import OrderedDict
import torch
import torch.nn
import torch.nn.functional as nnf
import numpy as np

def evaluate_classifier_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str = None):
    """
    Evaluates a classifier on a given dataset.
    The model is expected to return one-hot predictions as first element.
    Any other value that is returned is ignored.
    The dataloader is expected to return at least data and targets as first and second elements.
    Any other value that is returned is ignored.
    """
    if device is not None:
        model = model.to(device)
    
    with torch.no_grad():
        model.eval()
        
        seen = []
        test_losses = []
        hits = 0
        criterion = torch.nn.CrossEntropyLoss()
        
        # Get data and target, throw away any other returns from the dataloader.
        for data, targets, *_ in dataloader:
            if device is not None:
                data, targets = data.to(device), targets.to(device)
            model_returns = model(data)
            
            # Get predictions from model. If there is other returns, toss them.
            if isinstance(model_returns, tuple) and len(model_returns) > 1:
                oh_predictions, *_ = model_returns
            else:
                oh_predictions = model_returns

            loss = criterion(oh_predictions, targets)
            c_predictions = torch.argmax(oh_predictions, dim=1)
            hits += sum([1 if p == t else 0 for p, t in zip(c_predictions, targets)])
            seen.append(targets.size()[0])
            test_losses.append(loss.item())

        accuracy = hits/sum(seen)
        avg_test_loss = sum([l*s for l, s in zip(test_losses, seen)])/sum(seen)

    return accuracy, avg_test_loss

class Segmentation_Metrics():

    """
    Computes:

    Weighted Dice Score. 
    Weighted IoU.
    Weighted Precision.
    Weighted Recall.
    Targets must be a List of List of Tensors.
    (Outer list has batches, inner list has each target as a separate tensor.)
    """

    def __init__(self, classes, verbose: bool = False, **kwargs):
        self.results = OrderedDict()
        self.verbose = verbose
        self.kwargs = kwargs
        self.weights = self.kwargs.get("weights", None)
        self.classes = classes

    def forward(self, predictions: List[torch.Tensor, ], targets: List[torch.Tensor, ]):
        eps = 1e-6 # for stability reasons
        seen = 0
        bsl = {}

        for b in range(len(targets)):
            nt = len(targets[b])
            
            # Convert predictions to binary one-hot format for proper scoring
            p_arg = nnf.one_hot(torch.argmax(predictions[b].to(torch.float32), dim = 1), num_classes = nt+1).moveaxis(-1, 1)
            
            for c in range(nt+1):
                if c == 0:
                    # Build the background label target on the fly
                    all_targets = [torch.ones_like(targets[b][c]).to(targets[b][c].device)]
                    all_targets.extend(targets[b])
                    # Convert to onehot, last class index always has priority if two masks match in one location
                    c_targets = torch.squeeze(nt - torch.argmax(torch.stack(tensors = all_targets[::-1], dim = 1), dim = 1))
                    oh_targets = nnf.one_hot(c_targets, num_classes = self.classes).moveaxis(-1, 1)
                    target = oh_targets[:, c, :, :].type(torch.bool).squeeze()
                else:
                    # All other targets already exist
                    target = oh_targets[:, c, :, :].type(torch.bool).squeeze()
                
                prediction = p_arg[:, c, :, :].type(torch.bool)
                intersection = torch.sum(prediction * target)
                p_cardinality = torch.sum(prediction)
                t_cardinality = torch.sum(target)
                cardinality = p_cardinality + t_cardinality
                union = torch.sum((prediction + target))

                bs = target.size()[0]

                if self.weights is None:
                    weight = 1
                else:
                    weight = self.weights[c]

                # Dice Score
                if intersection.item() == 0 and cardinality.item() == 0:
                    # Special case where we match an all-empty target
                    dice_score = np.nan
                else:
                    # Regular case
                    dice_score = (2. * intersection / (cardinality + eps)).item()

                self.results[f"#dice_{b}_{c}"] = dice_score
                bsl[f"#dice_{b}_{c}"] = (np.nan if np.isnan(dice_score) else bs * weight)
                #####

                # IoU
                if intersection.item() == 0 and union.item() == 0:
                    # Special case where we match an all-empty target
                    iou = np.nan
                else:
                    # Regular case
                    iou = (intersection / (union + eps)).item() # DEBUG 2?

                self.results[f"#iou_{b}_{c}"] = iou
                bsl[f"#iou_{b}_{c}"] = (np.nan if np.isnan(dice_score) else bs * weight)
                #####

                # Precision
                if intersection.item() == 0 and p_cardinality.item() == 0:
                    # Special case where we match an all-empty target
                    precision = np.nan
                else:
                    # Regular case
                    precision = (intersection / (p_cardinality + eps)).item()

                self.results[f"#precision_{b}_{c}"] = precision
                bsl[f"#precision_{b}_{c}"] = (np.nan if np.isnan(dice_score) else bs * weight)
                #####

                # Recall
                if intersection.item() == 0 and t_cardinality.item() == 0:
                    # Special case where we match an all-empty target
                    recall = np.nan
                else:
                    # Regular case
                    recall = (intersection / (t_cardinality + eps)).item()

                self.results[f"#recall_{b}_{c}"] = recall
                bsl[f"#recall_{b}_{c}"] = (np.nan if np.isnan(dice_score) else bs * weight)
                #####

        # Compute the average for each metric. Exclude batches with edge cases (which were nan) from score and from seen.
        for c in range(nt+1):
            dice_seen = sum([v for b, v in bsl.items() if (b.startswith("#dice") and b.endswith(str(c)) and not np.isnan(v))])
            self.results[f"dice_avg_class_{c}"] = sum([self.results[f"#dice_{b}_{c}"] * bsl[f"#dice_{b}_{c}"] for b in range(len(targets)) if not np.isnan(self.results[f"#dice_{b}_{c}"])]) / (dice_seen + eps)
            iou_seen = sum([v for b, v in bsl.items() if (b.startswith("#iou") and b.endswith(str(c)) and not np.isnan(v))])
            self.results[f"iou_avg_class_{c}"] = sum([self.results[f"#iou_{b}_{c}"] * bsl[f"#iou_{b}_{c}"] for b in range(len(targets)) if not np.isnan(self.results[f"#iou_{b}_{c}"])]) / (iou_seen + eps)
            precision_seen = sum([v for b, v in bsl.items() if (b.startswith("#precision") and b.endswith(str(c)) and not np.isnan(v))])
            self.results[f"precision_avg_class_{c}"] = sum([self.results[f"#precision_{b}_{c}"] * bsl[f"#precision_{b}_{c}"] for b in range(len(targets)) if not np.isnan(self.results[f"#precision_{b}_{c}"])]) / (precision_seen + eps)
            recall_seen = sum([v for b, v in bsl.items() if (b.startswith("#recall") and b.endswith(str(c)) and not np.isnan(v))])
            self.results[f"recall_avg_class_{c}"] = sum([self.results[f"#recall_{b}_{c}"] * bsl[f"#recall_{b}_{c}"] for b in range(len(targets)) if not np.isnan(self.results[f"#recall_{b}_{c}"])]) / (recall_seen + eps)

        return self.results

def evaluate_segmentation_model(model, dataloader, device, w_l = None):
    
    """
    Evaluates a segmentation model on a given dataset.
    The model is expected to return one-hot predictions as first element.
    Any other value that is returned is ignored.
    The dataloader is expected to return at least data and targets as first and second elements.
    Any other value that is returned is ignored.
    """

    if device is not None:
        model = model.to(device)
    
    with torch.no_grad():
        model.eval()
        
        pl, tl = [], []
        Seg_Metrics = Segmentation_Metrics(classes = 3, weights = w_l)
        
        # Get data and target, throw away any other returns from the dataloader.
        for data, targets, *_ in dataloader:
            if device is not None:
                data = data.to(device)
                targets = [target.to(device) for target in targets]
            model_returns = model(data)
            
            # Get predictions from model. If there is other returns, toss them.
            if isinstance(model_returns, tuple) and len(model_returns) > 1:
                oh_predictions, *_ = model_returns
            else:
                oh_predictions = model_returns

            pl.extend([oh_predictions])
            tl.extend([targets])

        metrics = Seg_Metrics.forward(pl, tl)

    return metrics