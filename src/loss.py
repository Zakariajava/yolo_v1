import torch 
import torch.nn as nn
from .utils import intersection_over_union
from .config import SPLIT_SIZE, NUM_BOXES, NUM_CLASSES, LAMBDA_COORD, LAMBDA_NOOBJ

class YoloLoss(nn.Module):
    def __init__(self, S=SPLIT_SIZE, B=NUM_BOXES, C=NUM_CLASSES):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = LAMBDA_COORD
        self.lambda_noobj = LAMBDA_NOOBJ
        
        
    def forward(self, predictions, target):
        # Reshape flat predictions (batch, S*S*(C+5B)) into grid form (batch, S, S, C+5B).
        # Layout per cell: [C class probs | conf1, x1, y1, w1, h1 | conf2, x2, y2, w2, h2]
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        
        # Box layout indices, derived from C so the same code works for any class count.
        C = self.C
        OBJ_IDX = C                          # conf1 slot, right after the class probs
        BOX1_SLICE = slice(C + 1, C + 5)     # x1, y1, w1, h1
        CONF2_IDX = C + 5                    # conf2 slot
        BOX2_SLICE = slice(C + 6, C + 10)    # x2, y2, w2, h2
        
        # IoU between each predicted box and the ground truth box (GT is always in BOX1_SLICE).
        # Both iou_b1 and iou_b2 have shape (batch, S, S, 1) — one IoU per cell per image.
        iou_b1 = intersection_over_union(predictions[..., BOX1_SLICE], target[..., BOX1_SLICE])
        iou_b2 = intersection_over_union(predictions[..., BOX2_SLICE], target[..., BOX1_SLICE])
        
        # Stack the two IoU tensors on a new leading axis → shape (2, batch, S, S, 1).
        ious = torch.stack([iou_b1, iou_b2], dim=0)
       
        # For each cell, pick the box with the higher IoU.
        iou_maxes, best_box = torch.max(ious, dim=0)
        
        # exists_box: 1 if an object's center falls in the cell, 0 otherwise.
        # Shape: (batch, S, S, 1).
        exists_box = target[..., OBJ_IDX:OBJ_IDX+1]
        
        # ============================ #
        # Box coordinates Loss         #
        # ============================ #
        
        # Choose the "responsible" box (the one with higher IoU).
        box_predictions = exists_box * (
             best_box * predictions[..., BOX2_SLICE]
            + (1 - best_box) * predictions[..., BOX1_SLICE]
        )
        
        box_targets = exists_box * target[..., BOX1_SLICE]
   
        # Take square root of w and h (paper's trick to weight small/large boxes equally).
        # Use sign(x) * sqrt(|x|) to handle negative predictions without NaN.
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4]) + 1e-6
        )
        
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4] + 1e-6)
        
        # Flatten (N, S, S, 4) → (N*S*S, 4) and compute MSE.
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )
        
        
        # ============================ #
        # Object Loss                  #
        # ============================ #
        # Confidence of the "responsible" box should match the IoU between pred and GT.
        # We use the conf slot of whichever box won the responsibility match.
        pred_box_conf = (
            best_box * predictions[..., CONF2_IDX:CONF2_IDX+1]
            + (1 - best_box) * predictions[..., OBJ_IDX:OBJ_IDX+1]
        )
        
        # Flatten (N, S, S, 1) → (N*S*S, 1) and compute MSE only on cells with objects.
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box_conf, end_dim=-2),
            torch.flatten(exists_box * target[..., OBJ_IDX:OBJ_IDX+1], end_dim=-2)
        )
        
        
        # ============================ #
        # NO Object Loss               #
        # ============================ #
        # For cells with NO object, BOTH predicted boxes should output confidence ≈ 0.
        # (1 - exists_box) flips the mask: 1 where there is no object, 0 where there is.
        
        # Box 1 confidence in empty cells should be 0.
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., OBJ_IDX:OBJ_IDX+1], end_dim=-2),
            torch.flatten((1 - exists_box) * target[..., OBJ_IDX:OBJ_IDX+1], end_dim=-2)
        )
        
        # Box 2 confidence in empty cells should also be 0.
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., CONF2_IDX:CONF2_IDX+1], end_dim=-2),
            torch.flatten((1 - exists_box) * target[..., OBJ_IDX:OBJ_IDX+1], end_dim=-2)
        )
        
        # ============================ #
        # Class Loss                   #
        # ============================ #
        # Pull the predicted class distribution towards the one-hot target,
        # only in cells that contain an object.
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :C], end_dim=-2),
            torch.flatten(exists_box * target[..., :C], end_dim=-2)
        )
        
        # Total loss = terms 1+2 (coords) + term 3 (obj) + term 4 (noobj) + term 5 (class)
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
        
        return loss