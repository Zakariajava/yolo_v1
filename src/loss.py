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
        # Reshape flat predictions (batch, 1470) into grid form (batch, S, S, C + B*5)
        # Layout per cell: [20 class probs | conf1, x1, y1, w1, h1 | conf2, x2, y2, w2, h2]
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        
        # IoU between each predicted box and the ground truth box (GT is always at 21:25).
        # Both iou_b1 and iou_b2 have shape (batch, S, S, 1) — one IoU per cell per image.
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        
        # Stack the two IoU tensors on a new leading axis → shape (2, batch, S, S, 1).
        # Axis 0 is the "box 1 vs box 2" dimension.
        ious = torch.stack([iou_b1, iou_b2], dim=0)
       
        # For each cell of each image independently, pick the box with the higher IoU.
        # iou_maxes: best IoU per cell, shape (batch, S, S, 1)
        # best_box:   index of the winning box (0 or 1) per cell, shape (batch, S, S, 1)
        # Mini-example with a 3x3 grid (batch=1), to illustrate torch.max(dim=0):
        #
        #   iou_b1 grid:        iou_b2 grid:         iou_maxes:          best_box:
        #   [0.3, 0.1, 0.8]     [0.9, 0.4, 0.2]      [0.9, 0.4, 0.8]     [1, 1, 0]
        #   [0.5, 0.7, 0.2]     [0.7, 0.3, 0.6]      [0.7, 0.7, 0.6]     [1, 0, 1]
        #   [0.6, 0.4, 0.9]     [0.1, 0.8, 0.5]      [0.6, 0.8, 0.9]     [0, 1, 0]
        iou_maxes, best_box = torch.max(ious, dim=0)
        
        exists_box = target[...,20:21] # Iobje_i if an object is in the cell i
        
        # ============================ #
        # Box coordinates Loss         #
        # ============================ #
        
        box_predictions = exists_box * (
             best_box * predictions[..., 26:30]
            + (1 - best_box) * predictions[...,21:25]
        )
        
        box_targets = exists_box * target[..., 21:25]
   
        # box_predictions[..., 0] → x          
        # box_predictions[..., 1] → y            
        # box_predictions[..., 2] → sqrt(w)       
        # box_predictions[..., 3] → sqrt(h)  
             
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4] + 1e-6)
        
        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )
        
        
        # ============================ #
        # Object Loss                  #
        # ============================ #
        pred_box = (
            best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]
        )
        
        # (N, S, S, 1) → (N*S*S, 1) 
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box, end_dim=-2),
            torch.flatten(exists_box * target[..., 20:21], end_dim=-2)
        )
        
        
        # ============================ #
        # NO Object Loss               #
        # ============================ #
        # For cells with NO object, BOTH predicted boxes should output confidence ≈ 0.
        # Unlike the object loss, there's no "winning box" here — we penalize both.
        # (1 - exists_box) flips the mask: 1 where there is no object, 0 where there is.

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], end_dim=-2),
            torch.flatten((1 - exists_box) * target[..., 20:21], end_dim=-2)
        )

        # (N, S, S, 1) → (N*S*S, 1)
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], end_dim=-2),
            torch.flatten((1 - exists_box) * target[..., 20:21], end_dim=-2)
        )
        
        # ============================ #
        # Class Loss                   #
        # ============================ #
        # (N, S, S, 20) → (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )
        
        # Total loss = terms 1+2 (coords) + term 3 (obj) + term 4 (noobj) + term 5 (class)
        # lambda_coord amplifies localization; lambda_noobj damps empty-cell confidence.
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
        
        return loss