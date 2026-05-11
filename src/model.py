"""
YOLOv1 with ResNet18 backbone pretrained on ImageNet.

Architecture:
    Input (B, 3, 448, 448)
      ↓
    [ResNet18 backbone, pretrained on ImageNet]
      ↓ produces (B, 512, 14, 14)
    [Adapter: 2 conv layers]
      ↓ produces (B, 1024, 7, 7)
    [Detection head: FC layers]
      ↓ produces (B, 7*7*(C + 5*B))
    Output (B, 4410) for COCO (C=80, B=2)

Differences from the paper:
    - Backbone: ResNet18 instead of Darknet24.
    - Pretrained on ImageNet (vs paper which also uses ImageNet pretrain but with Darknet).
    - Smaller and faster than the original Darknet head.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class Yolov1(nn.Module):
    """
    YOLOv1 detection model with ResNet18 backbone.

    Args:
        split_size: grid size (S in the paper). Default 7.
        num_boxes:  bounding boxes predicted per cell (B). Default 2.
        num_classes: number of classes (C). Default 80 for COCO.
    """

    def __init__(self, split_size=7, num_boxes=2, num_classes=80):
        super().__init__()
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes

        # === Backbone: ResNet18 pretrained on ImageNet ===
        # Load with default weights (IMAGENET1K_V1).
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove the classification head (avgpool and fc).
        # We only keep the convolutional layers up to layer4.
        # Input  (B, 3, 448, 448)
        # Output (B, 512, 14, 14)
        self.backbone = nn.Sequential(
            resnet.conv1,    # (B, 64, 224, 224)
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # (B, 64, 112, 112)
            resnet.layer1,   # (B, 64, 112, 112)
            resnet.layer2,   # (B, 128, 56, 56)
            resnet.layer3,   # (B, 256, 28, 28)
            resnet.layer4,   # (B, 512, 14, 14)
        )

        # === Adapter: 2 conv layers ===
        # Bridge between backbone output (512, 14, 14) and head expectation (1024, 7, 7).
        # First conv: spatial downsample 14→7, channel up 512→1024.
        # Second conv: refine features at (1024, 7, 7).
        self.adapter = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # === Detection head: FC layers ===
        # Same as the original Darknet head.
        # 1024 * 7 * 7 = 50176 input features.
        # 4096 hidden units.
        # 7 * 7 * (C + 5*B) = 4410 output features for COCO.
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.C + self.B * 5)),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: tensor (B, 3, 448, 448).
        Returns:
            tensor (B, S*S*(C+5*B)) — flat output, to be reshaped by caller.
        """
        x = self.backbone(x)   # (B, 512, 14, 14)
        x = self.adapter(x)    # (B, 1024, 7, 7)
        x = self.head(x)       # (B, S*S*(C+5*B))
        return x


if __name__ == "__main__":
    # Quick sanity check: run a dummy forward pass.
    model = Yolov1(split_size=7, num_boxes=2, num_classes=80)
    x = torch.randn(2, 3, 448, 448)
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Expected:     (2, {7*7*(80+2*5)}) = (2, 4410)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")