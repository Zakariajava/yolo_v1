import torch
from src.model import Yolov1


def test(S=7, B=2, C=20):
    model = Yolov1(split_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((2, 3, 448, 448))
    output = model(x)
    
    expected_shape = (2, S * S * (C + B * 5))
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"Test passed. Output shape: {output.shape}")


if __name__ == "__main__":
    test()