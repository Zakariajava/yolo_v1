import torch
from src.loss import YoloLoss


def test_loss_runs_without_errors():
    """The loss should compute without crashing and return a scalar."""
    loss_fn = YoloLoss(S=7, B=2, C=20)

    predictions = torch.randn(2, 1470)
    target = torch.zeros(2, 7, 7, 30)

    loss = loss_fn(predictions, target)

    # Loss must be a scalar (0-dim tensor) and a finite number.
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)


def test_loss_is_near_zero_for_perfect_prediction():
    """When predictions match the target exactly, loss should be close to 0."""
    loss_fn = YoloLoss(S=7, B=2, C=20)

    # Target: one object in cell (3, 4), class 0, box center (0.5, 0.5), size (0.3, 0.3).
    target = torch.zeros(1, 7, 7, 30)
    target[0, 3, 4, 0] = 1                                            # class 0
    target[0, 3, 4, 20] = 1                                           # object confidence
    target[0, 3, 4, 21:25] = torch.tensor([0.5, 0.5, 0.3, 0.3])       # box coords

    # Predictions: both predicted boxes are identical to the GT box,
    # so whichever one "wins" will give near-zero error.
    predictions_grid = torch.zeros(1, 7, 7, 30)
    predictions_grid[0, 3, 4, 0] = 1                                  # class 0
    predictions_grid[0, 3, 4, 20] = 1                                 # box 1 confidence
    predictions_grid[0, 3, 4, 21:25] = torch.tensor([0.5, 0.5, 0.3, 0.3])
    predictions_grid[0, 3, 4, 25] = 0                                 # box 2 confidence (should be 0 on empty cells)
    predictions_grid[0, 3, 4, 26:30] = torch.tensor([0.5, 0.5, 0.3, 0.3])

    # Flatten to match model output shape (batch, 1470).
    predictions = predictions_grid.reshape(1, -1)

    loss = loss_fn(predictions, target)
    assert loss.item() < 1e-3


def test_loss_is_large_for_bad_prediction():
    """A wildly wrong prediction should produce a large loss."""
    loss_fn = YoloLoss(S=7, B=2, C=20)

    target = torch.zeros(1, 7, 7, 30)
    target[0, 3, 4, 0] = 1
    target[0, 3, 4, 20] = 1
    target[0, 3, 4, 21:25] = torch.tensor([0.5, 0.5, 0.3, 0.3])

    # Deliberately wrong predictions everywhere.
    predictions = torch.ones(1, 1470) * 2.0

    loss = loss_fn(predictions, target)
    assert loss.item() > 1.0