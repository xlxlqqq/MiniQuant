import torch

from MiniQuant.utils.metrics import(
    mse,
    mae,
    max_error,
    relative_error
)

def test_mse():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 3.0, 5.0])

    loss = mse(x, y)
    expected_loss = 5.0 / 3.0

    assert abs(loss - expected_loss) < 1e-6, f"Expected {expected_loss}, but got {loss}"

def test_mae():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 3.0, 5.0])

    assert mae(x, y) == 1.0

def test_relative_error():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([2.0, 2.0])

    result = relative_error(x, y)

    assert result > 0

def test_max_error():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 3.0, 5.0])

    assert max_error(x, y) == 2.0