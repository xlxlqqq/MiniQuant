import torch

# 计算均方误差（Mean Squared Error, MSE）
def mse(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    return torch.mean((x - y) ** 2).item()

# 计算平均绝对误差（Mean Absolute Error, MAE）
def mae(x: torch.Tensor, y: torch.Tensor) -> float:
    return torch.mean(torch.abs(x - y)).item()

# 计算最大误差（Max Error）
def max_error(x: torch.Tensor, y: torch.Tensor) -> float:
    return torch.max(torch.abs(x - y)).item()

# 计算相对误差（Relative Error）
def relative_error(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    nuerator = torch.norm(x - y)

    denominator = torch.norm(x).clamp_min(eps)

    return (nuerator / denominator).item()