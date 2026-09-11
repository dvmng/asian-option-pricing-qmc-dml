from __future__ import annotations

from typing import Dict

import torch
from torch import nn


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "softplus":
        return nn.Softplus(beta=1.0)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def init_linear(linear: nn.Linear) -> None:
    nn.init.xavier_uniform_(linear.weight)
    nn.init.zeros_(linear.bias)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, activation: str):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        init_linear(self.fc1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.act = make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.fc2(self.act(self.fc1(x)))
        return self.act(x + residual)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, width: int, n_blocks: int, activation: str):
        super().__init__()
        self.input = nn.Linear(input_dim, width)
        init_linear(self.input)
        self.act = make_activation(activation)
        self.blocks = nn.ModuleList(
            [ResidualBlock(width, activation) for _ in range(n_blocks)]
        )
        self.output = nn.Linear(width, 1)
        init_linear(self.output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.input(x))
        for block in self.blocks:
            h = block(h)
        return self.output(h)


def build_model(input_dim: int, params: Dict[str, object]) -> nn.Module:
    if str(params["backbone"]).lower() != "residual":
        raise ValueError("Phase 4B only accepts the frozen residual backbone.")
    return ResidualMLP(
        input_dim=input_dim,
        width=int(params["width"]),
        n_blocks=int(params["n_blocks"]),
        activation=str(params["activation"]),
    )


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def spot_derivative(
    model: nn.Module,
    x: torch.Tensor,
    spot_index: int = 0,
    create_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.requires_grad:
        x = x.requires_grad_(True)

    pred = model(x).squeeze(-1)
    grad_all = torch.autograd.grad(
        pred.sum(),
        x,
        create_graph=create_graph,
        retain_graph=create_graph,
        only_inputs=True,
    )[0]
    return pred, grad_all[:, spot_index]
