import torch
import torch.nn as nn
import torch.optim as optim
from typing import Iterable, Tuple


class DQN(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_sizes: Iterable[int] = (128, 128)
    ) -> None:
        super().__init__()

        layers = []
        input_dim = state_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h

        layers.append(nn.Linear(input_dim, num_actions))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_dqn(
    state_dim: int,
    num_actions: int,
    lr: float = 1e-3,
    hidden_sizes: Iterable[int] = (128, 128)
) -> Tuple[DQN, optim.Optimizer, nn.Module]:
    model = DQN(state_dim, num_actions, hidden_sizes)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    return model, optimizer, loss_fn
