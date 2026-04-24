import torch
import torch.nn as nn


class PlayerPolicyNet(nn.Module):
    def __init__(self, obs_size: int = 336, hidden_size: int = 256, action_logits_size: int = 9):
        super().__init__()
        self.hidden1 = nn.Linear(obs_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.logits = nn.Linear(hidden_size, action_logits_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.hidden1(obs))
        x = torch.relu(self.hidden2(x))
        return self.logits(x)
