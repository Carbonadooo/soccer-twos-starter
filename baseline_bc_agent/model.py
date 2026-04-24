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


class BranchBCPolicy(nn.Module):
    def __init__(self, obs_size: int = 336, hidden_size: int = 512, action_branches=None):
        super().__init__()
        if action_branches is None:
            action_branches = [3, 3, 3]
        self.action_branches = list(action_branches)
        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_size, n) for n in self.action_branches])

    def forward(self, obs: torch.Tensor):
        features = self.shared(obs)
        return [head(features) for head in self.heads]
