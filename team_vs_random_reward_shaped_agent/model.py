import torch
import torch.nn as nn


class TeamPolicyNet(nn.Module):
    """Policy head matching the RLlib Torch FCNet used by PPO team training."""

    def __init__(self, obs_size, hidden_size, action_logits_size):
        super().__init__()
        self.hidden1 = nn.Linear(obs_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.logits = nn.Linear(hidden_size, action_logits_size)

    def forward(self, obs):
        features = torch.tanh(self.hidden1(obs))
        features = torch.tanh(self.hidden2(features))
        return self.logits(features)
