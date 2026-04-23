import torch
import torch.nn as nn


class PPOPolicyNet(nn.Module):
    """Policy head matching the RLlib Torch FCNet used by PPO."""

    def __init__(self, obs_size, hidden_size, action_size):
        super().__init__()
        self.hidden = nn.Linear(obs_size, hidden_size)
        self.logits = nn.Linear(hidden_size, action_size)

    def forward(self, obs):
        features = torch.tanh(self.hidden(obs))
        return self.logits(features)
