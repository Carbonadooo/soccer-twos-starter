import os

import numpy as np
import torch
import torch.nn as nn
from soccer_twos import AgentInterface


class BCPolicy(nn.Module):
    def __init__(self, obs_size: int, action_branches: list):
        super().__init__()
        self.action_branches = action_branches
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(512, n) for n in action_branches])

    def forward(self, x):
        f = self.shared(x)
        return [head(f) for head in self.heads]


class BCAgent(AgentInterface):
    def __init__(self, env):
        super().__init__()
        self.name = "BC Agent"
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint.pth")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model = BCPolicy(ckpt["obs_size"], ckpt["action_branches"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def act(self, observation):
        actions = {}
        for player_id, obs in observation.items():
            x = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(x)
            actions[player_id] = np.array([torch.argmax(l).item() for l in logits])
        return actions
