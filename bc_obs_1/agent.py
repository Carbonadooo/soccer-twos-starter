import os

import numpy as np
import torch
import torch.nn as nn
from soccer_twos import AgentInterface

OBS_RAW_DIM = 336
N_PLAYERS   = 2


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


class BCObs1Agent(AgentInterface):
    """
    BC agent with 674-dim observation:
      [0:336]   z-score normalised obs
      [336:672] obs velocity (obs_t - obs_{t-1})
      [672:674] player ID one-hot
    """

    def __init__(self, env):
        super().__init__()
        self.name = "BC Obs1 Agent"
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint.pth")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        self.model = BCPolicy(ckpt["obs_size"], ckpt["action_branches"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.obs_mean = ckpt["obs_mean"]   # (336,)
        self.obs_std  = ckpt["obs_std"]    # (336,)
        self._prev_obs: dict = {}          # {player_id: prev_raw_obs}

    def _transform(self, obs: np.ndarray, player_id: int) -> np.ndarray:
        obs_norm = (obs - self.obs_mean) / self.obs_std
        prev     = self._prev_obs.get(player_id, np.zeros(OBS_RAW_DIM, dtype=np.float32))
        obs_diff = obs - prev
        self._prev_obs[player_id] = obs.copy()

        pid_oh = np.zeros(N_PLAYERS, dtype=np.float32)
        pid_oh[player_id % N_PLAYERS] = 1.0

        return np.concatenate([obs_norm, obs_diff, pid_oh]).astype(np.float32)

    def act(self, observation: dict) -> dict:
        actions = {}
        for player_id, obs in observation.items():
            feat = self._transform(obs, player_id)
            x = torch.FloatTensor(feat).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(x)
            actions[player_id] = np.array([torch.argmax(l).item() for l in logits])
        return actions
