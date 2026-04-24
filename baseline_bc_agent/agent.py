from pathlib import Path
from typing import Dict

import numpy as np
import torch
from soccer_twos import AgentInterface

from .model import PlayerPolicyNet


class BaselineBCAgent(AgentInterface):
    def __init__(self, env):
        super().__init__()
        self.model = PlayerPolicyNet(
            obs_size=int(env.observation_space.shape[0]),
            hidden_size=256,
            action_logits_size=int(len(env.action_space.nvec) * env.action_space.nvec[0]),
        )
        self._load_weights()
        self.model.eval()
        self.name = "BaselineBCPlayer"

    def _load_weights(self):
        weights_path = Path(__file__).resolve().parent / "weights" / "baseline_bc.pt"
        if not weights_path.exists():
            print("BC weights not found. Agent will use random initial weights.")
            return
        payload = torch.load(weights_path, map_location="cpu")
        state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
        self.model.load_state_dict(state_dict)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        ordered_ids = sorted(observation.keys())
        obs_array = np.stack(
            [np.asarray(observation[player_id], dtype=np.float32) for player_id in ordered_ids],
            axis=0,
        )
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_array).float()
            logits = self.model(obs_tensor).view(-1, 3, 3)
            actions = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int64)
        return {player_id: actions[idx] for idx, player_id in enumerate(ordered_ids)}
