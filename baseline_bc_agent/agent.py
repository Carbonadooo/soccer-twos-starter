from pathlib import Path
from typing import Dict

import numpy as np
import torch
from soccer_twos import AgentInterface

from .model import BranchBCPolicy, PlayerPolicyNet


class BaselineBCAgent(AgentInterface):
    def __init__(self, env):
        super().__init__()
        self.model = None
        self._uses_branch_heads = False
        self.obs_size = int(env.observation_space.shape[0])
        self.action_branches = list(env.action_space.nvec)
        self._load_weights()
        self.model.eval()
        self.name = "BaselineBCPlayer"

    def _load_weights(self):
        base_dir = Path(__file__).resolve().parent
        branch_checkpoint_path = base_dir / "weights" / "checkpoint.pth"
        legacy_weights_path = base_dir / "weights" / "baseline_bc.pt"

        if branch_checkpoint_path.exists():
            payload = torch.load(branch_checkpoint_path, map_location="cpu")
            obs_size = int(payload.get("obs_size", self.obs_size))
            action_branches = payload.get("action_branches", self.action_branches)
            self.model = BranchBCPolicy(
                obs_size=obs_size,
                hidden_size=512,
                action_branches=action_branches,
            )
            self.model.load_state_dict(payload["state_dict"])
            self._uses_branch_heads = True
            return

        self.model = PlayerPolicyNet(
            obs_size=self.obs_size,
            hidden_size=256,
            action_logits_size=int(len(self.action_branches) * self.action_branches[0]),
        )
        if not legacy_weights_path.exists():
            print("BC weights not found. Agent will use random initial weights.")
            return
        payload = torch.load(legacy_weights_path, map_location="cpu")
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
            if self._uses_branch_heads:
                logits = self.model(obs_tensor)
                actions = np.array(
                    [
                        [torch.argmax(branch_logits[idx]).item() for branch_logits in logits]
                        for idx in range(obs_tensor.shape[0])
                    ],
                    dtype=np.int64,
                )
            else:
                logits = self.model(obs_tensor).view(-1, 3, 3)
                actions = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int64)
        return {player_id: actions[idx] for idx, player_id in enumerate(ordered_ids)}
