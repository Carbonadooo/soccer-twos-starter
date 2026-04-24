import glob
import os
import pickle
from typing import Dict, List

import numpy as np
import torch
from soccer_twos import AgentInterface

from .model import TeamPolicyNet


class TeamVsRandomAgent(AgentInterface):
    """
    PPO team policy trained with train_ray_team_vs_random.py.

    The underlying training setup controls both teammates jointly, so this
    agent concatenates the two player observations and predicts six action
    branches: three for each player.
    """

    def __init__(self, env):
        super().__init__()

        self.player_ids = [0, 1]
        self.branch_size = int(env.action_space.nvec[0])
        self.num_branches_per_player = int(len(env.action_space.nvec))
        self.total_branches = self.num_branches_per_player * len(self.player_ids)

        obs_size = int(env.observation_space.shape[0] * len(self.player_ids))
        action_logits_size = self.total_branches * self.branch_size
        self.model = TeamPolicyNet(
            obs_size=obs_size,
            hidden_size=512,
            action_logits_size=action_logits_size,
        )
        self._load_checkpoint()
        self.model.eval()
        self.name = "TeamVsRandomPPO"

    def _load_checkpoint(self):
        checkpoint_path = self._find_checkpoint()
        if checkpoint_path is None:
            print("PPO checkpoint not found. Agent will use random initial weights.")
            return

        with open(checkpoint_path, "rb") as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)

        worker_state = pickle.loads(checkpoint["worker"])
        policy_state = worker_state["state"]["default_policy"]

        self.model.hidden1.weight.data.copy_(
            torch.from_numpy(policy_state["_hidden_layers.0._model.0.weight"])
        )
        self.model.hidden1.bias.data.copy_(
            torch.from_numpy(policy_state["_hidden_layers.0._model.0.bias"])
        )
        self.model.hidden2.weight.data.copy_(
            torch.from_numpy(policy_state["_hidden_layers.1._model.0.weight"])
        )
        self.model.hidden2.bias.data.copy_(
            torch.from_numpy(policy_state["_hidden_layers.1._model.0.bias"])
        )
        self.model.logits.weight.data.copy_(
            torch.from_numpy(policy_state["_logits._model.0.weight"])
        )
        self.model.logits.bias.data.copy_(
            torch.from_numpy(policy_state["_logits._model.0.bias"])
        )

    def _find_checkpoint(self):
        agent_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(agent_dir, "checkpoint")
        candidates = sorted(
            glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")),
            key=os.path.getmtime,
            reverse=True,
        )
        candidates = [
            path for path in candidates if not path.endswith(".tune_metadata")
        ]
        return candidates[0] if candidates else None

    def _split_actions(self, branch_actions: np.ndarray) -> List[np.ndarray]:
        player_actions = []
        for player_idx in range(len(self.player_ids)):
            start = player_idx * self.num_branches_per_player
            end = start + self.num_branches_per_player
            player_actions.append(branch_actions[start:end].astype(np.int64))
        return player_actions

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        ordered_player_ids = sorted(observation.keys())
        if len(ordered_player_ids) != len(self.player_ids):
            raise ValueError(
                f"Expected {len(self.player_ids)} players, got {len(ordered_player_ids)}"
            )

        joint_obs = np.concatenate(
            [observation[player_id] for player_id in ordered_player_ids], axis=0
        )

        with torch.no_grad():
            obs_tensor = torch.from_numpy(joint_obs).float().unsqueeze(0)
            logits = self.model(obs_tensor).view(self.total_branches, self.branch_size)
            branch_actions = torch.argmax(logits, dim=1).cpu().numpy()

        split_actions = self._split_actions(branch_actions)
        return {
            player_id: split_actions[idx]
            for idx, player_id in enumerate(ordered_player_ids)
        }
