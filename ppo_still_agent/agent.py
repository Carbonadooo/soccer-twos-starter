import glob
import os
import pickle
from typing import Dict

from gym_unity.envs import ActionFlattener
import numpy as np
import torch
from soccer_twos import AgentInterface

from .model import PPOPolicyNet


class PPOStillAgent(AgentInterface):
    """
    PPO policy trained with example_ray_ppo_sp_still.py.

    The training setup uses flatten_branched=True, so this agent predicts a
    Discrete action and maps it back to the Soccer-Twos MultiDiscrete action.
    """

    def __init__(self, env):
        super().__init__()

        self.flattener = ActionFlattener(env.action_space.nvec)
        self.model = PPOPolicyNet(
            obs_size=env.observation_space.shape[0],
            hidden_size=512,
            action_size=self.flattener.action_space.n,
        )
        self._load_checkpoint()
        self.model.eval()

    def _load_checkpoint(self):
        checkpoint_path = self._find_checkpoint()
        if checkpoint_path is None:
            print("PPO checkpoint not found. Agent will use random initial weights.")
            return

        with open(checkpoint_path, "rb") as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)

        worker_state = pickle.loads(checkpoint["worker"])
        policy_state = worker_state["state"]["default_policy"]

        self.model.hidden.weight.data.copy_(
            torch.from_numpy(policy_state["_hidden_layers.0._model.0.weight"])
        )
        self.model.hidden.bias.data.copy_(
            torch.from_numpy(policy_state["_hidden_layers.0._model.0.bias"])
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

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        with torch.no_grad():
            for player_id, obs in observation.items():
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                logits = self.model(obs_tensor)
                discrete_action = int(torch.argmax(logits, dim=1).item())
                actions[player_id] = self.flattener.lookup_action(discrete_action)
        return actions
