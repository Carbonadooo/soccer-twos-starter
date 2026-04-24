import glob
import os
import pickle
from typing import Dict

import numpy as np
import torch
from soccer_twos import AgentInterface

from .model import PlayerPolicyNet


class BaselineBCFinetuneShapedAgent(AgentInterface):
    def __init__(self, env):
        super().__init__()
        self.name = "BaselineBCFinetuneShapedAgent"
        obs_size = int(env.observation_space.shape[0])
        action_logits_size = int(len(env.action_space.nvec) * env.action_space.nvec[0])
        self.model = PlayerPolicyNet(
            obs_size=obs_size,
            hidden_size=256,
            action_logits_size=action_logits_size,
        )
        self._load_checkpoint()
        self.model.eval()

    def _find_checkpoint(self):
        agent_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(agent_dir, "checkpoint")
        candidates = sorted(
            glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")),
            key=os.path.getmtime,
            reverse=True,
        )
        candidates = [path for path in candidates if not path.endswith(".tune_metadata")]
        return candidates[0] if candidates else None

    def _load_checkpoint(self):
        checkpoint_path = self._find_checkpoint()
        if checkpoint_path is None:
            print("Shaped finetune checkpoint not found. Agent will use random initial weights.")
            return

        with open(checkpoint_path, "rb") as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)

        worker_state = pickle.loads(checkpoint["worker"])
        policy_state = worker_state["state"]["default"]
        if "hidden1.weight" in policy_state:
            self.model.hidden1.weight.data.copy_(torch.from_numpy(policy_state["hidden1.weight"]))
            self.model.hidden1.bias.data.copy_(torch.from_numpy(policy_state["hidden1.bias"]))
            self.model.hidden2.weight.data.copy_(torch.from_numpy(policy_state["hidden2.weight"]))
            self.model.hidden2.bias.data.copy_(torch.from_numpy(policy_state["hidden2.bias"]))
            self.model.logits.weight.data.copy_(torch.from_numpy(policy_state["logits.weight"]))
            self.model.logits.bias.data.copy_(torch.from_numpy(policy_state["logits.bias"]))
            return

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
