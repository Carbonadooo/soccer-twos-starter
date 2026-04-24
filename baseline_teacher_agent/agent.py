import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from soccer_twos import AgentInterface


BASELINE_CHECKPOINT_PATH = Path(
    __file__
).resolve().parents[1] / (
    "ceia_baseline_agent"
    "/ray_results/PPO_selfplay_twos/PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02"
    "/checkpoint_002449/checkpoint-2449"
)


class BaselinePolicyNet(nn.Module):
    def __init__(self, obs_size: int = 336, hidden_size: int = 256, action_logits_size: int = 9):
        super().__init__()
        self.hidden1 = nn.Linear(obs_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.logits = nn.Linear(hidden_size, action_logits_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.hidden1(obs))
        x = torch.relu(self.hidden2(x))
        return self.logits(x)


def _load_baseline_policy_state(checkpoint_path: Path):
    with checkpoint_path.open("rb") as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)
    worker_state = pickle.loads(checkpoint["worker"])
    return worker_state["state"]["default"]


def load_baseline_model(checkpoint_path: Path = BASELINE_CHECKPOINT_PATH) -> BaselinePolicyNet:
    policy_state = _load_baseline_policy_state(Path(checkpoint_path))
    model = BaselinePolicyNet(obs_size=336, hidden_size=256, action_logits_size=9)
    model.hidden1.weight.data.copy_(
        torch.from_numpy(policy_state["_hidden_layers.0._model.0.weight"])
    )
    model.hidden1.bias.data.copy_(
        torch.from_numpy(policy_state["_hidden_layers.0._model.0.bias"])
    )
    model.hidden2.weight.data.copy_(
        torch.from_numpy(policy_state["_hidden_layers.1._model.0.weight"])
    )
    model.hidden2.bias.data.copy_(
        torch.from_numpy(policy_state["_hidden_layers.1._model.0.bias"])
    )
    model.logits.weight.data.copy_(torch.from_numpy(policy_state["_logits._model.0.weight"]))
    model.logits.bias.data.copy_(torch.from_numpy(policy_state["_logits._model.0.bias"]))
    model.eval()
    return model


class TorchPolicyActor:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def act(self, observation):
        ordered_ids = sorted(observation.keys())
        obs_array = np.stack(
            [np.asarray(observation[player_id], dtype=np.float32) for player_id in ordered_ids],
            axis=0,
        )
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_array).to(self.device)
            logits = self.model(obs_tensor).view(-1, 3, 3)
            actions = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int64)
        return {player_id: actions[idx] for idx, player_id in enumerate(ordered_ids)}


class BaselineTeacherAgent(AgentInterface):
    def __init__(self, env):
        super().__init__()
        self.actor = TorchPolicyActor(load_baseline_model())
        self.name = "BaselineTeacherTorch"

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        return self.actor.act(observation)
