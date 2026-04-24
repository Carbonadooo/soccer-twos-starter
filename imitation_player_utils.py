import os
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


BASELINE_CHECKPOINT_PATH = Path(
    "ceia_baseline_agent"
    "/ray_results/PPO_selfplay_twos/PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02"
    "/checkpoint_002449/checkpoint-2449"
)


class PlayerPolicyNet(nn.Module):
    def __init__(self, obs_size: int = 336, hidden_size: int = 256, action_logits_size: int = 9):
        super().__init__()
        self.hidden1 = nn.Linear(obs_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.logits = nn.Linear(hidden_size, action_logits_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.hidden1(obs))
        x = torch.relu(self.hidden2(x))
        return self.logits(x)


def _load_rllib_policy_state(checkpoint_path: Path, policy_name: str = "default") -> Dict[str, np.ndarray]:
    with checkpoint_path.open("rb") as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)
    worker_state = pickle.loads(checkpoint["worker"])
    return worker_state["state"][policy_name]


def load_rllib_mlp_weights(
    model: PlayerPolicyNet,
    policy_state: Dict[str, np.ndarray],
    hidden_prefixes=("_hidden_layers.0._model.0", "_hidden_layers.1._model.0"),
    logits_prefix="_logits._model.0",
):
    model.hidden1.weight.data.copy_(torch.from_numpy(policy_state[f"{hidden_prefixes[0]}.weight"]))
    model.hidden1.bias.data.copy_(torch.from_numpy(policy_state[f"{hidden_prefixes[0]}.bias"]))
    model.hidden2.weight.data.copy_(torch.from_numpy(policy_state[f"{hidden_prefixes[1]}.weight"]))
    model.hidden2.bias.data.copy_(torch.from_numpy(policy_state[f"{hidden_prefixes[1]}.bias"]))
    model.logits.weight.data.copy_(torch.from_numpy(policy_state[f"{logits_prefix}.weight"]))
    model.logits.bias.data.copy_(torch.from_numpy(policy_state[f"{logits_prefix}.bias"]))


def load_baseline_model(checkpoint_path: Optional[Path] = None) -> PlayerPolicyNet:
    checkpoint_path = Path(checkpoint_path or BASELINE_CHECKPOINT_PATH)
    policy_state = _load_rllib_policy_state(checkpoint_path)
    model = PlayerPolicyNet(obs_size=336, hidden_size=256, action_logits_size=9)
    load_rllib_mlp_weights(model, policy_state)
    model.eval()
    return model


def action_logits_to_branches(logits: torch.Tensor) -> torch.Tensor:
    return logits.view(-1, 3, 3)


def greedy_branch_actions(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(action_logits_to_branches(logits), dim=-1)


class TorchPolicyActor:
    def __init__(self, model: PlayerPolicyNet, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        ordered_ids = sorted(observation.keys())
        obs_array = np.stack(
            [np.asarray(observation[player_id], dtype=np.float32) for player_id in ordered_ids],
            axis=0,
        )
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_array).to(self.device)
            logits = self.model(obs_tensor)
            actions = greedy_branch_actions(logits).cpu().numpy().astype(np.int64)
        return {player_id: actions[idx] for idx, player_id in enumerate(ordered_ids)}


def load_bc_weights(model: PlayerPolicyNet, checkpoint_path: Path):
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state_dict)


def export_bc_weights(src_checkpoint: Path, dst_checkpoint: Path):
    dst_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if src_checkpoint.resolve() != dst_checkpoint.resolve():
        torch.save(torch.load(src_checkpoint, map_location="cpu"), dst_checkpoint)


def find_latest_bc_checkpoint(root_dir: Path) -> Optional[Path]:
    candidates = sorted(root_dir.rglob("best.pt"), key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else None
