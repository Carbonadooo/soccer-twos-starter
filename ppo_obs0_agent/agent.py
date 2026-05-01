import glob
import os
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from soccer_twos import AgentInterface


POLICY_NAME = "default"
AGENT_DIR = Path(__file__).resolve().parent
BC_OBS0_CKPT = (AGENT_DIR / "../bc_obs_0/checkpoint.pth").resolve()


class PPOObs0Policy(nn.Module):
    def __init__(self, obs_size: int = 336, hidden_size: int = 512, action_logits_size: int = 9):
        super().__init__()
        self.hidden1 = nn.Linear(obs_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.logits = nn.Linear(hidden_size, action_logits_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.hidden1(obs))
        x = torch.relu(self.hidden2(x))
        return self.logits(x)


def _checkpoint_candidates() -> list:
    candidates = []
    candidates.extend(
        p for p in glob.glob(os.path.join(str(AGENT_DIR / "checkpoint"), "**", "checkpoint-*"), recursive=True)
        if not p.endswith(".tune_metadata") and not p.endswith(".is_checkpoint")
    )
    candidates.extend(
        p for p in glob.glob(os.path.join(str(AGENT_DIR), "checkpoint-*"))
        if not p.endswith(".tune_metadata") and not p.endswith(".is_checkpoint")
    )
    legacy_dir = (AGENT_DIR.parent / "checkpoint_001500").resolve()
    candidates.extend(
        p for p in glob.glob(os.path.join(str(legacy_dir), "checkpoint-*"))
        if not p.endswith(".tune_metadata") and not p.endswith(".is_checkpoint")
    )
    return sorted(set(candidates))


def _find_latest_checkpoint() -> str:
    candidates = _checkpoint_candidates()
    if not candidates:
        raise FileNotFoundError("No local checkpoint found for ppo_obs0_agent")

    def _num(path: str) -> int:
        try:
            return int(Path(path).stem.split("-")[-1])
        except ValueError:
            return -1

    return sorted(candidates, key=_num)[-1]


class PPOObs0Agent(AgentInterface):
    """
    PPO fine-tuned agent with local checkpoint loading.
    Uses the packaged checkpoint directly instead of spinning up a Ray trainer.
    """

    def __init__(self, env):
        super().__init__()
        self.name = "PPO Obs0 Agent"

        bc_data = torch.load(BC_OBS0_CKPT, map_location="cpu")
        self.obs_mean = bc_data["obs_mean"]
        self.obs_std = bc_data["obs_std"]

        checkpoint_path = _find_latest_checkpoint()
        print(f"[PPOObs0Agent] Loading: {checkpoint_path}")

        obs_size = int(env.observation_space.shape[0])
        action_logits_size = int(len(env.action_space.nvec) * env.action_space.nvec[0])
        self.model = PPOObs0Policy(obs_size=obs_size, hidden_size=512, action_logits_size=action_logits_size)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()

    def _load_checkpoint(self, checkpoint_path: str):
        with open(checkpoint_path, "rb") as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)

        worker_state = pickle.loads(checkpoint["worker"])
        policy_state = worker_state["state"][POLICY_NAME]

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

    def _norm(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.obs_mean) / self.obs_std

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        ordered_ids = sorted(observation.keys())
        obs_array = np.stack(
            [self._norm(np.asarray(observation[player_id], dtype=np.float32)) for player_id in ordered_ids],
            axis=0,
        )
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_array).float()
            logits = self.model(obs_tensor).view(-1, 3, 3)
            actions = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int64)
        return {player_id: actions[idx] for idx, player_id in enumerate(ordered_ids)}
