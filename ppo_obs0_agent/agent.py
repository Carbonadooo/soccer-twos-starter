import glob
import os
from pathlib import Path
from typing import Dict

import gym
import numpy as np
import ray
import torch
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.base_env import BaseEnv
from soccer_twos import AgentInterface

POLICY_NAME  = "default"

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../ray_results/PPO_bc_obs0_vs_baseline",
)

BC_OBS0_CKPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../bc_obs_0/checkpoint.pth",
)


def _find_latest_checkpoint(root: str) -> str:
    candidates = [
        p for p in glob.glob(os.path.join(root, "**/checkpoint-*"), recursive=True)
        if not p.endswith(".tune_metadata") and not p.endswith(".is_checkpoint")
    ]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {root}")
    def _num(p):
        try:
            return int(Path(p).stem.split("-")[-1])
        except ValueError:
            return -1
    return sorted(candidates, key=_num)[-1]


class PPOObs0Agent(AgentInterface):
    """
    PPO fine-tuned agent (bc_obs_0 init + reward shaping vs baseline).
    Reconstructs config from scratch — no params.pkl needed.
    Applies z-score obs normalisation (stats from bc_obs_0/checkpoint.pth).
    """

    def __init__(self, env):
        super().__init__()
        self.name = "PPO Obs0 Agent"

        # ── Obs normalisation stats ────────────────────────────────────────────
        bc_data = torch.load(BC_OBS0_CKPT, map_location="cpu")
        self.obs_mean = bc_data["obs_mean"]
        self.obs_std  = bc_data["obs_std"]

        # ── Find checkpoint ────────────────────────────────────────────────────
        checkpoint_path = _find_latest_checkpoint(CHECKPOINT_DIR)
        print(f"[PPOObs0Agent] Loading: {checkpoint_path}")

        # ── Reconstruct config (matches train_bc_obs0_vs_baseline.py) ─────────
        obs_space = env.observation_space   # Box(336,)
        act_space = env.action_space        # MultiDiscrete([3,3,3])

        ray.init(ignore_reinit_error=True)
        tune.registry.register_env("Soccer", lambda *_: BaseEnv())

        config = {
            "num_workers": 0,
            "num_gpus":    0,
            "framework":   "torch",
            "env":         "Soccer",
            "multiagent": {
                "policies": {"default": (None, obs_space, act_space, {})},
                "policy_mapping_fn": tune.function(lambda _: "default"),
                "policies_to_train": ["default"],
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512, 512],
            },
        }

        trainer = PPOTrainer(config=config, env="Soccer")
        trainer.restore(checkpoint_path)
        self.policy = trainer.get_policy(POLICY_NAME)

    def _norm(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.obs_mean) / self.obs_std

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id, obs in observation.items():
            obs_norm = self._norm(obs)
            action, *_ = self.policy.compute_single_action(obs_norm)
            actions[player_id] = action
        return actions
