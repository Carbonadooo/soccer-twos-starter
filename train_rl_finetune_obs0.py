"""
RL fine-tuning from bc_obs_0 weights.

Key difference vs train_rl_finetune.py:
  - Env applies the same z-score normalisation that bc_obs_0 was trained on
  - obs_mean / obs_std are loaded from bc_obs_0/checkpoint.pth and injected
    into every worker env via env_config
  - BC weight shapes are identical (336-dim input), only the mapping changes

Run after distill_bc_obs.py:
    python train_rl_finetune_obs0.py
"""

import os

import gym
import numpy as np
import ray
import torch
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import EnvType

import soccer_twos
from utils import RLLibWrapper

BC_CKPT          = "bc_obs_0/checkpoint.pth"
SAVE_DIR         = "./ray_results/PPO_bc_obs0_finetune"
NUM_ENVS_PER_WORKER = 3
MAX_TIMESTEPS    = 5_000_000
SAVE_EVERY_ITERS = 200

# Module-level globals so workers can access mean/std without env_config
_bc_data = torch.load(BC_CKPT, map_location="cpu")
OBS_MEAN = _bc_data["obs_mean"]   # np.ndarray (336,)
OBS_STD  = _bc_data["obs_std"]    # np.ndarray (336,)


# ── Obs normalisation wrapper ──────────────────────────────────────────────────

class ObsNormWrapper(gym.Wrapper):
    """Applies per-dim z-score normalisation to every agent's observation."""

    def __init__(self, env, mean: np.ndarray, std: np.ndarray):
        super().__init__(env)
        self.mean = mean
        self.std  = std

    def _norm(self, obs_dict: dict) -> dict:
        return {k: (v - self.mean) / self.std for k, v in obs_dict.items()}

    def reset(self):
        return self._norm(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._norm(obs), reward, done, info


# ── Env factory ───────────────────────────────────────────────────────────────

def create_obs0_env(env_config: dict = {}):
    # Use get (not pop) — Ray calls this multiple times per worker
    mean = np.array(env_config.get("obs_mean", OBS_MEAN))
    std  = np.array(env_config.get("obs_std",  OBS_STD))

    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )

    # Filter out custom keys before passing to soccer_twos.make
    _CUSTOM_KEYS = {"obs_mean", "obs_std"}
    soccer_config = {k: v for k, v in env_config.items() if k not in _CUSTOM_KEYS}

    env = soccer_twos.make(**soccer_config)
    env = ObsNormWrapper(env, mean, std)
    return RLLibWrapper(env)


# ── BC weight loader ───────────────────────────────────────────────────────────

def load_bc_weights(policy, bc_ckpt_path: str):
    """Same weight mapping as train_rl_finetune.py — shapes unchanged (336-dim)."""
    ckpt = torch.load(bc_ckpt_path, map_location="cpu")
    bc   = ckpt["state_dict"]

    logit_w = torch.cat([bc["heads.0.weight"], bc["heads.1.weight"], bc["heads.2.weight"]], dim=0)
    logit_b = torch.cat([bc["heads.0.bias"],   bc["heads.1.bias"],   bc["heads.2.bias"]],   dim=0)

    weights = policy.get_weights()
    weights["_hidden_layers.0._model.0.weight"] = bc["shared.0.weight"].numpy()
    weights["_hidden_layers.0._model.0.bias"]   = bc["shared.0.bias"].numpy()
    weights["_hidden_layers.1._model.0.weight"] = bc["shared.2.weight"].numpy()
    weights["_hidden_layers.1._model.0.bias"]   = bc["shared.2.bias"].numpy()
    weights["_logits._model.0.weight"]          = logit_w.numpy()
    weights["_logits._model.0.bias"]            = logit_b.numpy()
    policy.set_weights(weights)
    print("bc_obs_0 weights loaded into RLLib policy.")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [project_dir, os.environ.get("PYTHONPATH", "")]
    )

    # Load normalisation stats from bc_obs_0 checkpoint
    bc_data  = torch.load(BC_CKPT, map_location="cpu")
    obs_mean = bc_data["obs_mean"].tolist()   # list → JSON-serialisable for Ray
    obs_std  = bc_data["obs_std"].tolist()

    ray.init(num_gpus=0, include_dashboard=False)

    tune.registry.register_env("Soccer", create_obs0_env)

    # Build a temp env to get obs/act spaces (after normalisation obs shape is same 336)
    temp_env = create_obs0_env({
        "variation": EnvType.multiagent_player,
        "obs_mean": obs_mean,
        "obs_std":  obs_std,
    })
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    config = {
        # ── system ────────────────────────────────────────────────────────────
        "num_gpus": 0,
        "num_workers": 8,
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "log_level": "WARN",
        "framework": "torch",

        # ── multiagent self-play ───────────────────────────────────────────────
        "multiagent": {
            "policies": {"default": (None, obs_space, act_space, {})},
            "policy_mapping_fn": tune.function(lambda _: "default"),
            "policies_to_train": ["default"],
        },
        "env": "Soccer",
        "env_config": {
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "variation": EnvType.multiagent_player,
            "obs_mean": obs_mean,
            "obs_std":  obs_std,
        },

        # ── model ─────────────────────────────────────────────────────────────
        "model": {
            "vf_share_layers": True,
            "fcnet_hiddens": [512, 512],
        },

        # ── core PPO ──────────────────────────────────────────────────────────
        "clip_param":         0.1,    # small: protect BC init from being overwritten
        "lr":                 3e-5,   # fine-tune lr, smaller than default 5e-5
        "num_sgd_iter":       15,     # default 30 risks overfitting with sparse rewards
        "sgd_minibatch_size": 512,    # larger minibatch → more stable gradients

        # ── batch / episode ───────────────────────────────────────────────────
        "train_batch_size":   8000,   # soccer episodes ~600 steps, need big batches
        "batch_mode":         "complete_episodes",  # don't cut episodes mid-way

        # ── reward / advantage ────────────────────────────────────────────────
        "gamma":              0.99,   # long-horizon soccer
        "lambda":             0.95,   # GAE: reduce variance for sparse rewards

        # ── exploration & stability ───────────────────────────────────────────
        "entropy_coeff":      0.01,   # must have entropy for self-play exploration
        "vf_loss_coeff":      0.5,    # higher weight on critic (vf_share_layers=True)
    }

    trainer = PPOTrainer(config=config, env="Soccer")
    load_bc_weights(trainer.get_policy("default"), BC_CKPT)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_reward = float("-inf")
    best_ckpt   = None

    print(f"\nStarting RL fine-tuning from bc_obs_0 (target: {MAX_TIMESTEPS:,} steps)...\n")

    i = 0
    while True:
        i += 1
        result = trainer.train()
        ts     = result["timesteps_total"]
        reward = result.get("episode_reward_mean", float("nan"))

        print(f"Iter {i:4d} | ts={ts:>9,} | reward={reward:+.3f}")

        if i % SAVE_EVERY_ITERS == 0:
            ckpt = trainer.save(SAVE_DIR)
            print(f"  -> checkpoint: {ckpt}")

        if reward > best_reward and not np.isnan(reward):
            best_reward = reward
            best_ckpt   = trainer.save(os.path.join(SAVE_DIR, "best"))

        if ts >= MAX_TIMESTEPS:
            break

    final_ckpt = trainer.save(SAVE_DIR)
    print(f"\nDone.")
    print(f"Final : {final_ckpt}")
    print(f"Best  : {best_ckpt}  (reward={best_reward:.3f})")
