"""
Self-play PPO fine-tuning with:
  - bc_obs_0 weight initialisation
  - z-score obs normalisation (from bc_obs_0 checkpoint)
  - Potential-based reward shaping (ball progress + contest + coverage + support)

Run:
    python train_bc_obs0_shaped.py
"""

import os

import gym
import numpy as np
import ray
import torch
import torch.nn as nn
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import EnvType

import soccer_twos
from utils import RLLibWrapper

BC_CKPT          = "bc_obs_0/checkpoint.pth"
SAVE_DIR         = "./ray_results/PPO_bc_obs0_shaped"
NUM_ENVS_PER_WORKER = 3
MAX_TIMESTEPS    = 5_000_000
SAVE_EVERY_ITERS = 200

FIELD_HALF_LENGTH = 14.0
PREDICTION_HORIZON = 0.25
OWN_GOAL = np.array([-FIELD_HALF_LENGTH, 0.0], dtype=np.float32)
OPP_GOAL = np.array([ FIELD_HALF_LENGTH, 0.0], dtype=np.float32)

# Load normalisation stats at module level so workers can access them
_bc_data = torch.load(BC_CKPT, map_location="cpu")
OBS_MEAN = _bc_data["obs_mean"]
OBS_STD  = _bc_data["obs_std"]


# ── Reward shaping ─────────────────────────────────────────────────────────────

class BaselineShapingHelper:
    """
    Potential-based reward shaping for one team (2 players).
    Mirrors the design from train_ray_bc_finetune_vs_baseline_shaped.py.
    """

    def __init__(self):
        self._prev_ball_potential = None

    def reset(self):
        self._prev_ball_potential = None

    @staticmethod
    def _exp_dist(distance, scale):
        return float(np.exp(-distance / scale))

    def shape(self, info: dict, flip_x: bool = False) -> dict:
        """
        Args:
            info:   {0: player_info_dict, 1: player_info_dict}
            flip_x: True for team 1 (orange) — they attack in the -x direction
        Returns:
            {0: shaped_reward, 1: shaped_reward}
        """
        if not info or 0 not in info or 1 not in info:
            return {0: 0.0, 1: 0.0}

        sign = -1.0 if flip_x else 1.0   # flip field for orange team

        try:
            ball_pos = np.array(info[0]["ball_info"]["position"], dtype=np.float32)
            ball_vel = np.array(info[0]["ball_info"]["velocity"], dtype=np.float32)
        except (KeyError, TypeError):
            return {0: 0.0, 1: 0.0}

        pred_ball = ball_pos + PREDICTION_HORIZON * ball_vel
        pred_ball_x = float(np.clip(sign * pred_ball[0], -FIELD_HALF_LENGTH, FIELD_HALF_LENGTH))
        pred_ball_z = float(pred_ball[1])

        # Potential-based ball progress (only reward increments)
        progress_reward = 0.0
        ball_potential = pred_ball_x / FIELD_HALF_LENGTH
        if self._prev_ball_potential is not None:
            progress_reward = 0.02 * float(ball_potential - self._prev_ball_potential)
        self._prev_ball_potential = ball_potential

        # Per-player states
        player_states = []
        for pid in [0, 1]:
            try:
                p_pos = np.array(info[pid]["player_info"]["position"], dtype=np.float32)
                p_vel = np.array(info[pid]["player_info"].get("velocity", [0.0, 0.0]), dtype=np.float32)
            except (KeyError, TypeError):
                continue
            pred_player = p_pos + PREDICTION_HORIZON * p_vel
            pred_player[0] = np.clip(pred_player[0], -FIELD_HALF_LENGTH, FIELD_HALF_LENGTH)
            player_states.append({
                "pid":          pid,
                "pred_pos":     pred_player,
                "dist_to_ball": float(np.linalg.norm(pred_player - pred_ball)),
            })

        if len(player_states) != 2:
            return {0: 0.0, 1: 0.0}

        nearest_idx = int(np.argmin([s["dist_to_ball"] for s in player_states]))
        support_idx = 1 - nearest_idx
        nearest = player_states[nearest_idx]
        support = player_states[support_idx]

        shaped = {0: progress_reward, 1: progress_reward}

        # Nearest player: contest/control the ball
        shaped[nearest["pid"]] += 0.006 * self._exp_dist(nearest["dist_to_ball"], 2.2)

        # Defensive coverage: reward having a player between ball and own goal
        if pred_ball_x < 0.0:
            defenders_between = sum(
                1 for s in player_states
                if -FIELD_HALF_LENGTH <= sign * s["pred_pos"][0] <= pred_ball_x
                and abs(s["pred_pos"][1] - pred_ball_z) <= 2.5
            )
            bonus = 0.004 if defenders_between > 0 else -0.004
            shaped[0] += bonus
            shaped[1] += bonus

        # Support player positioning
        sx, sz = support["pred_pos"]
        sx_field = sign * sx
        if pred_ball_x >= 0.0:
            # Attacking: support stays slightly behind ball with useful width
            behind_ball = (pred_ball_x - 5.0) <= sx_field <= (pred_ball_x - 0.5)
            lateral_gap = abs(sz - pred_ball_z)
            width_reward = 0.003 * max(0.0, 1.0 - abs(lateral_gap - 2.5) / 2.5)
            shaped[support["pid"]] += width_reward if behind_ball else -0.003
        else:
            # Defensive: support covers goal lane
            between = -FIELD_HALF_LENGTH <= sx_field <= pred_ball_x
            lane = 0.0035 * self._exp_dist(abs(sz - pred_ball_z), 1.8)
            shaped[support["pid"]] += lane if between else -0.0035

        # Anti-crowding
        gap = float(np.linalg.norm(nearest["pred_pos"] - support["pred_pos"]))
        if gap < 1.2:
            penalty = 0.0035 * (1.2 - gap) / 1.2
            shaped[nearest["pid"]] -= penalty
            shaped[support["pid"]] -= penalty

        # Lateral separation bonus for support
        shaped[support["pid"]] += 0.002 * min(abs(sz - nearest["pred_pos"][1]) / 2.5, 1.0)

        return shaped


# ── Env wrapper ────────────────────────────────────────────────────────────────

class ObsNormRewardShapingWrapper(gym.Wrapper):
    """
    Wraps multiagent_player env:
      - z-score normalises all 4 agents' observations
      - applies BaselineShapingHelper to both teams
        (team 1 / orange has flip_x=True so directions are correct)
    """

    def __init__(self, env, mean: np.ndarray, std: np.ndarray):
        super().__init__(env)
        self.mean = mean
        self.std  = std
        self.shapers = {0: BaselineShapingHelper(), 1: BaselineShapingHelper()}

    def _norm(self, obs_dict: dict) -> dict:
        return {k: (v - self.mean) / self.std for k, v in obs_dict.items()}

    def reset(self):
        for s in self.shapers.values():
            s.reset()
        return self._norm(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Team 0 (players 0,1): attacks x=+14
        shaped0 = self.shapers[0].shape({0: info.get(0, {}), 1: info.get(1, {})}, flip_x=False)
        # Team 1 (players 2,3): attacks x=-14
        shaped1 = self.shapers[1].shape({0: info.get(2, {}), 1: info.get(3, {})}, flip_x=True)

        shaped_reward = {
            0: reward[0] + shaped0.get(0, 0.0),
            1: reward[1] + shaped0.get(1, 0.0),
            2: reward[2] + shaped1.get(0, 0.0),
            3: reward[3] + shaped1.get(1, 0.0),
        }

        return self._norm(obs), shaped_reward, done, info


# ── Env factory ───────────────────────────────────────────────────────────────

def create_shaped_env(env_config: dict = {}):
    mean = np.array(env_config.get("obs_mean", OBS_MEAN))
    std  = np.array(env_config.get("obs_std",  OBS_STD))

    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )

    _CUSTOM = {"obs_mean", "obs_std"}
    soccer_config = {k: v for k, v in env_config.items() if k not in _CUSTOM}

    env = soccer_twos.make(**soccer_config)
    env = ObsNormRewardShapingWrapper(env, mean, std)
    return RLLibWrapper(env)


# ── BC weight loader ───────────────────────────────────────────────────────────

def load_bc_weights(policy, bc_ckpt_path: str):
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
    print("bc_obs_0 weights loaded.")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [project_dir, os.environ.get("PYTHONPATH", "")]
    )

    ray.init(num_gpus=0, include_dashboard=False)

    tune.registry.register_env("Soccer", create_shaped_env)
    temp_env = create_shaped_env({
        "variation": EnvType.multiagent_player,
        "obs_mean": OBS_MEAN.tolist(),
        "obs_std":  OBS_STD.tolist(),
    })
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    config = {
        "num_gpus": 0,
        "num_workers": 8,
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "log_level": "WARN",
        "framework": "torch",
        "multiagent": {
            "policies": {"default": (None, obs_space, act_space, {})},
            "policy_mapping_fn": tune.function(lambda _: "default"),
            "policies_to_train": ["default"],
        },
        "env": "Soccer",
        "env_config": {
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "variation": EnvType.multiagent_player,
            "obs_mean": OBS_MEAN.tolist(),
            "obs_std":  OBS_STD.tolist(),
        },
        "model": {
            "vf_share_layers": True,
            "fcnet_hiddens": [512, 512],
        },
        "clip_param":         0.1,
        "lr":                 3e-5,
        "num_sgd_iter":       15,
        "sgd_minibatch_size": 512,
        "train_batch_size":   8000,
        "batch_mode":         "complete_episodes",
        "gamma":              0.99,
        "lambda":             0.95,
        "entropy_coeff":      0.01,
        "vf_loss_coeff":      0.5,
    }

    trainer = PPOTrainer(config=config, env="Soccer")
    load_bc_weights(trainer.get_policy("default"), BC_CKPT)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_reward = float("-inf")
    best_ckpt   = None

    print(f"\nStarting bc_obs_0 + reward shaping (target: {MAX_TIMESTEPS:,} steps)...\n")

    i = 0
    while True:
        i += 1
        result = trainer.train()
        ts     = result["timesteps_total"]
        reward = result.get("episode_reward_mean", float("nan"))

        print(f"Iter {i:4d} | ts={ts:>9,} | reward={reward:+.3f}")

        if i % SAVE_EVERY_ITERS == 0:
            ckpt = trainer.save(SAVE_DIR)
            print(f"  -> {ckpt}")

        if reward > best_reward and not np.isnan(reward):
            best_reward = reward
            best_ckpt   = trainer.save(os.path.join(SAVE_DIR, "best"))

        if ts >= MAX_TIMESTEPS:
            break

    final_ckpt = trainer.save(SAVE_DIR)
    print(f"\nDone.")
    print(f"Final : {final_ckpt}")
    print(f"Best  : {best_ckpt}  (reward={best_reward:.3f})")
