"""
PPO fine-tuning: our policy vs ceia_baseline_agent, random team each episode.

Features:
  - bc_obs_0 weight initialisation
  - z-score obs normalisation
  - Potential-based reward shaping (ball progress + contest + coverage + support)
  - Each episode randomly assigns our policy to blue (0,1) or orange (2,3)
    → prevents directional bias, baseline sees varied opponents

Run:
    python train_bc_obs0_vs_baseline.py
"""

import os
import random

import gym
import numpy as np
import ray
import torch
from ray import tune
from ray.rllib import MultiAgentEnv
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import EnvType

import soccer_twos
from imitation_player_utils import TorchPolicyActor, load_baseline_model

BC_CKPT          = "bc_obs_0/checkpoint.pth"
SAVE_DIR         = "./ray_results/PPO_bc_obs0_vs_baseline"
NUM_ENVS_PER_WORKER = 3
MAX_TIMESTEPS    = 15_000_000
SAVE_EVERY_ITERS = 250

FIELD_HALF_LENGTH  = 14.0
PREDICTION_HORIZON = 0.25
SHAPING_SCALE      = 0.15   # keeps total shaped reward << goal reward (±1)

_bc_data = torch.load(BC_CKPT, map_location="cpu")
OBS_MEAN = _bc_data["obs_mean"]
OBS_STD  = _bc_data["obs_std"]


# ── Reward shaping ─────────────────────────────────────────────────────────────

class BaselineShapingHelper:
    def __init__(self):
        self._prev_ball_potential = None

    def reset(self):
        self._prev_ball_potential = None

    @staticmethod
    def _exp_dist(d, scale):
        return float(np.exp(-d / scale))

    def shape(self, info: dict, flip_x: bool = False) -> dict:
        """
        info  : {0: player_info, 1: player_info}  (relative IDs within our team)
        flip_x: True when we are orange — flips x so 'forward' means x decreasing
        """
        if not info or 0 not in info or 1 not in info:
            return {0: 0.0, 1: 0.0}

        sign = -1.0 if flip_x else 1.0

        try:
            ball_pos = np.array(info[0]["ball_info"]["position"], dtype=np.float32)
            ball_vel = np.array(info[0]["ball_info"]["velocity"], dtype=np.float32)
        except (KeyError, TypeError):
            return {0: 0.0, 1: 0.0}

        pred_ball   = ball_pos + PREDICTION_HORIZON * ball_vel
        pred_ball_x = float(np.clip(sign * pred_ball[0], -FIELD_HALF_LENGTH, FIELD_HALF_LENGTH))
        pred_ball_z = float(pred_ball[1])

        # Potential-based ball progress
        progress_reward = 0.0
        ball_potential = pred_ball_x / FIELD_HALF_LENGTH
        if self._prev_ball_potential is not None:
            progress_reward = 0.02 * float(ball_potential - self._prev_ball_potential)
        self._prev_ball_potential = ball_potential

        # Per-player states
        states = []
        for pid in [0, 1]:
            try:
                p_pos = np.array(info[pid]["player_info"]["position"], dtype=np.float32)
                p_vel = np.array(info[pid]["player_info"].get("velocity", [0.0, 0.0]), dtype=np.float32)
            except (KeyError, TypeError):
                continue
            pred_p = p_pos + PREDICTION_HORIZON * p_vel
            pred_p[0] = np.clip(pred_p[0], -FIELD_HALF_LENGTH, FIELD_HALF_LENGTH)
            states.append({"pid": pid, "pred_pos": pred_p,
                           "dist_to_ball": float(np.linalg.norm(pred_p - pred_ball))})

        if len(states) != 2:
            return {0: 0.0, 1: 0.0}

        near_i  = int(np.argmin([s["dist_to_ball"] for s in states]))
        supp_i  = 1 - near_i
        nearest = states[near_i]
        support = states[supp_i]

        shaped = {0: progress_reward, 1: progress_reward}

        # Shooting signal: reward nearest player when ball moves toward opponent goal
        # (replaces raw proximity reward which caused ball-orbiting behaviour)
        if pred_ball_x > 0.0 and nearest["dist_to_ball"] < 3.0:
            shoot = 0.006 * max(0.0, float(sign * ball_vel[0]) / 10.0)
            shaped[nearest["pid"]] += shoot

        # Defensive coverage (only positive bonus — no negative penalty to avoid
        # discouraging forward play)
        if pred_ball_x < 0.0:
            n_between = sum(
                1 for s in states
                if -FIELD_HALF_LENGTH <= sign * s["pred_pos"][0] <= pred_ball_x
                and abs(s["pred_pos"][1] - pred_ball_z) <= 2.5
            )
            if n_between > 0:
                shaped[0] += 0.004
                shaped[1] += 0.004

        # Support player positioning
        sx, sz = support["pred_pos"]
        sx_field = sign * sx
        if pred_ball_x >= 0.0:
            behind = (pred_ball_x - 5.0) <= sx_field <= (pred_ball_x - 0.5)
            w = 0.003 * max(0.0, 1.0 - abs(abs(sz - pred_ball_z) - 2.5) / 2.5)
            shaped[support["pid"]] += w if behind else -0.003
        else:
            between = -FIELD_HALF_LENGTH <= sx_field <= pred_ball_x
            lane = 0.0035 * self._exp_dist(abs(sz - pred_ball_z), 1.8)
            shaped[support["pid"]] += lane if between else -0.0035

        # Anti-crowding (raised threshold: 1.2 → 3.0 so it actually triggers)
        gap = float(np.linalg.norm(nearest["pred_pos"] - support["pred_pos"]))
        if gap < 3.0:
            pen = 0.0035 * (3.0 - gap) / 3.0
            shaped[nearest["pid"]] -= pen
            shaped[support["pid"]] -= pen

        # Lateral separation bonus for support
        shaped[support["pid"]] += 0.002 * min(
            abs(sz - nearest["pred_pos"][1]) / 2.5, 1.0
        )

        # Global scale: keep total shaped reward well below goal reward (±1)
        return {k: v * SHAPING_SCALE for k, v in shaped.items()}


# ── Environment ────────────────────────────────────────────────────────────────

class VsBaselineShapedEnv(MultiAgentEnv):
    """
    Our policy (2 players) vs ceia_baseline_agent (2 players).
    Each episode randomly assigns our policy to blue (0,1) or orange (2,3).
    RLLib always sees our agents as IDs 0 and 1.
    """

    def __init__(self, env_config=None):
        env_config = dict(env_config or {})
        self.base_env = soccer_twos.make(
            variation=EnvType.multiagent_player,
            worker_id=env_config.get("worker_id", 0),
        )
        self.observation_space = self.base_env.observation_space
        self.action_space      = self.base_env.action_space

        self.obs_mean = np.array(env_config.get("obs_mean", OBS_MEAN))
        self.obs_std  = np.array(env_config.get("obs_std",  OBS_STD))

        self.baseline_actor = TorchPolicyActor(load_baseline_model())
        self.shaping        = BaselineShapingHelper()

        # Set at reset()
        self.our_ids     = [0, 1]
        self.opp_ids     = [2, 3]
        self.flip_x      = False
        self.last_raw    = None

    def _norm(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.obs_mean) / self.obs_std

    def reset(self):
        # Randomly assign our policy to blue or orange each episode
        if random.random() < 0.5:
            self.our_ids, self.opp_ids, self.flip_x = [0, 1], [2, 3], False
        else:
            self.our_ids, self.opp_ids, self.flip_x = [2, 3], [0, 1], True

        self.shaping.reset()
        self.last_raw = self.base_env.reset()

        return {
            0: self._norm(self.last_raw[self.our_ids[0]]),
            1: self._norm(self.last_raw[self.our_ids[1]]),
        }

    def step(self, action_dict):
        # Baseline acts on its own (raw) observations, relative IDs 0,1
        opp_raw = {
            0: self.last_raw[self.opp_ids[0]],
            1: self.last_raw[self.opp_ids[1]],
        }
        opp_actions = self.baseline_actor.act(opp_raw)

        env_actions = {
            self.our_ids[0]: action_dict[0],
            self.our_ids[1]: action_dict[1],
            self.opp_ids[0]: opp_actions[0],
            self.opp_ids[1]: opp_actions[1],
        }

        raw_obs, reward, done, info = self.base_env.step(env_actions)
        self.last_raw = raw_obs

        # Reward shaping — use our players' info, flip direction if orange
        our_info = {
            0: info.get(self.our_ids[0], {}),
            1: info.get(self.our_ids[1], {}),
        }
        shaped = self.shaping.shape(our_info, flip_x=self.flip_x)

        return (
            {
                0: self._norm(raw_obs[self.our_ids[0]]),
                1: self._norm(raw_obs[self.our_ids[1]]),
            },
            {
                0: reward[self.our_ids[0]] + shaped.get(0, 0.0),
                1: reward[self.our_ids[1]] + shaped.get(1, 0.0),
            },
            {0: done["__all__"], 1: done["__all__"], "__all__": done["__all__"]},
            {
                0: info.get(self.our_ids[0], {}),
                1: info.get(self.our_ids[1], {}),
            },
        )

    def close(self):
        self.base_env.close()


# ── Env factory ───────────────────────────────────────────────────────────────

def create_vs_baseline_env(env_config=None):
    raw = dict(env_config or {})
    if hasattr(env_config, "worker_index"):
        num_envs = raw.get("num_envs_per_worker", 1)
        raw["worker_id"] = env_config.worker_index * num_envs + env_config.vector_index
    return VsBaselineShapedEnv(raw)


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

    tune.registry.register_env("Soccer", create_vs_baseline_env)

    temp_env = create_vs_baseline_env({"num_envs_per_worker": 1})
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
        "env_config": {"num_envs_per_worker": NUM_ENVS_PER_WORKER},
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

    print(f"\nTraining vs baseline, random team each episode "
          f"(target: {MAX_TIMESTEPS:,} steps)...\n")

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
