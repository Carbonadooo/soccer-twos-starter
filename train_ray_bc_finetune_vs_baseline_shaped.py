import os
from argparse import ArgumentParser
from pathlib import Path

import gym
import numpy as np

if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "object"):
    np.object = object

import ray
import torch
import torch.nn as nn
from ray import tune
from ray.rllib import MultiAgentEnv
from ray.rllib.models import ModelCatalog
import ray.rllib.models.catalog as model_catalog_module
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.logger import CSVLoggerCallback, JsonLoggerCallback
from soccer_twos import EnvType

import soccer_twos
from imitation_player_utils import TorchPolicyActor, load_baseline_model


NUM_ENVS_PER_WORKER = 2
BC_CHECKPOINT_PATH = Path("bc_obs_0/checkpoint.pth")
FIELD_HALF_LENGTH = 14.0
PREDICTION_HORIZON = 0.25
GOAL_Z = 0.0
OWN_GOAL = np.asarray([-FIELD_HALF_LENGTH, GOAL_Z], dtype=np.float32)
OPP_GOAL = np.asarray([FIELD_HALF_LENGTH, GOAL_Z], dtype=np.float32)


def parse_args():
    parser = ArgumentParser(
        description="Finetune a BC-initialized player policy against the baseline with reward shaping."
    )
    parser.add_argument("--timesteps-total", type=int, default=3600000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-envs-per-worker", type=int, default=NUM_ENVS_PER_WORKER)
    parser.add_argument("--rollout-fragment-length", type=int, default=500)
    parser.add_argument("--train-batch-size", type=int, default=8000)
    parser.add_argument("--checkpoint-freq", type=int, default=20)
    parser.add_argument("--bc-checkpoint", default=str(BC_CHECKPOINT_PATH))
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--clip-param", type=float, default=0.1)
    parser.add_argument(
        "--experiment-name",
        default="PPO_bc_finetune_vs_baseline_shaped_lr2e5_clip01_3p6M",
    )
    return parser.parse_args()


class BaselineShapingHelper:
    def __init__(self):
        self._prev_ball_potential = None

    def reset(self):
        self._prev_ball_potential = None

    @staticmethod
    def _safe_array(value):
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _predicted_position(position, velocity):
        return position + PREDICTION_HORIZON * velocity

    @staticmethod
    def _exp_distance_reward(distance, scale):
        return float(np.exp(-distance / scale))

    def shape_rewards(self, info):
        if not info or 0 not in info or 1 not in info:
            return {0: 0.0, 1: 0.0}

        controlled_players = [info[0], info[1]]
        try:
            ball_pos = self._safe_array(controlled_players[0]["ball_info"]["position"])
            ball_vel = self._safe_array(controlled_players[0]["ball_info"]["velocity"])
        except (KeyError, TypeError, ValueError):
            return {0: 0.0, 1: 0.0}

        pred_ball = self._predicted_position(ball_pos, ball_vel)
        pred_ball_x = float(np.clip(pred_ball[0], -FIELD_HALF_LENGTH, FIELD_HALF_LENGTH))
        pred_ball_z = float(pred_ball[1])

        # Shared team-level ball progress: only reward forward progress increments.
        progress_reward = 0.0
        ball_potential = pred_ball_x / FIELD_HALF_LENGTH
        if self._prev_ball_potential is not None:
            progress_reward = 0.02 * float(ball_potential - self._prev_ball_potential)
        self._prev_ball_potential = ball_potential

        player_states = []
        for player_id, player_info in zip([0, 1], controlled_players):
            try:
                player_pos = self._safe_array(player_info["player_info"]["position"])
                player_vel = self._safe_array(
                    player_info["player_info"].get("velocity", [0.0, 0.0])
                )
            except (KeyError, TypeError, ValueError):
                continue

            pred_player = self._predicted_position(player_pos, player_vel)
            pred_player[0] = np.clip(pred_player[0], -FIELD_HALF_LENGTH, FIELD_HALF_LENGTH)
            player_states.append(
                {
                    "player_id": player_id,
                    "pred_pos": pred_player,
                    "dist_to_ball": float(np.linalg.norm(pred_player - pred_ball)),
                }
            )

        if len(player_states) != 2:
            return {0: 0.0, 1: 0.0}

        nearest_idx = int(np.argmin([state["dist_to_ball"] for state in player_states]))
        support_idx = 1 - nearest_idx
        nearest = player_states[nearest_idx]
        support = player_states[support_idx]

        shaped = {0: 0.0, 1: 0.0}
        shaped[0] += progress_reward
        shaped[1] += progress_reward

        # The closer player should actively contest/control the ball.
        contest_reward = 0.006 * self._exp_distance_reward(nearest["dist_to_ball"], 2.2)
        shaped[nearest["player_id"]] += contest_reward

        # Reward at least one defender being between ball and own goal in own half.
        defenders_between = 0
        for state in player_states:
            player_x, player_z = state["pred_pos"]
            between_goal_and_ball = OWN_GOAL[0] <= player_x <= pred_ball_x
            aligned_with_ball_lane = abs(player_z - pred_ball_z) <= 2.5
            if between_goal_and_ball and aligned_with_ball_lane:
                defenders_between += 1

        if pred_ball_x < 0.0:
            coverage_bonus = 0.004 if defenders_between > 0 else -0.004
            shaped[0] += coverage_bonus
            shaped[1] += coverage_bonus

        # Role-specific support shaping inspired by get-open / cover behavior.
        support_x, support_z = support["pred_pos"]
        nearest_x, nearest_z = nearest["pred_pos"]

        if pred_ball_x >= 0.0:
            # Attacking phase: support player stays slightly behind the ball with useful width.
            behind_ball = pred_ball_x - 5.0 <= support_x <= pred_ball_x - 0.5
            lateral_target = 2.5
            lateral_gap = abs(support_z - pred_ball_z)
            width_reward = 0.003 * max(0.0, 1.0 - abs(lateral_gap - lateral_target) / 2.5)
            support_reward = width_reward if behind_ball else -0.003
            shaped[support["player_id"]] += support_reward
        else:
            # Defensive phase: support player should cover the goal lane from behind the ball.
            between_goal_and_ball = OWN_GOAL[0] <= support_x <= pred_ball_x
            lane_alignment = 0.0035 * self._exp_distance_reward(abs(support_z - pred_ball_z), 1.8)
            shaped[support["player_id"]] += lane_alignment if between_goal_and_ball else -0.0035

        # Mild anti-crowding term: don't have both agents collapse onto the same point.
        teammate_gap = float(np.linalg.norm(nearest["pred_pos"] - support["pred_pos"]))
        if teammate_gap < 1.2:
            crowd_penalty = 0.0035 * (1.2 - teammate_gap) / 1.2
            shaped[nearest["player_id"]] -= crowd_penalty
            shaped[support["player_id"]] -= crowd_penalty

        # Encourage the support player to be on a different lateral line than the ball winner.
        lateral_separation = abs(support_z - nearest_z)
        shaped[support["player_id"]] += 0.002 * min(lateral_separation / 2.5, 1.0)

        return shaped


class BlueTeamVsBaselineShapedEnv(MultiAgentEnv):
    def __init__(self, env_config=None):
        env_config = dict(env_config or {})
        self.base_env = soccer_twos.make(
            variation=EnvType.multiagent_player,
            worker_id=env_config.get("worker_id", 0),
        )
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        self.last_obs = None
        self.baseline_actor = TorchPolicyActor(load_baseline_model())
        self.shaping = BaselineShapingHelper()

        # Load obs normalisation stats from bc_obs_0 checkpoint
        bc_path = Path(env_config.get("bc_checkpoint_path", str(BC_CHECKPOINT_PATH)))
        if bc_path.exists():
            bc_data = torch.load(bc_path, map_location="cpu")
            self.obs_mean = bc_data.get("obs_mean", None)
            self.obs_std  = bc_data.get("obs_std",  None)
        else:
            self.obs_mean = self.obs_std = None

    def _norm(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_mean is not None:
            return (obs - self.obs_mean) / self.obs_std
        return obs

    def reset(self):
        self.shaping.reset()
        self.last_obs = self.base_env.reset()
        return {0: self._norm(self.last_obs[0]), 1: self._norm(self.last_obs[1])}

    def step(self, action_dict):
        env_actions = {
            0: action_dict[0],
            1: action_dict[1],
        }
        # Baseline receives raw (unnormalised) obs — it was trained on raw obs
        orange_obs = {2: self.last_obs[2], 3: self.last_obs[3]}
        env_actions.update(self.baseline_actor.act(orange_obs))

        obs, reward, done, info = self.base_env.step(env_actions)
        self.last_obs = obs
        shaped = self.shaping.shape_rewards(info)
        return (
            {0: self._norm(obs[0]), 1: self._norm(obs[1])},
            {
                0: reward[0] + shaped.get(0, 0.0),
                1: reward[1] + shaped.get(1, 0.0),
            },
            {0: done["__all__"], 1: done["__all__"], "__all__": done["__all__"]},
            {0: info[0], 1: info[1]},
        )

    def close(self):
        self.base_env.close()


class BCInitPlayerModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Match bc_obs_0 architecture: [512, 512]
        self.hidden1 = nn.Linear(int(np.product(obs_space.shape)), 512)
        self.hidden2 = nn.Linear(512, 512)
        self.logits = nn.Linear(512, num_outputs)
        self.value_branch = nn.Linear(512, 1)
        self._value_out = None

        bc_path = model_config.get("custom_model_config", {}).get("bc_checkpoint_path")
        if bc_path and Path(bc_path).exists():
            payload = torch.load(bc_path, map_location="cpu")
            sd = payload["state_dict"]   # bc_obs_0 format
            # Map bc_obs_0 keys → model keys
            self.hidden1.weight.data.copy_(sd["shared.0.weight"])
            self.hidden1.bias.data.copy_(sd["shared.0.bias"])
            self.hidden2.weight.data.copy_(sd["shared.2.weight"])
            self.hidden2.bias.data.copy_(sd["shared.2.bias"])
            # Concatenate 3 branch heads → single logits layer [9, 512]
            logit_w = torch.cat([sd["heads.0.weight"], sd["heads.1.weight"], sd["heads.2.weight"]], dim=0)
            logit_b = torch.cat([sd["heads.0.bias"],   sd["heads.1.bias"],   sd["heads.2.bias"]],   dim=0)
            self.logits.weight.data.copy_(logit_w)
            self.logits.bias.data.copy_(logit_b)

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs_flat"].float()
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        self._value_out = self.value_branch(x).squeeze(1)
        return self.logits(x), state

    def value_function(self):
        return self._value_out


def policy_mapping_fn(agent_id, *args, **kwargs):
    return "default"


def create_env(env_config=None):
    raw_config = env_config or {}
    worker_id = 0
    if hasattr(raw_config, "worker_index"):
        num_envs = raw_config.get("num_envs_per_worker", 1)
        worker_id = raw_config.worker_index * num_envs + raw_config.vector_index
    config_dict = dict(raw_config)
    config_dict["worker_id"] = worker_id
    return BlueTeamVsBaselineShapedEnv(config_dict)


if __name__ == "__main__":
    args = parse_args()
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [project_dir, os.environ.get("PYTHONPATH", "")]
    )
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_SYNCER"] = "1"

    if getattr(model_catalog_module, "tf", None) is None:
        class _DummyKeras:
            class Model:
                pass

        class _DummyTF:
            keras = _DummyKeras

        model_catalog_module.tf = _DummyTF()

    ray.init(num_gpus=0, include_dashboard=False)

    tune.registry.register_env("BlueTeamVsBaselineShaped", create_env)
    ModelCatalog.register_custom_model("bc_init_player_model", BCInitPlayerModel)

    temp_env = create_env({"num_envs_per_worker": 1})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name=args.experiment_name,
        config={
            "num_gpus": 0,
            "num_workers": args.num_workers,
            "num_envs_per_worker": args.num_envs_per_worker,
            "log_level": "WARN",
            "framework": "torch",
            "lr": args.lr,
            "clip_param": args.clip_param,
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["default"],
            },
            "env": "BlueTeamVsBaselineShaped",
            "env_config": {
                "num_envs_per_worker": args.num_envs_per_worker,
                "bc_checkpoint_path": str(Path(args.bc_checkpoint).resolve()),
            },
            "model": {
                "custom_model": "bc_init_player_model",
                "custom_model_config": {
                    "bc_checkpoint_path": str(Path(args.bc_checkpoint).resolve()),
                },
            },
            "rollout_fragment_length": args.rollout_fragment_length,
            "train_batch_size": args.train_batch_size,
        },
        stop={
            "timesteps_total": args.timesteps_total,
        },
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        callbacks=[CSVLoggerCallback(), JsonLoggerCallback()],
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    if best_trial is None and analysis.trials:
        best_trial = analysis.trials[0]
    print(f"Best trial: {best_trial}")

    best_checkpoint = None
    latest_checkpoint = None
    if best_trial is not None:
        checkpoint_dirs = sorted(Path(best_trial.logdir).glob("checkpoint_*"))
        if checkpoint_dirs:
            latest_checkpoint = str(checkpoint_dirs[-1])
        try:
            best_checkpoint = analysis.get_best_checkpoint(
                trial=best_trial, metric="episode_reward_mean", mode="max"
            )
        except ValueError:
            best_checkpoint = latest_checkpoint
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Latest checkpoint: {latest_checkpoint}")
    ray.shutdown()
