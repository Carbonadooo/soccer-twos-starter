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
BC_CHECKPOINT_PATH = Path("bc_results/baseline_bc/best.pt")
FIELD_HALF_LENGTH = 14.0
PREDICTION_HORIZON = 0.25


def parse_args():
    parser = ArgumentParser(
        description="Finetune a BC-initialized player policy against the baseline with reward shaping."
    )
    parser.add_argument("--timesteps-total", type=int, default=600000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-envs-per-worker", type=int, default=NUM_ENVS_PER_WORKER)
    parser.add_argument("--rollout-fragment-length", type=int, default=500)
    parser.add_argument("--train-batch-size", type=int, default=8000)
    parser.add_argument("--checkpoint-freq", type=int, default=20)
    parser.add_argument("--bc-checkpoint", default=str(BC_CHECKPOINT_PATH))
    parser.add_argument("--experiment-name", default="PPO_bc_finetune_vs_baseline_shaped")
    return parser.parse_args()


class BaselineShapingHelper:
    def __init__(self):
        self._prev_ball_potential = None

    def reset(self):
        self._prev_ball_potential = None

    def shape_rewards(self, info):
        if not info or 0 not in info or 1 not in info:
            return {0: 0.0, 1: 0.0}

        controlled_players = [info[0], info[1]]
        try:
            ball_pos = np.asarray(
                controlled_players[0]["ball_info"]["position"], dtype=np.float32
            )
            ball_vel = np.asarray(
                controlled_players[0]["ball_info"]["velocity"], dtype=np.float32
            )
        except (KeyError, TypeError, ValueError):
            return {0: 0.0, 1: 0.0}

        pred_ball = ball_pos + PREDICTION_HORIZON * ball_vel
        pred_ball_x = float(np.clip(pred_ball[0], -FIELD_HALF_LENGTH, FIELD_HALF_LENGTH))
        pred_ball_z = float(pred_ball[1])

        # Team-level progress shaping shared by both trained players.
        progress_reward = 0.0
        ball_potential = pred_ball_x / FIELD_HALF_LENGTH
        if self._prev_ball_potential is not None:
            progress_reward = 0.02 * float(ball_potential - self._prev_ball_potential)
        self._prev_ball_potential = ball_potential

        shaped = {}
        for player_id, player_info in zip([0, 1], controlled_players):
            try:
                player_pos = np.asarray(
                    player_info["player_info"]["position"], dtype=np.float32
                )
                player_vel = np.asarray(
                    player_info["player_info"].get("velocity", [0.0, 0.0]),
                    dtype=np.float32,
                )
            except (KeyError, TypeError, ValueError):
                shaped[player_id] = 0.0
                continue

            pred_player = player_pos + PREDICTION_HORIZON * player_vel
            pred_player_x = float(
                np.clip(pred_player[0], -FIELD_HALF_LENGTH, FIELD_HALF_LENGTH)
            )
            pred_player_z = float(pred_player[1])
            lateral_alignment = float(np.exp(-abs(pred_player_z - pred_ball_z) / 3.0))

            lane_reward = 0.0
            if pred_ball_x < 0.0:
                in_defensive_lane = -FIELD_HALF_LENGTH <= pred_player_x <= pred_ball_x
                lane_reward = (
                    0.004 * lateral_alignment
                    if in_defensive_lane
                    else -0.004 * lateral_alignment
                )
            else:
                in_attacking_lane = pred_player_x <= pred_ball_x <= FIELD_HALF_LENGTH
                lane_reward = (
                    0.004 * lateral_alignment
                    if in_attacking_lane
                    else -0.004 * lateral_alignment
                )

            shaped[player_id] = float(progress_reward + lane_reward)

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

    def reset(self):
        self.shaping.reset()
        self.last_obs = self.base_env.reset()
        return {0: self.last_obs[0], 1: self.last_obs[1]}

    def step(self, action_dict):
        env_actions = {
            0: action_dict[0],
            1: action_dict[1],
        }
        orange_obs = {2: self.last_obs[2], 3: self.last_obs[3]}
        env_actions.update(self.baseline_actor.act(orange_obs))

        obs, reward, done, info = self.base_env.step(env_actions)
        self.last_obs = obs
        shaped = self.shaping.shape_rewards(info)
        return (
            {0: obs[0], 1: obs[1]},
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

        self.hidden1 = nn.Linear(int(np.product(obs_space.shape)), 256)
        self.hidden2 = nn.Linear(256, 256)
        self.logits = nn.Linear(256, num_outputs)
        self.value_branch = nn.Linear(256, 1)
        self._value_out = None

        bc_path = model_config.get("custom_model_config", {}).get("bc_checkpoint_path")
        if bc_path and Path(bc_path).exists():
            payload = torch.load(bc_path, map_location="cpu")
            state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
            self.hidden1.weight.data.copy_(state_dict["hidden1.weight"])
            self.hidden1.bias.data.copy_(state_dict["hidden1.bias"])
            self.hidden2.weight.data.copy_(state_dict["hidden2.weight"])
            self.hidden2.bias.data.copy_(state_dict["hidden2.bias"])
            self.logits.weight.data.copy_(state_dict["logits.weight"])
            self.logits.bias.data.copy_(state_dict["logits.bias"])

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
