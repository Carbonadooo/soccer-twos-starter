import os
import pickle
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional

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


NUM_ENVS_PER_WORKER = 2
BC_CHECKPOINT_PATH = Path("bc_agent/checkpoint.pth")
RESTORE_CHECKPOINT_PATH = Path(
    "ray_results"
    "/PPO_bc_finetune_vs_baseline_shaped_lr2e5_clip01_3p6M"
    "/PPO_BlueTeamVsBaselineShaped_99e44_00000_0_2026-04-24_02-55-38"
    "/checkpoint_000360/checkpoint-360"
)
HISTORY_TRIAL_DIR = RESTORE_CHECKPOINT_PATH.parents[1]
BASELINE_CHECKPOINT_PATH = Path(
    "ceia_baseline_agent"
    "/ray_results/PPO_selfplay_twos/PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02"
    "/checkpoint_002449/checkpoint-2449"
)

FIELD_HALF_LENGTH = 14.0
PREDICTION_HORIZON = 0.25
GOAL_Z = 0.0
OWN_GOAL = np.asarray([-FIELD_HALF_LENGTH, GOAL_Z], dtype=np.float32)


def parse_args():
    parser = ArgumentParser(
        description="Continue shaped PPO finetuning against a sampled opponent pool."
    )
    parser.add_argument("--timesteps-total", type=int, default=7200000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-envs-per-worker", type=int, default=NUM_ENVS_PER_WORKER)
    parser.add_argument("--rollout-fragment-length", type=int, default=500)
    parser.add_argument("--train-batch-size", type=int, default=8000)
    parser.add_argument("--checkpoint-freq", type=int, default=20)
    parser.add_argument("--bc-checkpoint", default=str(BC_CHECKPOINT_PATH))
    parser.add_argument("--restore-checkpoint", default=str(RESTORE_CHECKPOINT_PATH))
    parser.add_argument(
        "--restore-trial-dir",
        default="",
        help="Optional trial directory; if set, automatically restore from its latest checkpoint.",
    )
    parser.add_argument("--history-trial-dir", default=str(HISTORY_TRIAL_DIR))
    parser.add_argument("--num-history-opponents", type=int, default=6)
    parser.add_argument("--baseline-prob", type=float, default=0.5)
    parser.add_argument("--history-prob", type=float, default=0.3)
    parser.add_argument("--random-prob", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--clip-param", type=float, default=0.1)
    parser.add_argument(
        "--experiment-name",
        default="PPO_bc_finetune_vs_opponent_pool",
    )
    return parser.parse_args()


def discover_checkpoint_files(trial_dir: Path) -> List[Path]:
    checkpoint_files = []
    for checkpoint_dir in sorted(trial_dir.glob("checkpoint_*")):
        if not checkpoint_dir.is_dir():
            continue
        candidates = sorted(
            [
                path
                for path in checkpoint_dir.glob("checkpoint-*")
                if path.is_file() and not str(path).endswith(".tune_metadata")
            ]
        )
        checkpoint_files.extend(candidates)
    return checkpoint_files


def find_latest_checkpoint(trial_dir: Path) -> Optional[Path]:
    checkpoint_files = discover_checkpoint_files(trial_dir)
    if not checkpoint_files:
        return None
    return max(checkpoint_files, key=os.path.getmtime)


class SimplePolicyNet(nn.Module):
    def __init__(self, obs_size: int = 336, hidden_size: int = 256, action_logits_size: int = 9):
        super().__init__()
        self.hidden1 = nn.Linear(obs_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.logits = nn.Linear(hidden_size, action_logits_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.hidden1(obs))
        x = torch.relu(self.hidden2(x))
        return self.logits(x)


class TorchPolicyActor:
    def __init__(self, model: nn.Module, device: str = "cpu"):
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
            logits = self.model(obs_tensor).view(-1, 3, 3)
            actions = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int64)
        return {player_id: actions[idx] for idx, player_id in enumerate(ordered_ids)}


class RandomActor:
    def __init__(self, action_space: gym.Space):
        self.action_space = action_space

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        return {player_id: self.action_space.sample() for player_id in observation}


def _load_rllib_policy_state(checkpoint_path: Path):
    with checkpoint_path.open("rb") as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)
    worker_state = pickle.loads(checkpoint["worker"])
    return worker_state["state"]["default"]


def load_checkpoint_actor(checkpoint_path: Path) -> TorchPolicyActor:
    policy_state = _load_rllib_policy_state(Path(checkpoint_path))
    if "hidden1.weight" in policy_state:
        hidden_size = int(policy_state["hidden1.weight"].shape[0])
        model = SimplePolicyNet(obs_size=336, hidden_size=hidden_size, action_logits_size=9)
        model.hidden1.weight.data.copy_(torch.from_numpy(policy_state["hidden1.weight"]))
        model.hidden1.bias.data.copy_(torch.from_numpy(policy_state["hidden1.bias"]))
        model.hidden2.weight.data.copy_(torch.from_numpy(policy_state["hidden2.weight"]))
        model.hidden2.bias.data.copy_(torch.from_numpy(policy_state["hidden2.bias"]))
        model.logits.weight.data.copy_(torch.from_numpy(policy_state["logits.weight"]))
        model.logits.bias.data.copy_(torch.from_numpy(policy_state["logits.bias"]))
        model.eval()
        return TorchPolicyActor(model)

    hidden_size = int(policy_state["_hidden_layers.0._model.0.weight"].shape[0])
    model = SimplePolicyNet(obs_size=336, hidden_size=hidden_size, action_logits_size=9)
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
    return TorchPolicyActor(model)


def list_history_checkpoints(trial_dir: Path, exclude_checkpoint: Optional[Path], limit: int) -> List[Path]:
    files = [path.resolve() for path in discover_checkpoint_files(trial_dir)]

    if exclude_checkpoint is not None:
        exclude_checkpoint = exclude_checkpoint.resolve()
        files = [path for path in files if path != exclude_checkpoint]

    if limit <= 0 or limit >= len(files):
        return files
    if limit == 1:
        return [files[-1]]

    positions = np.linspace(0, len(files) - 1, num=limit)
    indices = sorted({int(round(float(pos))) for pos in positions})
    if len(indices) < limit:
        for idx in range(len(files)):
            if idx not in indices:
                indices.append(idx)
            if len(indices) == limit:
                break
    indices = sorted(indices)[:limit]
    return [files[idx] for idx in indices]


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

        contest_reward = 0.006 * self._exp_distance_reward(nearest["dist_to_ball"], 2.2)
        shaped[nearest["player_id"]] += contest_reward

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

        support_x, support_z = support["pred_pos"]
        nearest_z = nearest["pred_pos"][1]

        if pred_ball_x >= 0.0:
            behind_ball = pred_ball_x - 5.0 <= support_x <= pred_ball_x - 0.5
            lateral_target = 2.5
            lateral_gap = abs(support_z - pred_ball_z)
            width_reward = 0.003 * max(0.0, 1.0 - abs(lateral_gap - lateral_target) / 2.5)
            support_reward = width_reward if behind_ball else -0.003
            shaped[support["player_id"]] += support_reward
        else:
            between_goal_and_ball = OWN_GOAL[0] <= support_x <= pred_ball_x
            lane_alignment = 0.0035 * self._exp_distance_reward(abs(support_z - pred_ball_z), 1.8)
            shaped[support["player_id"]] += lane_alignment if between_goal_and_ball else -0.0035

        teammate_gap = float(np.linalg.norm(nearest["pred_pos"] - support["pred_pos"]))
        if teammate_gap < 1.2:
            crowd_penalty = 0.0035 * (1.2 - teammate_gap) / 1.2
            shaped[nearest["player_id"]] -= crowd_penalty
            shaped[support["player_id"]] -= crowd_penalty

        lateral_separation = abs(support_z - nearest_z)
        shaped[support["player_id"]] += 0.002 * min(lateral_separation / 2.5, 1.0)
        return shaped


class BlueTeamVsOpponentPoolEnv(MultiAgentEnv):
    def __init__(self, env_config=None):
        env_config = dict(env_config or {})
        self.base_env = soccer_twos.make(
            variation=EnvType.multiagent_player,
            worker_id=env_config.get("worker_id", 0),
        )
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        self.last_obs = None
        self.shaping = BaselineShapingHelper()
        self.random_state = random.Random(env_config.get("seed", 0))

        self.baseline_prob = float(env_config.get("baseline_prob", 0.5))
        self.history_prob = float(env_config.get("history_prob", 0.3))
        self.random_prob = float(env_config.get("random_prob", 0.2))

        self.baseline_actor = load_checkpoint_actor(Path(env_config["baseline_checkpoint_path"]))
        self.history_checkpoint_paths = [
            Path(path) for path in env_config.get("history_checkpoint_paths", [])
        ]
        self.history_actors = [load_checkpoint_actor(path) for path in self.history_checkpoint_paths]
        self.random_actor = RandomActor(self.action_space)
        self.current_opponent = self.baseline_actor
        self.current_opponent_name = "baseline"

    def _sample_opponent(self):
        probs = np.asarray(
            [self.baseline_prob, self.history_prob, self.random_prob], dtype=np.float64
        )
        probs = probs / probs.sum()
        opponent_type = self.random_state.choices(
            ["baseline", "history", "random"],
            weights=probs.tolist(),
            k=1,
        )[0]

        if opponent_type == "history" and self.history_actors:
            idx = self.random_state.randrange(len(self.history_actors))
            self.current_opponent = self.history_actors[idx]
            self.current_opponent_name = f"history:{self.history_checkpoint_paths[idx].name}"
        elif opponent_type == "random":
            self.current_opponent = self.random_actor
            self.current_opponent_name = "random"
        else:
            self.current_opponent = self.baseline_actor
            self.current_opponent_name = "baseline"

    def reset(self):
        self._sample_opponent()
        self.shaping.reset()
        self.last_obs = self.base_env.reset()
        return {0: self.last_obs[0], 1: self.last_obs[1]}

    def step(self, action_dict):
        env_actions = {
            0: action_dict[0],
            1: action_dict[1],
        }
        orange_obs = {2: self.last_obs[2], 3: self.last_obs[3]}
        env_actions.update(self.current_opponent.act(orange_obs))

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
            {
                0: {"env_info": info[0], "opponent_type": self.current_opponent_name},
                1: {"env_info": info[1], "opponent_type": self.current_opponent_name},
            },
        )

    def close(self):
        self.base_env.close()


class BCInitPlayerModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        hidden_size = 512
        self.hidden1 = nn.Linear(int(np.product(obs_space.shape)), hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.logits = nn.Linear(hidden_size, num_outputs)
        self.value_branch = nn.Linear(hidden_size, 1)
        self._value_out = None

        bc_path = model_config.get("custom_model_config", {}).get("bc_checkpoint_path")
        if bc_path and Path(bc_path).exists():
            payload = torch.load(bc_path, map_location="cpu")
            state_dict = payload["state_dict"] if "state_dict" in payload else payload
            self.hidden1.weight.data.copy_(state_dict["shared.0.weight"])
            self.hidden1.bias.data.copy_(state_dict["shared.0.bias"])
            self.hidden2.weight.data.copy_(state_dict["shared.2.weight"])
            self.hidden2.bias.data.copy_(state_dict["shared.2.bias"])
            head_weights = [state_dict[f"heads.{branch_idx}.weight"] for branch_idx in range(3)]
            head_biases = [state_dict[f"heads.{branch_idx}.bias"] for branch_idx in range(3)]
            self.logits.weight.data.copy_(torch.cat(head_weights, dim=0))
            self.logits.bias.data.copy_(torch.cat(head_biases, dim=0))

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
    return BlueTeamVsOpponentPoolEnv(config_dict)


if __name__ == "__main__":
    args = parse_args()
    restore_trial_dir = Path(args.restore_trial_dir).resolve() if args.restore_trial_dir else None
    if restore_trial_dir:
        restore_checkpoint = find_latest_checkpoint(restore_trial_dir)
        if restore_checkpoint is None:
            raise SystemExit(f"No checkpoint-* files found under restore trial dir: {restore_trial_dir}")
    else:
        restore_checkpoint = Path(args.restore_checkpoint).resolve()

    history_trial_dir = (
        restore_trial_dir
        if restore_trial_dir is not None
        else Path(args.history_trial_dir).resolve()
    )
    history_checkpoint_paths = list_history_checkpoints(
        history_trial_dir,
        exclude_checkpoint=restore_checkpoint,
        limit=args.num_history_opponents,
    )

    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = os.pathsep.join([project_dir, os.environ.get("PYTHONPATH", "")])
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

    tune.registry.register_env("BlueTeamVsOpponentPool", create_env)
    ModelCatalog.register_custom_model("bc_init_player_model", BCInitPlayerModel)

    temp_env = create_env(
        {
            "num_envs_per_worker": 1,
            "baseline_checkpoint_path": str(BASELINE_CHECKPOINT_PATH.resolve()),
            "history_checkpoint_paths": [str(path) for path in history_checkpoint_paths],
            "baseline_prob": args.baseline_prob,
            "history_prob": args.history_prob,
            "random_prob": args.random_prob,
        }
    )
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name=args.experiment_name,
        restore=str(restore_checkpoint),
        config={
            "num_gpus": 0,
            "num_workers": args.num_workers,
            "num_envs_per_worker": args.num_envs_per_worker,
            "log_level": "WARN",
            "framework": "torch",
            "lr": args.lr,
            "clip_param": args.clip_param,
            "entropy_coeff": 0.001,
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["default"],
            },
            "env": "BlueTeamVsOpponentPool",
            "env_config": {
                "num_envs_per_worker": args.num_envs_per_worker,
                "baseline_checkpoint_path": str(BASELINE_CHECKPOINT_PATH.resolve()),
                "history_checkpoint_paths": [str(path) for path in history_checkpoint_paths],
                "baseline_prob": args.baseline_prob,
                "history_prob": args.history_prob,
                "random_prob": args.random_prob,
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
        stop={"timesteps_total": args.timesteps_total},
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
    print(f"History opponents: {[path.name for path in history_checkpoint_paths]}")
    ray.shutdown()
