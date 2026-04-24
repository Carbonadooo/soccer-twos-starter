import os
from pathlib import Path

import gym
import numpy as np
import ray
from ray import tune
from ray.tune.logger import CSVLoggerCallback, JsonLoggerCallback
from soccer_twos import EnvType

import soccer_twos
from utils import RLLibWrapper


FIELD_HALF_LENGTH = 14.0
PREDICTION_HORIZON = 0.25
NUM_ENVS_PER_WORKER = 2


class TeamVsRandomWrapper(gym.Wrapper):
    """
    Team-vs-random wrapper that preserves both controlled players' info blocks.

    This mirrors soccer_twos.wrappers.TeamVsPolicyWrapper closely, but returns
    a team-level info dict containing data for both controlled players so reward
    shaping can reason about team positioning instead of only player 0.
    """

    def __init__(self, env, opponent_policy=None):
        super().__init__(env)
        self.env = env
        self.last_obs = None

        self.observation_space = gym.spaces.Box(
            0,
            1,
            dtype=np.float32,
            shape=(env.observation_space.shape[0] * 2,),
        )
        if isinstance(env.action_space, gym.spaces.MultiDiscrete):
            self.action_space = gym.spaces.MultiDiscrete(
                np.repeat(env.action_space.nvec, 2)
            )
            self.action_space_n = len(env.action_space.nvec)
        else:
            raise TypeError("Expected a MultiDiscrete action space for team control.")

        if opponent_policy is None:
            self.opponent_policy = lambda *_: self.env.action_space.sample()
        else:
            self.opponent_policy = opponent_policy

    def _preprocess_obs(self, obs):
        self.last_obs = obs
        return np.concatenate((obs[0], obs[1]))

    def reset(self):
        return self._preprocess_obs(self.env.reset())

    def step(self, action):
        env_action = {
            0: action[: self.action_space_n],
            1: action[self.action_space_n :],
            2: self.opponent_policy(self.last_obs[2]),
            3: self.opponent_policy(self.last_obs[3]),
        }
        obs, reward, done, info = self.env.step(env_action)
        team_info = {
            "controlled_players": [info[0], info[1]],
            "opponents": [info[2], info[3]],
        }
        return (
            self._preprocess_obs(obs),
            reward[0] + reward[1],
            done["__all__"],
            team_info,
        )


class TeamVsRandomRewardShapingWrapper(gym.Wrapper):
    """
    Reward shaping for team-vs-random training.

    Design goals:
    - Reward only incremental progress toward the opponent goal.
    - Use velocity-aware predicted positions instead of raw positions.
    - Encourage defensive positioning when the ball is in our half.
    - Encourage attacking support from behind the ball in the opponent half.

    Important:
    - This shaping uses absolute world coordinates.
    - It is intended for the left-side training team used by TeamVsPolicyWrapper.
    - If the trained checkpoint is later deployed on the right side, the agent
      should be wrapped with a coordinate/action conversion layer.
    """

    def __init__(self, env):
        super().__init__(env)
        self._prev_ball_potential = None

    def reset(self):
        self._prev_ball_potential = None
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        shaped_reward = self._shape(info)
        if done:
            self._prev_ball_potential = None
        return obs, reward + shaped_reward, done, info

    def _shape(self, info):
        if not info:
            return 0.0

        controlled_players = info.get("controlled_players")
        if not controlled_players or len(controlled_players) != 2:
            return 0.0

        try:
            ball_pos = np.asarray(
                controlled_players[0]["ball_info"]["position"], dtype=np.float32
            )
            ball_vel = np.asarray(
                controlled_players[0]["ball_info"]["velocity"], dtype=np.float32
            )
        except (KeyError, TypeError, ValueError):
            return 0.0

        pred_ball = ball_pos + PREDICTION_HORIZON * ball_vel
        pred_ball_x = float(np.clip(pred_ball[0], -FIELD_HALF_LENGTH, FIELD_HALF_LENGTH))
        pred_ball_z = float(pred_ball[1])

        reward = 0.0

        # Potential-based shaping: only reward incremental predicted progress.
        ball_potential = pred_ball_x / FIELD_HALF_LENGTH
        if self._prev_ball_potential is not None:
            reward += 0.02 * float(ball_potential - self._prev_ball_potential)
        self._prev_ball_potential = ball_potential

        lane_reward = 0.0
        for player_info in controlled_players:
            try:
                player_pos = np.asarray(
                    player_info["player_info"]["position"], dtype=np.float32
                )
                player_vel = np.asarray(
                    player_info["player_info"].get("velocity", [0.0, 0.0]),
                    dtype=np.float32,
                )
            except (KeyError, TypeError, ValueError):
                continue

            pred_player = player_pos + PREDICTION_HORIZON * player_vel
            pred_player_x = float(
                np.clip(pred_player[0], -FIELD_HALF_LENGTH, FIELD_HALF_LENGTH)
            )
            pred_player_z = float(pred_player[1])
            lateral_alignment = float(np.exp(-abs(pred_player_z - pred_ball_z) / 3.0))

            if pred_ball_x < 0.0:
                # Defensive phase: stand between the ball and our goal at x=-14.
                in_defensive_lane = -FIELD_HALF_LENGTH <= pred_player_x <= pred_ball_x
                lane_reward += (
                    0.004 * lateral_alignment
                    if in_defensive_lane
                    else -0.004 * lateral_alignment
                )
            else:
                # Offensive phase: approach from behind so the ball is between us and the opponent goal.
                in_attacking_lane = pred_player_x <= pred_ball_x <= FIELD_HALF_LENGTH
                lane_reward += (
                    0.004 * lateral_alignment
                    if in_attacking_lane
                    else -0.004 * lateral_alignment
                )

        reward += lane_reward / 2.0

        return float(reward)


def create_shaped_env(env_config: dict = {}):
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    base_env_config = dict(env_config)
    base_env_config.pop("variation", None)
    base_env_config.pop("multiagent", None)
    env = soccer_twos.make(**base_env_config)
    env = TeamVsRandomWrapper(env)
    env = TeamVsRandomRewardShapingWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        return env
    return RLLibWrapper(env)


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [project_dir, os.environ.get("PYTHONPATH", "")]
    )
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_SYNCER"] = "1"

    ray.init(num_gpus=0, include_dashboard=False)

    tune.registry.register_env("Soccer", create_shaped_env)

    analysis = tune.run(
        "PPO",
        name="PPO_team_random_shaped",
        config={
            # system settings
            "num_gpus": 0,
            "num_workers": 12,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "WARN",
            "framework": "torch",
            # RL setup
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512, 512],
            },
            "rollout_fragment_length": 500,
            "train_batch_size": 12000,
        },
        stop={
            "timesteps_total": 1200000,
            # "time_total_s": 14400, # 4h
        },
        checkpoint_freq=20,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        callbacks=[CSVLoggerCallback(), JsonLoggerCallback()],
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    if best_trial is None and analysis.trials:
        best_trial = analysis.trials[0]
        print(
            "episode_reward_mean is not available yet; "
            "falling back to the first completed trial."
        )
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
    print("Done training")
    ray.shutdown()
