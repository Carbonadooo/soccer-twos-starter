import os

import gym
import numpy as np
import ray
from ray import tune
from soccer_twos import EnvType

import soccer_twos
from utils import RLLibWrapper


class RewardShapingWrapper(gym.Wrapper):
    """
    Dense reward shaping on top of sparse goal rewards.

    Requires the env binary to send 345-dim obs (training env does this).
    If info is empty (336-dim env), shaped reward is 0 and training still works.

    Shaped components (all small to not overwhelm goal reward ±1):
      +0.01  ball progress toward opponent goal (x axis, max per step)
      +0.005 agent proximity to ball (exp decay)
      +0.005 ball velocity toward opponent goal
    """

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward + self._shape(info), done, info

    def _shape(self, info):
        if not info:
            return 0.0
        try:
            player_pos = np.array(info["player_info"]["position"])  # [x, z]
            ball_pos   = np.array(info["ball_info"]["position"])    # [x, z]
            ball_vel   = np.array(info["ball_info"]["velocity"])    # [vx, vz]

            # Ball progress toward opponent goal (x=+14); ranges -0.01 to +0.01
            ball_progress = 0.01 * ball_pos[0] / 14.0

            # Agent proximity to ball via exp decay; max ~0.005
            dist = np.linalg.norm(player_pos - ball_pos)
            ball_proximity = 0.005 * np.exp(-dist / 5.0)

            # Ball moving toward opponent goal (positive x velocity)
            ball_direction = 0.005 * float(np.clip(ball_vel[0] / 10.0, -1.0, 1.0))

            return float(ball_progress + ball_proximity + ball_direction)
        except (KeyError, TypeError, IndexError):
            return 0.0


NUM_ENVS_PER_WORKER = 5


def create_shaped_env(env_config: dict = {}):
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    env = RewardShapingWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        return env
    return RLLibWrapper(env)


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [project_dir, os.environ.get("PYTHONPATH", "")]
    )

    ray.init(num_gpus=0, include_dashboard=False)

    tune.registry.register_env("Soccer", create_shaped_env)

    analysis = tune.run(
        "PPO",
        name="PPO_shaped",
        config={
            "num_gpus": 0,
            "num_workers": 8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
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
        },
        stop={
            "timesteps_total": 5000000,
            "time_total_s": 28000,  # 5.8h safety cap
        },
        checkpoint_freq=200,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
