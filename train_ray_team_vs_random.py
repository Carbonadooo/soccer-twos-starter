import os
from pathlib import Path

import ray
from ray import tune
from ray.tune.logger import CSVLoggerCallback, JsonLoggerCallback
from soccer_twos import EnvType

from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 4


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [project_dir, os.environ.get("PYTHONPATH", "")]
    )
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_SYNCER"] = "1"

    ray.init(num_gpus=0, include_dashboard=False)

    tune.registry.register_env("Soccer", create_rllib_env)

    analysis = tune.run(
        "PPO",
        name="PPO_1",
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
            "train_batch_size": 24000,
        },
        stop={
            "timesteps_total": 3600000,
            # "time_total_s": 14400, # 4h
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        callbacks=[CSVLoggerCallback(), JsonLoggerCallback()],
        # restore="./ray_results/PPO_selfplay_1/PPO_Soccer_ID/checkpoint_00X/checkpoint-X",
    )

    # Prefer the best trial by reward, but fall back gracefully when reward is NaN.
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
