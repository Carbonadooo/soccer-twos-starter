"""
Phase 2: RL fine-tuning starting from BC (behavioral cloning) weights.

Uses multiagent self-play so the agent improves beyond the baseline.
BC weights are loaded into the policy before any RL training starts.

Run after distill_baseline.py:
    python train_rl_finetune.py
"""

import os

import numpy as np
import ray
import torch
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import EnvType

from utils import create_rllib_env

BC_CKPT          = "bc_agent/checkpoint.pth"
SAVE_DIR         = "./ray_results/PPO_bc_finetune"
NUM_ENVS_PER_WORKER = 3
MAX_TIMESTEPS    = 5_000_000
SAVE_EVERY_ITERS = 200


def load_bc_weights(policy, bc_ckpt_path: str):
    """
    Map BCPolicy weights → RLLib FullyConnectedNetwork weights.

    BC layout:
        shared.0  = Linear(336, 512)   → _hidden_layers.0._model.0
        shared.2  = Linear(512, 512)   → _hidden_layers.1._model.0
        heads.0/1/2 = Linear(512, 3)   → _logits._model.0  (concatenated → 9)

    Value branch keeps random init (BC has no critic).
    """
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
    print("BC weights loaded into RLLib policy.")


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [project_dir, os.environ.get("PYTHONPATH", "")]
    )

    ray.init(num_gpus=0, include_dashboard=False)

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env({"variation": EnvType.multiagent_player})
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
        },
        "model": {
            "vf_share_layers": True,
            "fcnet_hiddens": [512, 512],
        },
    }

    trainer = PPOTrainer(config=config, env="Soccer")

    # Inject BC weights before any RL updates
    load_bc_weights(trainer.get_policy("default"), BC_CKPT)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_reward = float("-inf")
    best_ckpt   = None

    print(f"\nStarting RL fine-tuning (target: {MAX_TIMESTEPS:,} steps)...\n")

    i = 0
    while True:
        i += 1
        result  = trainer.train()
        ts      = result["timesteps_total"]
        reward  = result.get("episode_reward_mean", float("nan"))

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
    print(f"Final checkpoint : {final_ckpt}")
    print(f"Best  checkpoint : {best_ckpt}  (reward={best_reward:.3f})")
