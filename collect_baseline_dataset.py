import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

if not hasattr(np, "bool"):
    np.bool = bool

import soccer_twos
from soccer_twos import EnvType

from imitation_player_utils import BASELINE_CHECKPOINT_PATH, TorchPolicyActor, load_baseline_model


def parse_args():
    parser = ArgumentParser(description="Collect imitation dataset from the CEIA baseline policy.")
    parser.add_argument("--episodes", type=int, default=80, help="Number of episodes to collect.")
    parser.add_argument(
        "--output-dir",
        default="bc_data/baseline_selfplay",
        help="Directory to save dataset files into.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default=str(BASELINE_CHECKPOINT_PATH),
        help="Path to the baseline RLlib checkpoint file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher = TorchPolicyActor(load_baseline_model(Path(args.teacher_checkpoint)))
    env = soccer_twos.make(variation=EnvType.multiagent_player)

    observations = []
    actions = []
    episode_lengths = []
    episode_team_rewards = []

    for episode_idx in range(args.episodes):
        obs = env.reset()
        done = {"__all__": False}
        episode_steps = 0
        team0_return = 0.0
        while not done["__all__"]:
            action = teacher.act(obs)
            for player_id, player_obs in obs.items():
                observations.append(np.asarray(player_obs, dtype=np.float32))
                actions.append(np.asarray(action[player_id], dtype=np.int64))

            obs, reward, done, _info = env.step(action)
            team0_return += float(reward[0] + reward[1])
            episode_steps += 1

        episode_lengths.append(episode_steps)
        episode_team_rewards.append(team0_return)
        if (episode_idx + 1) % 10 == 0:
            print(
                f"Collected {episode_idx + 1}/{args.episodes} episodes | "
                f"mean len={np.mean(episode_lengths):.1f} | "
                f"mean team0 reward={np.mean(episode_team_rewards):.3f}"
            )

    env.close()

    obs_array = np.stack(observations, axis=0)
    action_array = np.stack(actions, axis=0)
    np.savez_compressed(output_dir / "dataset.npz", observations=obs_array, actions=action_array)

    metadata = {
        "episodes": args.episodes,
        "samples": int(obs_array.shape[0]),
        "observation_shape": list(obs_array.shape),
        "action_shape": list(action_array.shape),
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "mean_team0_reward": float(np.mean(episode_team_rewards)) if episode_team_rewards else 0.0,
        "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved dataset to {output_dir.resolve()}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
