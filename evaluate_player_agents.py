import importlib
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

if not hasattr(np, "bool"):
    np.bool = bool

import soccer_twos
from soccer_twos import AgentInterface, EnvType


def parse_args():
    parser = ArgumentParser(description="Evaluate two player-agent packages headlessly.")
    parser.add_argument("--agent-1", required=True, help="Blue team agent module name.")
    parser.add_argument("--agent-2", required=True, help="Orange team agent module name.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save evaluation metrics as JSON.",
    )
    return parser.parse_args()


def load_agent_class(module_name: str):
    module = importlib.import_module(module_name)
    for value in module.__dict__.values():
        if isinstance(value, type) and issubclass(value, AgentInterface) and value is not AgentInterface:
            return value
    raise ValueError(f"Could not find AgentInterface subclass in module {module_name}")


def main():
    args = parse_args()
    env = soccer_twos.make(variation=EnvType.multiagent_player)

    agent_1_cls = load_agent_class(args.agent_1)
    agent_2_cls = load_agent_class(args.agent_2)
    agent_1 = agent_1_cls(env)
    agent_2 = agent_2_cls(env)

    wins_1 = 0
    wins_2 = 0
    draws = 0
    team1_rewards = []
    team2_rewards = []
    episode_lengths = []

    for episode_idx in range(args.episodes):
        obs = env.reset()
        done = {"__all__": False}
        episode_steps = 0
        team1_return = 0.0
        team2_return = 0.0

        while not done["__all__"]:
            blue_obs = {0: obs[0], 1: obs[1]}
            orange_obs = {2: obs[2], 3: obs[3]}
            actions = {}
            actions.update(agent_1.act(blue_obs))
            actions.update(agent_2.act(orange_obs))

            obs, reward, done, _info = env.step(actions)
            team1_return += float(reward[0] + reward[1])
            team2_return += float(reward[2] + reward[3])
            episode_steps += 1

        team1_rewards.append(team1_return)
        team2_rewards.append(team2_return)
        episode_lengths.append(episode_steps)

        if team1_return > team2_return:
            wins_1 += 1
        elif team2_return > team1_return:
            wins_2 += 1
        else:
            draws += 1

        print(
            f"Episode {episode_idx + 1}/{args.episodes} | "
            f"blue={team1_return:.3f} orange={team2_return:.3f} len={episode_steps}"
        )

    env.close()

    metrics = {
        "agent_1": args.agent_1,
        "agent_2": args.agent_2,
        "episodes": args.episodes,
        "agent_1_win_rate": wins_1 / args.episodes,
        "agent_2_win_rate": wins_2 / args.episodes,
        "draw_rate": draws / args.episodes,
        "agent_1_mean_team_reward": float(np.mean(team1_rewards)),
        "agent_2_mean_team_reward": float(np.mean(team2_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
    }
    print(json.dumps(metrics, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
