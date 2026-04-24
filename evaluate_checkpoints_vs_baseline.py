import json
import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mlagents_envs.exception import UnityWorkerInUseException
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if not hasattr(np, "bool"):
    np.bool = bool

import soccer_twos
from soccer_twos import EnvType


BASELINE_CHECKPOINT_PATH = Path(
    "ceia_baseline_agent"
    "/ray_results/PPO_selfplay_twos/PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02"
    "/checkpoint_002449/checkpoint-2449"
)


class PlayerPolicyNet(nn.Module):
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


def parse_args():
    parser = ArgumentParser(
        description="Evaluate every RLlib checkpoint in a trial directory against ceia baseline."
    )
    parser.add_argument(
        "--trial-dir",
        required=True,
        help="Trial directory under ray_results containing checkpoint_* subdirectories.",
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument(
        "--num-checkpoints",
        type=int,
        default=0,
        help="If > 0, uniformly sample this many checkpoints (including first/last when possible).",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=40,
        help="Unity worker_id to use for evaluation. Change if another env is already running.",
    )
    parser.add_argument(
        "--worker-id-retries",
        type=int,
        default=20,
        help="How many successive worker_ids to try if the starting one is busy.",
    )
    parser.add_argument(
        "--output-prefix",
        default="vs_ceia_baseline",
        help="Prefix for output CSV/JSON saved into the trial directory.",
    )
    return parser.parse_args()


def load_baseline_actor(checkpoint_path: Path = BASELINE_CHECKPOINT_PATH) -> TorchPolicyActor:
    with checkpoint_path.open("rb") as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)
    worker_state = pickle.loads(checkpoint["worker"])
    policy_state = worker_state["state"]["default"]
    model = PlayerPolicyNet(obs_size=336, hidden_size=256, action_logits_size=9)
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


def load_checkpoint_actor(checkpoint_path: Path) -> TorchPolicyActor:
    with checkpoint_path.open("rb") as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)
    worker_state = pickle.loads(checkpoint["worker"])
    policy_state = worker_state["state"]["default"]

    if "hidden1.weight" in policy_state:
        hidden_size = int(policy_state["hidden1.weight"].shape[0])
        model = PlayerPolicyNet(obs_size=336, hidden_size=hidden_size, action_logits_size=9)
        model.hidden1.weight.data.copy_(torch.from_numpy(policy_state["hidden1.weight"]))
        model.hidden1.bias.data.copy_(torch.from_numpy(policy_state["hidden1.bias"]))
        model.hidden2.weight.data.copy_(torch.from_numpy(policy_state["hidden2.weight"]))
        model.hidden2.bias.data.copy_(torch.from_numpy(policy_state["hidden2.bias"]))
        model.logits.weight.data.copy_(torch.from_numpy(policy_state["logits.weight"]))
        model.logits.bias.data.copy_(torch.from_numpy(policy_state["logits.bias"]))
        model.eval()
        return TorchPolicyActor(model)

    hidden_size = int(policy_state["_hidden_layers.0._model.0.weight"].shape[0])
    model = PlayerPolicyNet(obs_size=336, hidden_size=hidden_size, action_logits_size=9)
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


def discover_checkpoint_files(trial_dir: Path) -> List[Path]:
    checkpoint_files = []
    for checkpoint_dir in sorted(trial_dir.glob("checkpoint_*")):
        if checkpoint_dir.is_dir():
            candidates = sorted(
                [
                    path
                    for path in checkpoint_dir.glob("checkpoint-*")
                    if path.is_file() and not str(path).endswith(".tune_metadata")
                ]
            )
            checkpoint_files.extend(candidates)
    return checkpoint_files


def checkpoint_step(checkpoint_name: str) -> int:
    return int(checkpoint_name.split("-")[-1])


def select_uniform_checkpoints(checkpoint_files: List[Path], num_checkpoints: int) -> List[Path]:
    if num_checkpoints <= 0 or num_checkpoints >= len(checkpoint_files):
        return checkpoint_files
    if num_checkpoints == 1:
        return [checkpoint_files[-1]]

    positions = np.linspace(0, len(checkpoint_files) - 1, num=num_checkpoints)
    indices = []
    seen = set()
    for pos in positions:
        idx = int(round(float(pos)))
        idx = max(0, min(idx, len(checkpoint_files) - 1))
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)

    # If rounding collapsed some indices, fill remaining slots from left to right.
    if len(indices) < num_checkpoints:
        for idx in range(len(checkpoint_files)):
            if idx not in seen:
                indices.append(idx)
                seen.add(idx)
            if len(indices) == num_checkpoints:
                break

    indices = sorted(indices)[:num_checkpoints]
    return [checkpoint_files[idx] for idx in indices]


def plot_results(results: List[Dict[str, float]], output_dir: Path, output_prefix: str):
    if not results:
        return

    checkpoints = [checkpoint_step(row["checkpoint"]) for row in results]
    episodes = [row["episodes"] for row in results]
    win_rates = [row["win_rate"] for row in results]
    wins = [row["win_rate"] * row["episodes"] for row in results]
    mean_rewards = [row["mean_team_reward"] for row in results]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(checkpoints, win_rates, marker="o", linewidth=2)
    axes[0].set_ylabel("Win rate")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Checkpoint vs CEIA baseline")

    axes[1].plot(checkpoints, wins, marker="o", linewidth=2)
    axes[1].set_ylabel(f"Wins / {episodes[0]} eps")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(checkpoints, mean_rewards, marker="o", linewidth=2)
    axes[2].set_ylabel("Mean team reward")
    axes[2].set_xlabel("Checkpoint step")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    combined_path = output_dir / f"{output_prefix}_curves.png"
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(checkpoints, win_rates, marker="o", linewidth=2)
    ax.set_title("Win rate vs CEIA baseline")
    ax.set_ylabel("Win rate")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{output_prefix}_win_rate_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(checkpoints, wins, marker="o", linewidth=2)
    ax.set_title("Wins vs CEIA baseline")
    ax.set_ylabel(f"Wins / {episodes[0]} eps")
    ax.set_xlabel("Checkpoint step")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{output_prefix}_wins_curve.png", dpi=150)
    plt.close(fig)


def evaluate_checkpoint(
    checkpoint_path: Path,
    baseline_actor: TorchPolicyActor,
    episodes: int,
    worker_id: int,
    worker_id_retries: int,
) -> Dict[str, float]:
    env = None
    last_error = None
    for worker_candidate in range(worker_id, worker_id + worker_id_retries):
        try:
            env = soccer_twos.make(
                variation=EnvType.multiagent_player,
                worker_id=worker_candidate,
            )
            worker_id = worker_candidate
            break
        except UnityWorkerInUseException as exc:
            last_error = exc
            continue
    if env is None:
        raise last_error if last_error is not None else RuntimeError("Failed to create evaluation env.")
    actor = load_checkpoint_actor(checkpoint_path)

    wins = 0
    losses = 0
    draws = 0
    team_rewards = []
    opp_rewards = []
    episode_lengths = []

    try:
        for episode_idx in range(episodes):
            obs = env.reset()
            done = {"__all__": False}
            team_return = 0.0
            opp_return = 0.0
            steps = 0

            while not done["__all__"]:
                blue_obs = {0: obs[0], 1: obs[1]}
                orange_obs = {2: obs[2], 3: obs[3]}
                actions = {}
                actions.update(actor.act(blue_obs))
                actions.update(baseline_actor.act(orange_obs))
                obs, reward, done, _info = env.step(actions)
                team_return += float(reward[0] + reward[1])
                opp_return += float(reward[2] + reward[3])
                steps += 1

            team_rewards.append(team_return)
            opp_rewards.append(opp_return)
            episode_lengths.append(steps)

            if team_return > opp_return:
                wins += 1
            elif team_return < opp_return:
                losses += 1
            else:
                draws += 1

            print(
                f"{checkpoint_path.name} | episode {episode_idx + 1}/{episodes} | "
                f"blue={team_return:.3f} orange={opp_return:.3f} len={steps}"
            )
    finally:
        env.close()

    return {
        "checkpoint": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path),
        "worker_id": worker_id,
        "episodes": episodes,
        "win_rate": wins / episodes,
        "loss_rate": losses / episodes,
        "draw_rate": draws / episodes,
        "mean_team_reward": float(np.mean(team_rewards)),
        "mean_opp_reward": float(np.mean(opp_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
    }


def main():
    args = parse_args()
    trial_dir = Path(args.trial_dir).resolve()
    if not trial_dir.exists():
        raise SystemExit(f"Trial directory not found: {trial_dir}")

    checkpoint_files = discover_checkpoint_files(trial_dir)
    if not checkpoint_files:
        raise SystemExit(f"No checkpoint-* files found under {trial_dir}")
    checkpoint_files = select_uniform_checkpoints(checkpoint_files, args.num_checkpoints)

    baseline_actor = load_baseline_actor()
    results = []

    for checkpoint_path in checkpoint_files:
        result = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            baseline_actor=baseline_actor,
            episodes=args.episodes,
            worker_id=args.worker_id,
            worker_id_retries=args.worker_id_retries,
        )
        results.append(result)
        print(json.dumps(result, indent=2))

    results.sort(key=lambda row: checkpoint_step(row["checkpoint"]))

    output_json = trial_dir / f"{args.output_prefix}.json"
    output_csv = trial_dir / f"{args.output_prefix}.csv"
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(results).to_csv(output_csv, index=False)
    plot_results(results, trial_dir, args.output_prefix)

    print(f"Saved JSON results to {output_json}")
    print(f"Saved CSV results to {output_csv}")
    print(f"Saved plots to {trial_dir}")


if __name__ == "__main__":
    main()
