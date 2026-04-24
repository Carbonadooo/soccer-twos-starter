"""
Behavioral cloning from the baseline agent (ceia_baseline_agent).

Steps:
  1. Collect (obs, action) pairs by running baseline agent vs itself
  2. Train a BC policy via supervised learning (cross-entropy per action branch)
  3. Save model weights to bc_agent/checkpoint.pth

Run:
    python distill_baseline.py

Requires ceia_baseline_agent/ folder in the project root.
After this, run RL fine-tuning with:
    python train_rl_finetune.py
"""

import importlib
import os

import numpy as np
import ray
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import soccer_twos
from soccer_twos.utils import get_agent_class

# Pre-init Ray with num_gpus=0 so the baseline agent's ray.init()
# (which has ignore_reinit_error=True) skips GPU auto-detection on Windows.
ray.init(num_gpus=0, ignore_reinit_error=True, include_dashboard=False)


# ── Hyperparameters ────────────────────────────────────────────────────────────
N_EPISODES = 300   # episodes to collect; ~150-300k samples
N_EPOCHS   = 50
BATCH_SIZE = 512
LR         = 1e-3
SAVE_DIR   = "bc_agent"
# ──────────────────────────────────────────────────────────────────────────────


class BCPolicy(nn.Module):
    def __init__(self, obs_size: int, action_branches: list):
        """
        Args:
            obs_size:        observation dimension (336 for soccer-twos)
            action_branches: list of branch sizes, e.g. [3, 3, 3]
        """
        super().__init__()
        self.action_branches = action_branches
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(512, n) for n in action_branches])

    def forward(self, x: torch.Tensor):
        f = self.shared(x)
        return [head(f) for head in self.heads]

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(x)
        return np.array([torch.argmax(l).item() for l in logits])


def collect_data(baseline_agent, n_episodes: int):
    """Run baseline agent vs itself and collect (obs, action) pairs."""
    env = soccer_twos.make()
    obs_list, act_list = [], []

    print(f"Collecting {n_episodes} episodes...")
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            # Baseline controls both teams (relative player IDs 0,1 per team)
            blue_actions   = baseline_agent.act({0: obs[0], 1: obs[1]})
            orange_actions = baseline_agent.act({0: obs[2], 1: obs[3]})

            # Record blue team transitions
            for pid in [0, 1]:
                obs_list.append(obs[pid].copy())
                act_list.append(blue_actions[pid].copy())

            env_actions = {
                0: blue_actions[0],
                1: blue_actions[1],
                2: orange_actions[0],
                3: orange_actions[1],
            }
            obs, _, done_dict, _ = env.step(env_actions)
            done = max(done_dict.values())

        if (ep + 1) % 50 == 0:
            print(f"  [{ep+1}/{n_episodes}] samples: {len(obs_list)}")

    env.close()
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.int64)


def train_bc(obs_data: np.ndarray, act_data: np.ndarray, action_branches: list):
    """Supervised training: predict each action branch via cross-entropy."""
    obs_t = torch.FloatTensor(obs_data)
    act_t = torch.LongTensor(act_data)  # [N, n_branches]

    loader  = DataLoader(TensorDataset(obs_t, act_t), batch_size=BATCH_SIZE, shuffle=True)
    model   = BCPolicy(obs_t.shape[1], action_branches)
    optim   = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\nTraining BC policy for {N_EPOCHS} epochs "
          f"({len(obs_data)} samples, batch {BATCH_SIZE})...")

    for epoch in range(N_EPOCHS):
        total_loss = 0.0
        correct = [0] * len(action_branches)
        total   = 0

        for obs_b, act_b in loader:
            optim.zero_grad()
            logits = model(obs_b)
            loss = sum(loss_fn(logits[i], act_b[:, i]) for i in range(len(action_branches)))
            loss.backward()
            optim.step()
            total_loss += loss.item()

            for i in range(len(action_branches)):
                correct[i] += (logits[i].argmax(1) == act_b[:, i]).sum().item()
            total += len(obs_b)

        if (epoch + 1) % 10 == 0:
            acc = [c / total for c in correct]
            print(f"  Epoch {epoch+1:3d}/{N_EPOCHS}  "
                  f"loss={total_loss/len(loader):.4f}  "
                  f"acc={[f'{a:.2f}' for a in acc]}")

    return model


def save_model(model: BCPolicy):
    os.makedirs(SAVE_DIR, exist_ok=True)
    ckpt_path = os.path.join(SAVE_DIR, "checkpoint.pth")
    torch.save({
        "state_dict":      model.state_dict(),
        "action_branches": model.action_branches,
        "obs_size":        model.shared[0].in_features,
    }, ckpt_path)
    print(f"\nSaved model → {ckpt_path}")


if __name__ == "__main__":
    # ── Load baseline agent ────────────────────────────────────────────────────
    print("Loading baseline agent...")
    env = soccer_twos.make()
    obs_size       = env.observation_space.shape[0]
    action_branches = list(env.action_space.nvec)
    print(f"  obs_size={obs_size}  action_branches={action_branches}")

    agent_module   = importlib.import_module("ceia_baseline_agent")
    baseline_agent = get_agent_class(agent_module)(env)
    env.close()

    # ── Collect data ───────────────────────────────────────────────────────────
    obs_data, act_data = collect_data(baseline_agent, N_EPISODES)
    print(f"Dataset: {obs_data.shape[0]} samples")

    # ── Train BC ───────────────────────────────────────────────────────────────
    model = train_bc(obs_data, act_data, action_branches)

    # ── Save ───────────────────────────────────────────────────────────────────
    save_model(model)
    print("\nNext step: run  python train_rl_finetune.py")
