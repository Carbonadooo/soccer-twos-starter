"""
BC with observation normalization (z-score, per-dimension).

Modification vs plain BC:
  - After collecting data, compute per-dim mean & std from ALL samples
  - Normalize obs: (obs - mean) / (std + 1e-8)
  - Train BC on normalized obs
  - Save mean/std in checkpoint so inference applies the same transform

Why this helps:
  - Raw 336-dim obs has mixed scales (ray fractions ~[0,1], velocities ~[-10,10])
  - Normalization keeps every feature in ~[-3,3], which stabilises gradient flow
  - Same 336 dims, fully compatible with RL fine-tuning afterwards

Output: bc_obs_0/checkpoint.pth
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


# ── Hyperparameters ────────────────────────────────────────────────────────────
N_EPISODES = 300
N_EPOCHS   = 50
BATCH_SIZE = 512
LR         = 1e-3
SAVE_DIR   = "bc_obs_0"
# ──────────────────────────────────────────────────────────────────────────────


class BCPolicy(nn.Module):
    def __init__(self, obs_size: int, action_branches: list):
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
    env = soccer_twos.make()
    obs_list, act_list = [], []

    print(f"Collecting {n_episodes} episodes...")
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            blue_actions   = baseline_agent.act({0: obs[0], 1: obs[1]})
            orange_actions = baseline_agent.act({0: obs[2], 1: obs[3]})

            for pid in [0, 1]:
                obs_list.append(obs[pid].copy())
                act_list.append(blue_actions[pid].copy())

            env_actions = {0: blue_actions[0], 1: blue_actions[1],
                           2: orange_actions[0], 3: orange_actions[1]}
            obs, _, done_dict, _ = env.step(env_actions)
            done = max(done_dict.values())

        if (ep + 1) % 50 == 0:
            print(f"  [{ep+1}/{n_episodes}] samples: {len(obs_list)}")

    env.close()
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.int64)


def normalize_obs(obs_data: np.ndarray):
    """Compute per-dim mean/std and return normalised data + statistics."""
    mean = obs_data.mean(axis=0)           # shape: (336,)
    std  = obs_data.std(axis=0) + 1e-8    # avoid div-by-zero
    obs_norm = (obs_data - mean) / std
    return obs_norm, mean, std


def train_bc(obs_norm: np.ndarray, act_data: np.ndarray, action_branches: list):
    obs_t = torch.FloatTensor(obs_norm)
    act_t = torch.LongTensor(act_data)

    loader  = DataLoader(TensorDataset(obs_t, act_t), batch_size=BATCH_SIZE, shuffle=True)
    model   = BCPolicy(obs_t.shape[1], action_branches)
    optim   = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\nTraining BC (obs-normalised) for {N_EPOCHS} epochs "
          f"({len(obs_norm)} samples)...")

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


def save_model(model: BCPolicy, mean: np.ndarray, std: np.ndarray):
    os.makedirs(SAVE_DIR, exist_ok=True)
    ckpt_path = os.path.join(SAVE_DIR, "checkpoint.pth")
    torch.save({
        "state_dict":      model.state_dict(),
        "action_branches": model.action_branches,
        "obs_size":        model.shared[0].in_features,
        "obs_mean":        mean,   # saved for inference-time normalisation
        "obs_std":         std,
    }, ckpt_path)
    print(f"\nSaved → {ckpt_path}")


if __name__ == "__main__":
    ray.init(num_gpus=0, ignore_reinit_error=True, include_dashboard=False)

    print("Loading baseline agent...")
    env = soccer_twos.make()
    obs_size        = env.observation_space.shape[0]
    action_branches = list(env.action_space.nvec)
    print(f"  obs_size={obs_size}  action_branches={action_branches}")

    agent_module   = importlib.import_module("ceia_baseline_agent")
    baseline_agent = get_agent_class(agent_module)(env)
    env.close()

    # 1. Collect
    obs_data, act_data = collect_data(baseline_agent, N_EPISODES)
    print(f"Dataset: {obs_data.shape[0]} samples  raw obs range "
          f"[{obs_data.min():.2f}, {obs_data.max():.2f}]")

    # 2. Normalise
    obs_norm, mean, std = normalize_obs(obs_data)
    print(f"Normalised obs range [{obs_norm.min():.2f}, {obs_norm.max():.2f}]  "
          f"mean≈{mean.mean():.4f}  std≈{std.mean():.4f}")

    # 3. Train
    model = train_bc(obs_norm, act_data, action_branches)

    # 4. Save
    save_model(model, mean, std)
    print("\nDone. Test with:")
    print(f"  python -m soccer_twos.watch -m1 {SAVE_DIR} -m2 ceia_baseline_agent")
