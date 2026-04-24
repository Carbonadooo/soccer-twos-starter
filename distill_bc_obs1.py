"""
BC with expanded observation (674-dim):
  [0:336]   z-score normalised obs_t
  [336:672] obs_t - obs_{t-1}  (velocity / temporal context)
  [672:674] one-hot player ID  [1,0]=player0  [0,1]=player1

Why each component helps:
  normalised obs  → mixed-scale raw obs hurts gradient flow; z-score fixes this
  obs velocity    → single frame is partially observable; diff reveals dynamics
  player ID       → allows model to learn role specialisation (striker vs goalie)

Output: bc_obs_1/checkpoint.pth
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
N_EPISODES  = 300
N_EPOCHS    = 50
BATCH_SIZE  = 512
LR          = 1e-3
SAVE_DIR    = "bc_obs_1"
OBS_RAW_DIM = 336
N_PLAYERS   = 2          # players per team
# ──────────────────────────────────────────────────────────────────────────────


# ── Observation transformer ────────────────────────────────────────────────────

class ObsTransform:
    """
    Stateful per-player observation transformer.
    Must call reset() at the start of every episode.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean          # (336,)
        self.std  = std           # (336,)
        self._prev: dict = {}     # {player_id: prev_raw_obs}

    def reset(self):
        self._prev = {}

    def __call__(self, obs: np.ndarray, player_id: int) -> np.ndarray:
        # 1. z-score normalised obs
        obs_norm = (obs - self.mean) / self.std

        # 2. velocity: diff from previous frame (zeros on first step)
        prev     = self._prev.get(player_id, np.zeros(OBS_RAW_DIM, dtype=np.float32))
        obs_diff = obs - prev
        self._prev[player_id] = obs.copy()

        # 3. player one-hot
        pid_oh = np.zeros(N_PLAYERS, dtype=np.float32)
        pid_oh[player_id % N_PLAYERS] = 1.0

        return np.concatenate([obs_norm, obs_diff, pid_oh]).astype(np.float32)

    @property
    def out_dim(self):
        return OBS_RAW_DIM + OBS_RAW_DIM + N_PLAYERS   # 674


# ── Model ──────────────────────────────────────────────────────────────────────

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

    def forward(self, x):
        f = self.shared(x)
        return [head(f) for head in self.heads]


# ── Data collection ────────────────────────────────────────────────────────────

def collect_raw(baseline_agent, n_episodes: int):
    """Collect raw (obs, action) pairs — normalisation applied after."""
    env = soccer_twos.make()
    obs_list, act_list, pid_list = [], [], []

    print(f"Collecting {n_episodes} episodes...")
    for ep in range(n_episodes):
        obs  = env.reset()
        done = False
        while not done:
            blue   = baseline_agent.act({0: obs[0], 1: obs[1]})
            orange = baseline_agent.act({0: obs[2], 1: obs[3]})

            for pid in [0, 1]:
                obs_list.append(obs[pid].copy())
                act_list.append(blue[pid].copy())
                pid_list.append(pid)

            env_actions = {0: blue[0], 1: blue[1], 2: orange[0], 3: orange[1]}
            obs, _, done_dict, _ = env.step(env_actions)
            done = max(done_dict.values())

        if (ep + 1) % 50 == 0:
            print(f"  [{ep+1}/{n_episodes}] samples: {len(obs_list)}")

    env.close()
    return (np.array(obs_list, dtype=np.float32),
            np.array(act_list, dtype=np.int64),
            np.array(pid_list, dtype=np.int64))


def build_features(obs_data, pid_data, mean, std, n_episodes_approx):
    """Apply ObsTransform to every sample, respecting episode boundaries."""
    transform = ObsTransform(mean, std)
    feats = []
    episode_len = len(obs_data) // n_episodes_approx  # approximate

    for i, (obs, pid) in enumerate(zip(obs_data, pid_data)):
        # Heuristic reset: player 0 appearing after player 1 signals new episode
        if i > 0 and pid == 0 and pid_data[i - 1] == 1:
            transform.reset()
        feats.append(transform(obs, int(pid)))

    return np.array(feats, dtype=np.float32)


# ── Training ───────────────────────────────────────────────────────────────────

def train_bc(feat_data, act_data, action_branches):
    obs_t = torch.FloatTensor(feat_data)
    act_t = torch.LongTensor(act_data)

    loader  = DataLoader(TensorDataset(obs_t, act_t), batch_size=BATCH_SIZE, shuffle=True)
    model   = BCPolicy(obs_t.shape[1], action_branches)
    optim   = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\nTraining BC (674-dim obs) for {N_EPOCHS} epochs "
          f"({len(feat_data)} samples)...")

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


# ── Save ───────────────────────────────────────────────────────────────────────

def save_model(model, mean, std):
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, "checkpoint.pth")
    torch.save({
        "state_dict":      model.state_dict(),
        "action_branches": model.action_branches,
        "obs_size":        model.shared[0].in_features,
        "obs_mean":        mean,
        "obs_std":         std,
    }, path)
    print(f"\nSaved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ray.init(num_gpus=0, ignore_reinit_error=True, include_dashboard=False)

    print("Loading baseline agent...")
    env = soccer_twos.make()
    action_branches = list(env.action_space.nvec)
    print(f"  raw obs_dim={OBS_RAW_DIM}  action_branches={action_branches}")

    agent_module   = importlib.import_module("ceia_baseline_agent")
    baseline_agent = get_agent_class(agent_module)(env)
    env.close()

    # 1. Collect raw data
    obs_data, act_data, pid_data = collect_raw(baseline_agent, N_EPISODES)
    print(f"Dataset: {len(obs_data)} samples")

    # 2. Compute normalisation stats from raw obs
    mean = obs_data.mean(axis=0)
    std  = obs_data.std(axis=0) + 1e-8
    print(f"Obs stats: mean≈{mean.mean():.4f}  std≈{std.mean():.4f}")

    # 3. Build 674-dim features
    feat_data = build_features(obs_data, pid_data, mean, std, N_EPISODES)
    print(f"Feature shape: {feat_data.shape}  "
          f"range [{feat_data.min():.2f}, {feat_data.max():.2f}]")

    # 4. Train
    model = train_bc(feat_data, act_data, action_branches)

    # 5. Save
    save_model(model, mean, std)
    print("\nDone. Test with:")
    print(f"  python -m soccer_twos.evaluate -m1 {SAVE_DIR} -m2 ceia_baseline_agent -e 10")
