import json
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from imitation_player_utils import PlayerPolicyNet, export_bc_weights


class BranchActionDataset(Dataset):
    def __init__(self, obs: np.ndarray, actions: np.ndarray):
        self.obs = torch.from_numpy(obs).float()
        self.actions = torch.from_numpy(actions).long()

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


def parse_args():
    parser = ArgumentParser(description="Train behavior cloning model from baseline dataset.")
    parser.add_argument("--dataset", default="bc_data/baseline_selfplay/dataset.npz")
    parser.add_argument("--output-dir", default="bc_results/baseline_bc")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--export-agent-weights",
        default="baseline_bc_agent/weights/baseline_bc.pt",
        help="Where to copy the best BC checkpoint for packaged agent use.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def branch_ce_loss(logits: torch.Tensor, actions: torch.Tensor):
    branch_logits = logits.view(-1, 3, 3)
    loss = 0.0
    for branch_idx in range(3):
        loss = loss + F.cross_entropy(branch_logits[:, branch_idx, :], actions[:, branch_idx])
    return loss / 3.0


def branch_accuracy(logits: torch.Tensor, actions: torch.Tensor):
    preds = logits.view(-1, 3, 3).argmax(dim=-1)
    return float((preds == actions).float().mean().item())


def evaluate(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    with torch.no_grad():
        for obs, actions in loader:
            obs = obs.to(device)
            actions = actions.to(device)
            logits = model(obs)
            total_loss += float(branch_ce_loss(logits, actions).item())
            total_acc += branch_accuracy(logits, actions)
            total_batches += 1
    if total_batches == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {
        "loss": total_loss / total_batches,
        "accuracy": total_acc / total_batches,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_npz = np.load(args.dataset)
    dataset = BranchActionDataset(dataset_npz["observations"], dataset_npz["actions"])

    total_size = len(dataset)
    val_size = max(1, int(total_size * args.val_split))
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PlayerPolicyNet(obs_size=336, hidden_size=256, action_logits_size=9).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_val_loss = float("inf")
    best_checkpoint_path = output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_acc = 0.0
        total_batches = 0
        for obs, actions in train_loader:
            obs = obs.to(device)
            actions = actions.to(device)
            optimizer.zero_grad()
            logits = model(obs)
            loss = branch_ce_loss(logits, actions)
            loss.backward()
            optimizer.step()

            total_train_loss += float(loss.item())
            total_train_acc += branch_accuracy(logits.detach(), actions)
            total_batches += 1

        train_metrics = {
            "loss": total_train_loss / max(total_batches, 1),
            "accuracy": total_train_acc / max(total_batches, 1),
        }
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        print(json.dumps(row))

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                },
                best_checkpoint_path,
            )

    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    export_bc_weights(best_checkpoint_path, Path(args.export_agent_weights))
    print(f"Saved best BC checkpoint to {best_checkpoint_path.resolve()}")
    print(f"Exported packaged agent weights to {Path(args.export_agent_weights).resolve()}")


if __name__ == "__main__":
    main()
