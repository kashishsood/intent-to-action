#!/usr/bin/env python3
"""
Behavior Cloning (BC) Training Script for Meta-World MT10 Demonstrations

Trains a lightweight Multi-Layer Perceptron (MLP) on the recorded demonstration
trajectories to predict the next robot action given the current observation.
Saves model checkpoints, evaluation metrics on the test partition, and training curves.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class BCPolicyMLP(nn.Module):
    """
    Feed-Forward Behavioral Cloning Policy Network.
    Maps continuous observation (39 dims) to continuous action (4 dims) bounded by Tanh.
    """

    def __init__(self, obs_dim: int = 39, act_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


def load_partition(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads observations and actions from compressed NPZ archive."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset partition not found: '{npz_path}'")
    data = np.load(npz_path)
    return data["obs"].astype(np.float32), data["actions"].astype(np.float32)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs one training epoch, returning mean MSE loss and action MAE."""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    for obs, act in loader:
        obs, act = obs.to(device), act.to(device)
        optimizer.zero_grad()
        pred_act = model(obs)
        loss = criterion(pred_act, act)
        loss.backward()
        optimizer.step()

        batch_size = obs.size(0)
        total_loss += loss.item() * batch_size
        total_mae += torch.abs(pred_act - act).mean().item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples, total_mae / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluates policy on a dataset loader, returning mean MSE loss and action MAE."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    for obs, act in loader:
        obs, act = obs.to(device), act.to(device)
        pred_act = model(obs)
        loss = criterion(pred_act, act)

        batch_size = obs.size(0)
        total_loss += loss.item() * batch_size
        total_mae += torch.abs(pred_act - act).mean().item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples, total_mae / total_samples


def plot_training_curves(history: Dict[str, list], output_path: str, best_epoch: int):
    """Generates and saves publication-quality training curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Panel 1: MSE Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss (MSE)", color="#2563EB", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss (MSE)", color="#DC2626", lw=2)
    ax1.axvline(best_epoch, color="#16A34A", linestyle="--", label=f"Best Model (Epoch {best_epoch})")
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="medium")
    ax1.set_ylabel("Mean Squared Error (MSE)", fontsize=11, fontweight="medium")
    ax1.set_title("Behavior Cloning Loss Curve", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True, facecolor="white", edgecolor="#E5E7EB")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Action MAE
    ax2.plot(epochs, history["train_mae"], label="Train Action MAE", color="#2563EB", lw=2)
    ax2.plot(epochs, history["val_mae"], label="Val Action MAE", color="#DC2626", lw=2)
    ax2.axvline(best_epoch, color="#16A34A", linestyle="--", label=f"Best Model (Epoch {best_epoch})")
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="medium")
    ax2.set_ylabel("Mean Absolute Error (MAE)", fontsize=11, fontweight="medium")
    ax2.set_title("Action Prediction Error Curve", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True, facecolor="white", edgecolor="#E5E7EB")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved training curves plot to '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Train Behavior Cloning model on Meta-World.")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Directory with train/val/test NPZ files.")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save checkpoints and plots.")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs (default: 35).")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size (default: 256).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (default: 1e-3).")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW (default: 1e-4).")
    parser.add_argument("--hidden-dim", type=int, default=256, help="MLP hidden dimension (default: 256).")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cpu', 'cuda').")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    # Device configuration
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n==========================================")
    print(f"BEHAVIOR CLONING (BC) MODEL TRAINING")
    print(f"Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print(f"==========================================")

    # Set reproducibility seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load data partitions
    train_obs, train_act = load_partition(os.path.join(args.data_dir, "train.npz"))
    val_obs, val_act = load_partition(os.path.join(args.data_dir, "val.npz"))
    test_obs, test_act = load_partition(os.path.join(args.data_dir, "test.npz"))

    print(f"Train set: {train_obs.shape[0]:,} samples (obs: {train_obs.shape[1]}, act: {train_act.shape[1]})")
    print(f"Val set:   {val_obs.shape[0]:,} samples")
    print(f"Test set:  {test_obs.shape[0]:,} samples")

    # 2. Compute observation normalization statistics strictly from train partition
    obs_mean = train_obs.mean(axis=0)
    obs_std = train_obs.std(axis=0) + 1e-6

    # Normalize observations
    train_obs_norm = (train_obs - obs_mean) / obs_std
    val_obs_norm = (val_obs - obs_mean) / obs_std
    test_obs_norm = (test_obs - obs_mean) / obs_std

    # Build PyTorch DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_obs_norm), torch.from_numpy(train_act)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_obs_norm), torch.from_numpy(val_act)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_obs_norm), torch.from_numpy(test_act)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # 3. Instantiate model
    model = BCPolicyMLP(obs_dim=train_obs.shape[1], act_dim=train_act.shape[1], hidden_dim=args.hidden_dim).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized BCPolicyMLP with {param_count:,} trainable parameters.\n")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 1

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
        "lr": [],
    }

    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.perf_counter()
        train_loss, train_mae = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["lr"].append(current_lr)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_mae": val_mae,
                "obs_mean": obs_mean,
                "obs_std": obs_std,
                "obs_dim": model.obs_dim,
                "act_dim": model.act_dim,
                "hidden_dim": args.hidden_dim,
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "best_bc_model.pt"))

        # Save latest checkpoint
        latest_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "history": history,
        }
        torch.save(latest_ckpt, os.path.join(args.output_dir, "latest_bc_model.pt"))

        mark = " *" if is_best else ""
        ep_duration = time.perf_counter() - ep_start
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f} | "
            f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | "
            f"Time: {ep_duration:.2f}s{mark}"
        )

    total_training_time = time.perf_counter() - start_time
    print(f"\nTraining completed in {total_training_time:.2f}s ({total_training_time / 60:.2f} min).")
    print(f"Best Validation Loss: {best_val_loss:.5f} at Epoch {best_epoch}.")

    # 4. Final Evaluation on Held-Out Test Partition
    print("\nEvaluating Best Model on Held-Out Test Episodes...")
    best_checkpoint = torch.load(os.path.join(args.output_dir, "best_bc_model.pt"), map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss, test_mae = evaluate(model, test_loader, criterion, device)
    print(f"[OK] Test Loss (MSE): {test_loss:.5f} | Test Action MAE: {test_mae:.4f}")

    # 5. Save training curves plot
    plot_path = os.path.join(args.output_dir, "training_curves.png")
    plot_training_curves(history, plot_path, best_epoch)

    # 6. Save training metrics metadata
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    metrics_payload = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_mae": test_mae,
        "total_training_time_seconds": total_training_time,
        "history": history,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[OK] Saved training metrics to '{metrics_path}'")
    print("==========================================\n")


if __name__ == "__main__":
    main()
