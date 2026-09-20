#!/usr/bin/env python3
"""
train_12tasks_study.py
-----------------------
Phase 3: Multi-Task Policy Training for the Force-Augmented Observation Study.

Trains 3 policy architectures on the 12-task dataset (dataset_12tasks/):
  1. Kinematic MLP:
     - Input: 39-D kinematic observation
     - Architecture: 4-layer MLP (LayerNorm + ReLU, hidden_dim=256) -> Tanh action head (4-D)
  2. Kinematic GRU:
     - Input: 39-D kinematic observation
     - Architecture: Linear(39, 256) projection -> GRU(256, 256, 1 layer) -> 2-layer MLP head -> Tanh action head (4-D)
     - History length: H = 8
  3. Kinematic + Contact Force/Torque GRU:
     - Input: 45-D observation (39-D kinematic + 6-D contact wrench)
     - Architecture: Linear(45, 256) projection -> GRU(256, 256, 1 layer) -> 2-layer MLP head -> Tanh action head (4-D)
     - Minimal input-layer modification: ONLY in_features of projection layer changes from 39 to 45; identical GRU, head, optimiser, scheduler, epochs, seed.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

class BCPolicyMLP(nn.Module):
    """Feed-Forward Behavioral Cloning Policy Network."""

    def __init__(self, obs_dim: int = 39, act_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

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


class BCPolicyGRU(nn.Module):
    """
    GRU-based Behavioral Cloning Policy.
    Supports sequence input during training and step() during closed-loop rollout.
    """

    def __init__(
        self,
        obs_dim: int = 39,
        act_dim: int = 4,
        hidden_dim: int = 256,
        num_gru_layers: int = 1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Recurrent core
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
        )

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, act_dim),
            nn.Tanh(),
        )

        self._h: torch.Tensor = None

    def forward(
        self, obs_seq: torch.Tensor, hidden: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        obs_seq: (B, H, obs_dim)
        hidden:  (num_layers, B, hidden_dim) or None
        """
        proj = self.input_proj(obs_seq)          # (B, H, hidden_dim)
        gru_out, h_n = self.gru(proj, hidden)    # gru_out: (B, H, hidden_dim)
        last_out = gru_out[:, -1, :]             # take last step (many-to-one)
        actions = self.action_head(last_out)
        return actions, h_n

    def reset_hidden(self, device: torch.device):
        self._h = torch.zeros(self.num_gru_layers, 1, self.hidden_dim, device=device)

    def step(self, obs_norm: torch.Tensor) -> torch.Tensor:
        """
        Single-step inference maintaining internal hidden state.
        obs_norm: (1, obs_dim)
        """
        obs_seq = obs_norm.unsqueeze(1)          # (1, 1, obs_dim)
        action, self._h = self.forward(obs_seq, self._h)
        return action.squeeze(0)                 # (act_dim,)


# ---------------------------------------------------------------------------
# Data Preparation Helpers
# ---------------------------------------------------------------------------

def build_sequences_vectorized(
    obs_array: np.ndarray,
    episode_starts: np.ndarray,
    episode_ends: np.ndarray,
    history_len: int = 8,
) -> np.ndarray:
    """
    Constructs sliding window sequences of length H for all episodes vectorized.
    Each episode is padded at the start with (H - 1) copies of its initial observation.
    Returns: (N_total_steps, H, obs_dim)
    """
    all_windows = []
    for start, end in zip(episode_starts, episode_ends):
        ep_obs = obs_array[start:end]  # (T, D)
        T, D = ep_obs.shape
        pad = np.tile(ep_obs[0:1], (history_len - 1, 1))
        padded = np.concatenate([pad, ep_obs], axis=0)  # (T + H - 1, D)
        # sliding_window_view on axis 0
        w = np.lib.stride_tricks.sliding_window_view(padded, window_shape=history_len, axis=0)  # (T, D, H)
        w = np.swapaxes(w, 1, 2)  # (T, H, D)
        all_windows.append(w)
    return np.concatenate(all_windows, axis=0).astype(np.float32)


def train_epoch_mlp(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_mae, total_n = 0.0, 0.0, 0
    for obs, act in loader:
        obs, act = obs.to(device), act.to(device)
        optimizer.zero_grad()
        pred = model(obs)
        loss = criterion(pred, act)
        loss.backward()
        optimizer.step()
        n = obs.size(0)
        total_loss += loss.item() * n
        total_mae += torch.abs(pred - act).mean().item() * n
        total_n += n
    return total_loss / total_n, total_mae / total_n


@torch.no_grad()
def evaluate_mlp(model, loader, criterion, device):
    model.eval()
    total_loss, total_mae, total_n = 0.0, 0.0, 0
    for obs, act in loader:
        obs, act = obs.to(device), act.to(device)
        pred = model(obs)
        loss = criterion(pred, act)
        n = obs.size(0)
        total_loss += loss.item() * n
        total_mae += torch.abs(pred - act).mean().item() * n
        total_n += n
    return total_loss / total_n, total_mae / total_n


def train_epoch_gru(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_mae, total_n = 0.0, 0.0, 0
    for seq, act in loader:
        seq, act = seq.to(device), act.to(device)
        optimizer.zero_grad()
        pred, _ = model(seq)
        loss = criterion(pred, act)
        loss.backward()
        optimizer.step()
        n = seq.size(0)
        total_loss += loss.item() * n
        total_mae += torch.abs(pred - act).mean().item() * n
        total_n += n
    return total_loss / total_n, total_mae / total_n


@torch.no_grad()
def evaluate_gru(model, loader, criterion, device):
    model.eval()
    total_loss, total_mae, total_n = 0.0, 0.0, 0
    for seq, act in loader:
        seq, act = seq.to(device), act.to(device)
        pred, _ = model(seq)
        loss = criterion(pred, act)
        n = seq.size(0)
        total_loss += loss.item() * n
        total_mae += torch.abs(pred - act).mean().item() * n
        total_n += n
    return total_loss / total_n, total_mae / total_n


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def run_training(
    model_name: str,
    arch_type: str,
    feature_key: str,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 256,
    history_len: int = 8,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
):
    print(f"\n" + "=" * 80)
    print(f" TRAINING: {model_name} (Arch: {arch_type}, Features: '{feature_key}')")
    print(f"=" * 80)
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr} | H: {history_len}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load NPZ partitions
    train_npz = np.load("dataset_12tasks/train.npz")
    val_npz = np.load("dataset_12tasks/val.npz")
    test_npz = np.load("dataset_12tasks/test.npz")

    train_raw = train_npz[feature_key].astype(np.float32)
    val_raw = val_npz[feature_key].astype(np.float32)
    test_raw = test_npz[feature_key].astype(np.float32)

    train_act = train_npz["actions"].astype(np.float32)
    val_act = val_npz["actions"].astype(np.float32)
    test_act = test_npz["actions"].astype(np.float32)

    obs_dim = train_raw.shape[1]
    act_dim = train_act.shape[1]

    # Compute normalization from train partition
    obs_mean = train_raw.mean(axis=0)
    obs_std = train_raw.std(axis=0)
    obs_std = np.where(obs_std < 1e-6, 1.0, obs_std)  # prevent division by zero for inactive channels

    train_norm = (train_raw - obs_mean) / obs_std
    val_norm = (val_raw - obs_mean) / obs_std
    test_norm = (test_raw - obs_mean) / obs_std

    # Prepare DataLoaders
    if arch_type == "mlp":
        train_ds = TensorDataset(torch.from_numpy(train_norm), torch.from_numpy(train_act))
        val_ds = TensorDataset(torch.from_numpy(val_norm), torch.from_numpy(val_act))
        test_ds = TensorDataset(torch.from_numpy(test_norm), torch.from_numpy(test_act))
        model = BCPolicyMLP(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim).to(device)
        train_fn = train_epoch_mlp
        eval_fn = evaluate_mlp
    elif arch_type == "gru":
        print("  Building sliding-window sequences for GRU (vectorized)...")
        t0_seq = time.time()
        train_seq = build_sequences_vectorized(train_norm, train_npz["episode_starts"], train_npz["episode_ends"], history_len)
        val_seq = build_sequences_vectorized(val_norm, val_npz["episode_starts"], val_npz["episode_ends"], history_len)
        test_seq = build_sequences_vectorized(test_norm, test_npz["episode_starts"], test_npz["episode_ends"], history_len)
        print(f"  [OK] Sequences built in {time.time() - t0_seq:.2f}s. Train shape: {train_seq.shape}")

        train_ds = TensorDataset(torch.from_numpy(train_seq), torch.from_numpy(train_act))
        val_ds = TensorDataset(torch.from_numpy(val_seq), torch.from_numpy(val_act))
        test_ds = TensorDataset(torch.from_numpy(test_seq), torch.from_numpy(test_act))
        model = BCPolicyGRU(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim).to(device)
        train_fn = train_epoch_gru
        eval_fn = evaluate_gru
    else:
        raise ValueError(f"Unknown arch_type: {arch_type}")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized {model.__class__.__name__} with {param_count:,} trainable parameters.\n")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None
    best_val_mae = float("inf")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
        "lr": [],
    }

    t0_train = time.time()
    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        train_loss, train_mae = train_fn(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = eval_fn(model, val_loader, criterion, device)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["lr"].append(cur_lr)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_mae = val_mae
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        mark = " *" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f} | "
            f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | "
            f"Time: {time.time() - ep_t0:.2f}s{mark}"
        )

    total_time = time.time() - t0_train
    print(f"\nTraining completed in {total_time:.1f}s ({total_time / 60:.2f} min).")
    print(f"Best Val MSE: {best_val_loss:.5f} at Epoch {best_epoch}.")

    # Evaluate best model on test set
    model.load_state_dict(best_state_dict)
    test_loss, test_mae = eval_fn(model, test_loader, criterion, device)
    print(f"[TEST EVALUATION] Held-out Test MSE: {test_loss:.5f} | Test Action MAE: {test_mae:.4f}")

    # Save checkpoint
    ckpt_path = f"models/{model_name}.pt"
    checkpoint = {
        "model_name": model_name,
        "arch_type": arch_type,
        "feature_key": feature_key,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hidden_dim": hidden_dim,
        "history_len": history_len if arch_type == "gru" else 1,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "model_state_dict": best_state_dict,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_mae": best_val_mae,
        "test_loss": test_loss,
        "test_mae": test_mae,
        "history": history,
        "total_time_seconds": total_time,
    }
    torch.save(checkpoint, ckpt_path)
    print(f"  [OK] Saved model checkpoint to '{ckpt_path}'")

    # Save metrics JSON
    metrics_path = f"models/{model_name}_metrics.json"
    metrics_payload = {
        "model_name": model_name,
        "arch_type": arch_type,
        "feature_key": feature_key,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hidden_dim": hidden_dim,
        "history_len": history_len if arch_type == "gru" else 1,
        "param_count": param_count,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_mae": best_val_mae,
        "test_loss": test_loss,
        "test_mae": test_mae,
        "total_time_seconds": total_time,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"  [OK] Saved metrics to '{metrics_path}'")

    return metrics_payload


def main():
    parser = argparse.ArgumentParser(description="Train 12-task imitation learning models.")
    parser.add_argument("--model", type=str, default="all", choices=["mlp", "gru_kinematic", "gru_force", "all"],
                        help="Which model variant to train.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20).")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size (default: 512).")
    parser.add_argument("--threads", type=int, default=14, help="PyTorch CPU threads.")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs("models", exist_ok=True)

    configs = {
        "mlp": {
            "model_name": "bc_12tasks_mlp_kinematic",
            "arch_type": "mlp",
            "feature_key": "obs",
        },
        "gru_kinematic": {
            "model_name": "bc_12tasks_gru_kinematic",
            "arch_type": "gru",
            "feature_key": "obs",
        },
        "gru_force": {
            "model_name": "bc_12tasks_gru_force",
            "arch_type": "gru",
            "feature_key": "obs_force",
        },
    }

    models_to_train = [args.model] if args.model != "all" else ["mlp", "gru_kinematic", "gru_force"]

    all_metrics = {}
    for m in models_to_train:
        cfg = configs[m]
        res = run_training(
            model_name=cfg["model_name"],
            arch_type=cfg["arch_type"],
            feature_key=cfg["feature_key"],
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        all_metrics[m] = res

    print("\n" + "=" * 80)
    print(" PHASE 3 TRAINING SUMMARY ACROSS 12 TASKS")
    print("=" * 80)
    for m, res in all_metrics.items():
        print(f"{res['model_name']:<30} | Obs: {res['obs_dim']}D | Val MSE: {res['best_val_loss']:.5f} | Test MSE: {res['test_loss']:.5f} | Test MAE: {res['test_mae']:.4f} | Time: {res['total_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
