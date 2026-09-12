#!/usr/bin/env python3
"""
train_bc_gru.py
---------------
GRU-based Behavior Cloning policy for Meta-World MT10.

Architecture:
  - Normalised obs at each step fed into a single-layer GRU (hidden_dim=256)
  - GRU hidden state passed through a 2-layer MLP head -> action (4 dims)
  - Tanh output to match [-1, 1] action bounds

Training setup is intentionally identical to train_bc.py (same splits, AdamW,
cosine schedule, epochs, seed) so results are directly comparable.

History length (--history-len, default 8): number of past observations the GRU
sees per prediction. At episode boundaries the hidden state is reset to zero.
Data is constructed by sliding a window of size H over each episode in order;
the first H-1 steps of each episode are padded with the first observation.

Usage:
    python train_bc_gru.py
    python train_bc_gru.py --history-len 4 --epochs 35 --device cpu
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BCPolicyGRU(nn.Module):
    """
    GRU-based Behavioral Cloning Policy.

    At inference time, call reset_hidden() at the start of each episode, then
    call forward(obs) at each step with a single (unnormalised) observation.
    The GRU hidden state is maintained internally between calls.

    At training time, the model is fed sequences of length H and only the
    final hidden state is used to predict the action (many-to-one).
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

        # Input projection: obs -> hidden_dim (adds capacity before GRU)
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
        )

        # Action head: GRU output -> act_dim
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, act_dim),
            nn.Tanh(),
        )

        # Hidden state for single-step inference (reset per episode)
        self._h: torch.Tensor | None = None

    def forward(
        self, obs_seq: torch.Tensor, hidden: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_seq: (batch, seq_len, obs_dim)  — normalised observations
            hidden:  (num_layers, batch, hidden_dim) or None

        Returns:
            actions: (batch, act_dim)  — prediction at the last step of the sequence
            hidden:  (num_layers, batch, hidden_dim) — updated hidden state
        """
        proj = self.input_proj(obs_seq)          # (B, T, hidden_dim)
        gru_out, h_n = self.gru(proj, hidden)    # gru_out: (B, T, hidden_dim)
        last_out = gru_out[:, -1, :]             # take last timestep
        actions = self.action_head(last_out)
        return actions, h_n

    # ------------------------------------------------------------------
    # Single-step inference helpers (used during closed-loop rollout)
    # ------------------------------------------------------------------

    def reset_hidden(self, device: torch.device):
        """Reset the internal hidden state (call at episode start)."""
        self._h = torch.zeros(
            self.num_gru_layers, 1, self.hidden_dim, device=device
        )

    def step(self, obs_norm: torch.Tensor) -> torch.Tensor:
        """
        Single-step inference with maintained hidden state.

        Args:
            obs_norm: (1, obs_dim) normalised observation tensor on the correct device

        Returns:
            action: (act_dim,) action tensor
        """
        obs_seq = obs_norm.unsqueeze(1)          # (1, 1, obs_dim)
        action, self._h = self.forward(obs_seq, self._h)
        return action.squeeze(0)                 # (act_dim,)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """
    Builds fixed-length observation sequences from episode-structured parquet data.

    For each timestep t in an episode, the input is obs[t-H+1 : t+1] (padded
    with the first obs of the episode if t < H-1), and the target is action[t].
    """

    def __init__(
        self,
        parquet_path: str,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        history_len: int = 8,
    ):
        df = pd.read_parquet(parquet_path)
        self.history_len = history_len

        all_obs_seqs: List[np.ndarray] = []
        all_actions: List[np.ndarray] = []

        # Group by (task_name, global_episode_id) to respect episode boundaries
        group_cols = ["task_name", "global_episode_id"] if "global_episode_id" in df.columns else ["task_name", "episode_id"]
        for _, episode_df in df.groupby(group_cols, sort=False):
            episode_df = episode_df.sort_values("timestep")
            obs_raw = np.array(episode_df["obs"].tolist(), dtype=np.float32)
            actions = np.array(episode_df["action"].tolist(), dtype=np.float32)

            # Normalise
            obs_norm = (obs_raw - obs_mean) / obs_std

            T = len(obs_norm)
            for t in range(T):
                start = t - history_len + 1
                if start < 0:
                    # Pad with repetition of the first observation
                    pad_count = -start
                    window = np.concatenate([
                        np.tile(obs_norm[0], (pad_count, 1)),
                        obs_norm[0 : t + 1],
                    ], axis=0)
                else:
                    window = obs_norm[start : t + 1]
                assert window.shape == (history_len, obs_norm.shape[1])
                all_obs_seqs.append(window)
                all_actions.append(actions[t])

        self.obs_seqs = torch.from_numpy(np.stack(all_obs_seqs))   # (N, H, obs_dim)
        self.actions  = torch.from_numpy(np.stack(all_actions))    # (N, act_dim)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.obs_seqs[idx], self.actions[idx]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: BCPolicyGRU,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = total_mae = total_n = 0
    for obs_seq, act in loader:
        obs_seq, act = obs_seq.to(device), act.to(device)
        optimizer.zero_grad()
        pred, _ = model(obs_seq)
        loss = criterion(pred, act)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        n = obs_seq.size(0)
        total_loss += loss.item() * n
        total_mae  += torch.abs(pred - act).mean().item() * n
        total_n    += n
    return total_loss / total_n, total_mae / total_n


@torch.no_grad()
def evaluate_epoch(
    model: BCPolicyGRU,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = total_mae = total_n = 0
    for obs_seq, act in loader:
        obs_seq, act = obs_seq.to(device), act.to(device)
        pred, _ = model(obs_seq)
        loss = criterion(pred, act)
        n = obs_seq.size(0)
        total_loss += loss.item() * n
        total_mae  += torch.abs(pred - act).mean().item() * n
        total_n    += n
    return total_loss / total_n, total_mae / total_n


def plot_training_curves(history: Dict, output_path: str, best_epoch: int):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    ax1.plot(epochs, history["train_loss"], label="Train MSE", color="#2563EB", lw=2)
    ax1.plot(epochs, history["val_loss"],   label="Val MSE",   color="#DC2626", lw=2)
    ax1.axvline(best_epoch, color="#16A34A", ls="--", label=f"Best (ep {best_epoch})")
    ax1.set(xlabel="Epoch", ylabel="MSE", title="GRU-BC Loss")
    ax1.legend(); ax1.grid(True, ls=":", alpha=0.6)
    ax2.plot(epochs, history["train_mae"], label="Train MAE", color="#2563EB", lw=2)
    ax2.plot(epochs, history["val_mae"],   label="Val MAE",   color="#DC2626", lw=2)
    ax2.axvline(best_epoch, color="#16A34A", ls="--", label=f"Best (ep {best_epoch})")
    ax2.set(xlabel="Epoch", ylabel="MAE", title="GRU-BC Action Error")
    ax2.legend(); ax2.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Training curves -> '{output_path}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train GRU-based BC policy on Meta-World.")
    parser.add_argument("--data-dir",    type=str, default="dataset",          help="Directory with parquet splits.")
    parser.add_argument("--output-dir",  type=str, default="models",           help="Checkpoint + plot output directory.")
    parser.add_argument("--epochs",      type=int, default=35,                 help="Training epochs (default 35, same as MLP).")
    parser.add_argument("--batch-size",  type=int, default=256,                help="Minibatch size.")
    parser.add_argument("--lr",          type=float, default=1e-3,             help="Initial learning rate.")
    parser.add_argument("--weight-decay",type=float, default=1e-4,             help="AdamW weight decay.")
    parser.add_argument("--hidden-dim",  type=int, default=256,                help="GRU hidden dimension.")
    parser.add_argument("--history-len", type=int, default=8,                  help="Number of past observations per input sequence.")
    parser.add_argument("--device",      type=str, default="auto",             help="'auto', 'cpu', or 'cuda'.")
    parser.add_argument("--seed",        type=int, default=42,                 help="Random seed (same as MLP baseline).")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  GRU BEHAVIOR CLONING — history_len={args.history_len}")
    print(f"  Device: {device} | Epochs: {args.epochs} | Batch: {args.batch_size}")
    print(f"{'='*60}")

    # -----------------------------------------------------------------
    # 1. Compute normalisation stats from train NPZ (same as MLP)
    # -----------------------------------------------------------------
    # Use NPZ for stats (consistent with MLP baseline), Parquet for sequences
    train_npz_path = os.path.join(args.data_dir, "train.npz")
    if not os.path.exists(train_npz_path):
        raise FileNotFoundError(
            f"Train NPZ not found at '{train_npz_path}'. "
            "Run collect_demonstrations.py first."
        )
    npz = np.load(train_npz_path)
    train_obs_raw = npz["obs"].astype(np.float32)
    obs_mean = train_obs_raw.mean(axis=0)
    obs_std  = train_obs_raw.std(axis=0) + 1e-6

    obs_dim = obs_mean.shape[0]
    act_dim = npz["actions"].shape[1]
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"  Normalisation: mean={obs_mean[:3]} ... std={obs_std[:3]} ...")

    # -----------------------------------------------------------------
    # 2. Build sequence datasets
    # -----------------------------------------------------------------
    print("\nBuilding sequence datasets (this may take ~30s for H=8)...")
    t0 = time.perf_counter()
    train_ds = SequenceDataset(
        os.path.join(args.data_dir, "train.parquet"), obs_mean, obs_std, args.history_len
    )
    val_ds = SequenceDataset(
        os.path.join(args.data_dir, "val.parquet"), obs_mean, obs_std, args.history_len
    )
    print(f"  Train: {len(train_ds):,} samples | Val: {len(val_ds):,} samples  "
          f"({time.perf_counter()-t0:.1f}s)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # -----------------------------------------------------------------
    # 3. Model, optimizer, scheduler — identical hyperparams to MLP
    # -----------------------------------------------------------------
    model = BCPolicyGRU(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=args.hidden_dim).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  BCPolicyGRU: {param_count:,} trainable parameters")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch    = 1
    history       = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": [], "lr": []}
    train_start   = time.perf_counter()

    # -----------------------------------------------------------------
    # 4. Training loop
    # -----------------------------------------------------------------
    print()
    for epoch in range(1, args.epochs + 1):
        ep_t = time.perf_counter()
        tr_loss, tr_mae = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_mae = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_mae"].append(tr_mae)
        history["val_mae"].append(vl_mae)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        is_best = vl_loss < best_val_loss
        if is_best:
            best_val_loss = vl_loss
            best_epoch    = epoch
            ckpt = {
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":         vl_loss,
                "val_mae":          vl_mae,
                "obs_mean":         obs_mean,
                "obs_std":          obs_std,
                "obs_dim":          obs_dim,
                "act_dim":          act_dim,
                "hidden_dim":       args.hidden_dim,
                "history_len":      args.history_len,
                "model_type":       "gru",
            }
            torch.save(ckpt, os.path.join(args.output_dir, "best_bc_gru_model.pt"))

        star = " *" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{args.epochs}]  "
            f"Train MSE={tr_loss:.5f}  Val MSE={vl_loss:.5f}  "
            f"Train MAE={tr_mae:.4f}  Val MAE={vl_mae:.4f}  "
            f"({time.perf_counter()-ep_t:.1f}s){star}"
        )

    total_time = time.perf_counter() - train_start
    print(f"\nTraining done in {total_time:.0f}s ({total_time/60:.1f} min).")
    print(f"Best val loss {best_val_loss:.5f} at epoch {best_epoch}.")

    # -----------------------------------------------------------------
    # 5. Training curves + metrics JSON
    # -----------------------------------------------------------------
    plot_training_curves(
        history,
        os.path.join(args.output_dir, "training_curves_gru.png"),
        best_epoch,
    )
    metrics = {
        "model_type":      "gru",
        "history_len":     args.history_len,
        "hidden_dim":      args.hidden_dim,
        "epochs":          args.epochs,
        "best_epoch":      best_epoch,
        "best_val_loss":   best_val_loss,
        "total_train_sec": total_time,
        "history":         history,
    }
    metrics_path = os.path.join(args.output_dir, "training_metrics_gru.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Metrics -> '{metrics_path}'")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
