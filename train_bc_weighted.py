#!/usr/bin/env python3
"""
train_bc_weighted.py
---------------------
Proximity-Weighted Behavior Cloning (BC) Training Script.

Implements proximity-weighted BC loss targeting the contact/grasp boundary:
- Distance metric: d = ||hand_pos - obj_pos||_2 (meters) from obs[0:3] and obs[4:7]
- Grasp threshold: d0 = 0.035m (~3.5cm contact boundary)
- Floor value: W_floor = 1.0 (baseline weight outside vicinity)
- Peak weight: W_max = 5.0 (for d <= 0.035m)
- Smooth decay: Gaussian decay with sigma = 0.02m (2cm) for d > 0.035m
- Held-out evaluation band: 0.045m < d <= 0.080m (pre-contact alignment corridor)
  tracked as a separate isolated validation/test metric.
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

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from train_bc import BCPolicyMLP


def compute_sample_weights(obs_raw: np.ndarray, floor_val: float = 1.0, max_weight: float = 5.0, d0: float = 0.035, sigma: float = 0.02) -> np.ndarray:
    """
    Computes sample-wise loss weights based on gripper-to-object proximity.
    w(d) = floor_val + (max_weight - floor_val) * exp(-max(0, d - d0)^2 / (2 * sigma^2))
    """
    hand_pos = obs_raw[:, 0:3]
    obj_pos = obs_raw[:, 4:7]
    dists = np.linalg.norm(hand_pos - obj_pos, axis=1)
    
    excess_dist = np.maximum(0.0, dists - d0)
    decay = np.exp(- (excess_dist ** 2) / (2.0 * (sigma ** 2)))
    weights = floor_val + (max_weight - floor_val) * decay
    return weights.astype(np.float32), dists.astype(np.float32)


def evaluate_with_heldout_band(
    model: nn.Module,
    obs_t: torch.Tensor,
    act_t: torch.Tensor,
    dists_t: torch.Tensor,
    weights_t: torch.Tensor,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Evaluates overall metrics as well as isolated metrics on the held-out transition band (0.045m < d <= 0.080m)."""
    model.eval()
    dataset = TensorDataset(obs_t, act_t, dists_t, weights_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    heldout_sq_err = 0.0
    heldout_abs_err = 0.0
    heldout_samples = 0

    with torch.no_grad():
        for b_obs, b_act, b_dist, b_w in loader:
            b_obs, b_act, b_dist, b_w = b_obs.to(device), b_act.to(device), b_dist.to(device), b_w.to(device)
            pred = model(b_obs)
            
            sq_err = (pred - b_act) ** 2
            abs_err = torch.abs(pred - b_act)
            
            # Overall unweighted MAE and weighted loss
            weighted_sq_err = b_w.unsqueeze(1) * sq_err
            total_loss += weighted_sq_err.mean().item() * b_obs.size(0)
            total_mae += abs_err.mean().item() * b_obs.size(0)
            total_samples += b_obs.size(0)

            # Held-out band: 0.045 < d <= 0.080 m
            mask = (b_dist > 0.045) & (b_dist <= 0.080)
            if mask.sum() > 0:
                heldout_sq_err += sq_err[mask].sum().item()
                heldout_abs_err += abs_err[mask].sum().item()
                heldout_samples += mask.sum().item() * b_act.size(1)

    return {
        "loss": total_loss / total_samples,
        "mae": total_mae / total_samples,
        "heldout_mse": (heldout_sq_err / heldout_samples) if heldout_samples > 0 else 0.0,
        "heldout_mae": (heldout_abs_err / heldout_samples) if heldout_samples > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Proximity-Weighted Behavior Cloning policy.")
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w-floor", type=float, default=1.0)
    parser.add_argument("--w-max", type=float, default=5.0)
    parser.add_argument("--d0", type=float, default=0.035)
    parser.add_argument("--sigma", type=float, default=0.02)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\n" + "=" * 80)
    print(" PHASE 3: PROXIMITY-WEIGHTED BEHAVIOR CLONING (BC) TRAINING")
    print(f" Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print(f" Weighting formula: w(d) = {args.w_floor} + {args.w_max - args.w_floor} * exp(-max(0, d - {args.d0})^2 / (2 * {args.sigma}^2))")
    print(f" Floor value: {args.w_floor:.1f} | Peak weight: {args.w_max:.1f} (at d <= {args.d0*100:.1f}cm)")
    print(f" Held-Out Evaluation Band: 4.5cm < d <= 8.0cm (pre-contact corridor)")
    print("=" * 80)

    # 1. Load data
    train_npz = np.load(os.path.join(args.data_dir, "train.npz"))
    val_npz = np.load(os.path.join(args.data_dir, "val.npz"))
    test_npz = np.load(os.path.join(args.data_dir, "test.npz"))

    train_obs, train_act = train_npz["obs"].astype(np.float32), train_npz["actions"].astype(np.float32)
    val_obs, val_act = val_npz["obs"].astype(np.float32), val_npz["actions"].astype(np.float32)
    test_obs, test_act = test_npz["obs"].astype(np.float32), test_npz["actions"].astype(np.float32)

    # 2. Compute sample weights and distances
    train_weights, train_dists = compute_sample_weights(train_obs, args.w_floor, args.w_max, args.d0, args.sigma)
    val_weights, val_dists = compute_sample_weights(val_obs, args.w_floor, args.w_max, args.d0, args.sigma)
    test_weights, test_dists = compute_sample_weights(test_obs, args.w_floor, args.w_max, args.d0, args.sigma)

    # 3. Normalisation (strictly identical to baseline MLP)
    obs_mean = train_obs.mean(axis=0)
    obs_std = train_obs.std(axis=0) + 1e-6

    train_obs_norm = (train_obs - obs_mean) / obs_std
    val_obs_norm = (val_obs - obs_mean) / obs_std
    test_obs_norm = (test_obs - obs_mean) / obs_std

    # PyTorch Tensors
    train_obs_t = torch.from_numpy(train_obs_norm)
    train_act_t = torch.from_numpy(train_act)
    train_w_t = torch.from_numpy(train_weights)
    train_d_t = torch.from_numpy(train_dists)

    val_obs_t = torch.from_numpy(val_obs_norm)
    val_act_t = torch.from_numpy(val_act)
    val_w_t = torch.from_numpy(val_weights)
    val_d_t = torch.from_numpy(val_dists)

    test_obs_t = torch.from_numpy(test_obs_norm)
    test_act_t = torch.from_numpy(test_act)
    test_w_t = torch.from_numpy(test_weights)
    test_d_t = torch.from_numpy(test_dists)

    train_ds = TensorDataset(train_obs_t, train_act_t, train_w_t)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # 4. Model (identical architecture to baseline MLP)
    model = BCPolicyMLP(obs_dim=train_obs.shape[1], act_dim=train_act.shape[1], hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None

    history = {
        "train_loss": [], "val_loss": [], "train_mae": [], "val_mae": [],
        "heldout_val_mae": [], "heldout_val_mse": []
    }

    train_start = time.perf_counter()
    print("\nStarting training loop...")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        tr_loss_accum = 0.0
        tr_mae_accum = 0.0
        tr_samples = 0

        for b_obs, b_act, b_w in train_loader:
            b_obs, b_act, b_w = b_obs.to(device), b_act.to(device), b_w.to(device)
            optimizer.zero_grad()
            pred = model(b_obs)
            
            # Element-wise squared error weighted by proximity weight
            loss = (b_w.unsqueeze(1) * ((pred - b_act) ** 2)).mean()
            loss.backward()
            optimizer.step()

            tr_loss_accum += loss.item() * b_obs.size(0)
            tr_mae_accum += torch.abs(pred - b_act).mean().item() * b_obs.size(0)
            tr_samples += b_obs.size(0)

        scheduler.step()
        tr_loss = tr_loss_accum / tr_samples
        tr_mae = tr_mae_accum / tr_samples

        # Validation with held-out band tracking
        val_metrics = evaluate_with_heldout_band(model, val_obs_t, val_act_t, val_d_t, val_w_t, args.batch_size, device)
        vl_loss = val_metrics["loss"]
        vl_mae = val_metrics["mae"]
        ho_mae = val_metrics["heldout_mae"]
        ho_mse = val_metrics["heldout_mse"]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_mae"].append(tr_mae)
        history["val_mae"].append(vl_mae)
        history["heldout_val_mae"].append(ho_mae)
        history["heldout_val_mse"].append(ho_mse)

        is_best = vl_loss < best_val_loss
        if is_best:
            best_val_loss = vl_loss
            best_epoch = epoch
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": vl_loss,
                "val_mae": vl_mae,
                "heldout_val_mae": ho_mae,
                "heldout_val_mse": ho_mse,
                "obs_mean": obs_mean,
                "obs_std": obs_std,
                "obs_dim": train_obs.shape[1],
                "act_dim": train_act.shape[1],
                "hidden_dim": args.hidden_dim,
                "w_floor": args.w_floor,
                "w_max": args.w_max,
                "d0": args.d0,
                "sigma": args.sigma,
            }
            torch.save(best_state, os.path.join(args.output_dir, "best_bc_weighted_model.pt"))

        star = " *" if is_best else ""
        ep_sec = time.perf_counter() - ep_t0
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({ep_sec:4.1f}s) | "
              f"Train Loss={tr_loss:.4f} Val Loss={vl_loss:.4f} | "
              f"Train MAE={tr_mae:.4f} Val MAE={vl_mae:.4f} | "
              f"Held-Out Band MAE={ho_mae:.4f} MSE={ho_mse:.5f}{star}")

    total_time = time.perf_counter() - train_start
    print("-" * 80)
    print(f"Training completed in {total_time:.1f}s. Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}.")

    # Evaluate best model on test set (overall and held-out band)
    model.load_state_dict(best_state["model_state_dict"])
    test_metrics = evaluate_with_heldout_band(model, test_obs_t, test_act_t, test_d_t, test_w_t, args.batch_size, device)
    print(f"\nHeld-Out Test Evaluation (Weighted Model):")
    print(f"  Overall Test MAE:          {test_metrics['mae']:.4f}")
    print(f"  Overall Test Loss (W-MSE): {test_metrics['loss']:.5f}")
    print(f"  Held-Out Band Test MAE:    {test_metrics['heldout_mae']:.4f}")
    print(f"  Held-Out Band Test MSE:    {test_metrics['heldout_mse']:.5f}")

    # Evaluate baseline model on test set for direct comparison
    baseline_ckpt = torch.load(os.path.join(args.output_dir, "best_bc_model.pt"), map_location=device)
    baseline_model = BCPolicyMLP(obs_dim=train_obs.shape[1], act_dim=train_act.shape[1], hidden_dim=args.hidden_dim).to(device)
    baseline_model.load_state_dict(baseline_ckpt["model_state_dict"])
    base_test_metrics = evaluate_with_heldout_band(baseline_model, test_obs_t, test_act_t, test_d_t, test_w_t, args.batch_size, device)
    print(f"\nHeld-Out Test Evaluation (Baseline Model):")
    print(f"  Overall Test MAE:          {base_test_metrics['mae']:.4f}")
    print(f"  Overall Test Loss (W-MSE): {base_test_metrics['loss']:.5f}")
    print(f"  Held-Out Band Test MAE:    {base_test_metrics['heldout_mae']:.4f}")
    print(f"  Held-Out Band Test MSE:    {base_test_metrics['heldout_mse']:.5f}")

    metrics_out = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_time_sec": total_time,
        "weighted_model_test": test_metrics,
        "baseline_model_test": base_test_metrics,
        "history": history,
    }
    with open(os.path.join(args.output_dir, "training_metrics_weighted.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n[OK] Saved metrics to 'models/training_metrics_weighted.json'")


if __name__ == "__main__":
    main()
