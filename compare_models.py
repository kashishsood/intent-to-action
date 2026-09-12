#!/usr/bin/env python3
"""
compare_models.py
-----------------
Runs both the MLP baseline and the GRU history-based model through the same
evaluation pipeline (step-wise MAE/MSE on the held-out test set + 50-episode
closed-loop rollout) and appends a side-by-side comparison table to
eval_report.md.

Usage:
    python compare_models.py
    python compare_models.py \\
        --mlp-path  models/best_bc_model.pt \\
        --gru-path  models/best_bc_gru_model.pt \\
        --test-file dataset/test.parquet \\
        --episodes-per-task 50 \\
        --device cpu

Output:
    - Prints the comparison table to stdout.
    - Appends a new section ("Model Comparison: MLP vs GRU") to eval_report.md.
    - Writes comparison_results.json with raw numbers for downstream use.
"""

import argparse
import datetime
import json
import os
import sys
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tabulate import tabulate
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="metaworld")

import metaworld
from train_bc import BCPolicyMLP
from train_bc_gru import BCPolicyGRU

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TASK_NAMES = [
    "reach-v3",
    "pick-place-v3",
    "door-open-v3",
    "drawer-open-v3",
    "button-press-topdown-v3",
]


# ---------------------------------------------------------------------------
# Step-wise offline metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_stepwise_errors_mlp(
    model: BCPolicyMLP,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    test_parquet_path: str,
    device: torch.device,
) -> Dict[str, Dict]:
    """Compute per-task MAE/MSE for MLP (single-step, no sequence)."""
    df = pd.read_parquet(test_parquet_path)
    model.eval()
    results = {}
    for task in TASK_NAMES:
        grp = df[df["task_name"] == task]
        if len(grp) == 0:
            continue
        obs_raw = np.array(grp["obs"].tolist(), dtype=np.float32)
        act_true = np.array(grp["action"].tolist(), dtype=np.float32)
        obs_norm = (obs_raw - obs_mean) / obs_std
        pred = model(torch.from_numpy(obs_norm).to(device)).cpu().numpy()
        results[task] = {
            "step_mae": float(np.mean(np.abs(pred - act_true))),
            "step_mse": float(np.mean((pred - act_true) ** 2)),
            "test_steps": len(grp),
        }
    return results


@torch.no_grad()
def compute_stepwise_errors_gru(
    model: BCPolicyGRU,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    test_parquet_path: str,
    history_len: int,
    device: torch.device,
) -> Dict[str, Dict]:
    """
    Compute per-task MAE/MSE for GRU.
    Processes each episode sequentially (maintaining hidden state across steps)
    to match exactly what happens in the closed-loop rollout.
    """
    df = pd.read_parquet(test_parquet_path)
    model.eval()
    results = {}
    group_cols = (
        ["task_name", "global_episode_id"]
        if "global_episode_id" in df.columns
        else ["task_name", "episode_id"]
    )

    task_preds: Dict[str, List] = {t: [] for t in TASK_NAMES}
    task_trues: Dict[str, List] = {t: [] for t in TASK_NAMES}

    for (task, _ep), ep_df in df.groupby(group_cols, sort=False):
        if task not in TASK_NAMES:
            continue
        ep_df = ep_df.sort_values("timestep")
        obs_raw = np.array(ep_df["obs"].tolist(), dtype=np.float32)
        acts    = np.array(ep_df["action"].tolist(), dtype=np.float32)
        obs_norm = (obs_raw - obs_mean) / obs_std

        # Reset hidden state at episode boundary
        h = torch.zeros(model.num_gru_layers, 1, model.hidden_dim, device=device)

        for t in range(len(obs_norm)):
            obs_t = torch.from_numpy(obs_norm[t]).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,obs_dim)
            pred, h = model(obs_t, h)
            task_preds[task].append(pred.squeeze(0).cpu().numpy())
            task_trues[task].append(acts[t])

    for task in TASK_NAMES:
        if not task_preds[task]:
            continue
        preds = np.stack(task_preds[task])
        trues = np.stack(task_trues[task])
        results[task] = {
            "step_mae":   float(np.mean(np.abs(preds - trues))),
            "step_mse":   float(np.mean((preds - trues) ** 2)),
            "test_steps": len(trues),
        }
    return results


# ---------------------------------------------------------------------------
# Closed-loop rollouts
# ---------------------------------------------------------------------------

def run_rollouts_mlp(
    model: BCPolicyMLP,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    episodes_per_task: int,
    max_steps: int,
    device: torch.device,
    seed: int,
) -> Dict[str, Dict]:
    mt10 = metaworld.MT10()
    results = {}
    print("\n[MLP] Closed-loop rollouts...")
    for task in TASK_NAMES:
        env_cls = mt10.train_classes[task]
        env = env_cls()
        tasks = [t for t in mt10.train_tasks if t.env_name == task]
        success_count = 0
        for ep in tqdm(range(episodes_per_task), desc=f"  MLP {task}", unit="ep"):
            env.set_task(tasks[ep % len(tasks)])
            obs, _ = env.reset(seed=seed + ep * 17)
            ep_success = False
            for _ in range(max_steps):
                norm = (obs - obs_mean) / obs_std
                obs_t = torch.from_numpy(norm).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    action = model(obs_t).squeeze(0).cpu().numpy()
                obs, _r, terminated, truncated, info = env.step(action)
                if float(info.get("success", 0.0)) > 0.5:
                    ep_success = True
                if terminated or truncated:
                    break
            if ep_success:
                success_count += 1
        results[task] = {
            "episodes":    episodes_per_task,
            "successes":   success_count,
            "success_rate": success_count / episodes_per_task * 100.0,
        }
    return results


def run_rollouts_gru(
    model: BCPolicyGRU,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    episodes_per_task: int,
    max_steps: int,
    device: torch.device,
    seed: int,
) -> Dict[str, Dict]:
    mt10 = metaworld.MT10()
    results = {}
    print("\n[GRU] Closed-loop rollouts...")
    for task in TASK_NAMES:
        env_cls = mt10.train_classes[task]
        env = env_cls()
        tasks = [t for t in mt10.train_tasks if t.env_name == task]
        success_count = 0
        for ep in tqdm(range(episodes_per_task), desc=f"  GRU {task}", unit="ep"):
            env.set_task(tasks[ep % len(tasks)])
            obs, _ = env.reset(seed=seed + ep * 17)
            # Reset hidden state at episode start
            model.reset_hidden(device)
            ep_success = False
            for _ in range(max_steps):
                norm = (obs - obs_mean) / obs_std
                obs_t = torch.from_numpy(norm).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    action = model.step(obs_t).cpu().numpy()
                obs, _r, terminated, truncated, info = env.step(action)
                if float(info.get("success", 0.0)) > 0.5:
                    ep_success = True
                if terminated or truncated:
                    break
            if ep_success:
                success_count += 1
        results[task] = {
            "episodes":    episodes_per_task,
            "successes":   success_count,
            "success_rate": success_count / episodes_per_task * 100.0,
        }
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_comparison_table(
    mlp_offline: Dict,
    mlp_rollout: Dict,
    gru_offline: Dict,
    gru_rollout: Dict,
) -> str:
    """Return a markdown comparison table."""
    header = (
        "| Task | MLP MAE | MLP MSE | MLP Success | "
        "GRU MAE | GRU MSE | GRU Success | MAE Δ | Success Δ |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    rows = []
    for task in TASK_NAMES:
        mo = mlp_offline.get(task, {})
        mr = mlp_rollout.get(task, {})
        go = gru_offline.get(task, {})
        gr = gru_rollout.get(task, {})

        mlp_mae  = mo.get("step_mae", float("nan"))
        mlp_mse  = mo.get("step_mse", float("nan"))
        mlp_succ = mr.get("success_rate", float("nan"))
        gru_mae  = go.get("step_mae", float("nan"))
        gru_mse  = go.get("step_mse", float("nan"))
        gru_succ = gr.get("success_rate", float("nan"))

        mae_delta  = gru_mae  - mlp_mae   # negative = GRU better
        succ_delta = gru_succ - mlp_succ  # positive = GRU better

        # Bold the better success rate
        mlp_succ_s = f"**{mlp_succ:.1f}%**" if mlp_succ > gru_succ  else f"{mlp_succ:.1f}%"
        gru_succ_s = f"**{gru_succ:.1f}%**" if gru_succ >= mlp_succ else f"{gru_succ:.1f}%"
        delta_s    = f"{succ_delta:+.1f}pp"

        rows.append(
            f"| `{task}` "
            f"| {mlp_mae:.4f} | {mlp_mse:.5f} | {mlp_succ_s} "
            f"| {gru_mae:.4f} | {gru_mse:.5f} | {gru_succ_s} "
            f"| {mae_delta:+.4f} | {delta_s} |"
        )

    return header + "\n" + "\n".join(rows)


def append_comparison_to_report(
    report_path: str,
    table_md: str,
    mlp_rollout: Dict,
    gru_rollout: Dict,
    gru_history_len: int,
    episodes_per_task: int,
):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pick_mlp  = mlp_rollout.get("pick-place-v3", {}).get("success_rate", float("nan"))
    pick_gru  = gru_rollout.get("pick-place-v3", {}).get("success_rate", float("nan"))
    pick_delta = pick_gru - pick_mlp

    # Regression check for tasks that were 100%
    perfect_tasks = ["door-open-v3", "drawer-open-v3", "button-press-topdown-v3"]
    regression_notes = []
    for t in perfect_tasks:
        gsr = gru_rollout.get(t, {}).get("success_rate", 0.0)
        if gsr < 90.0:
            regression_notes.append(f"- `{t}`: GRU success = {gsr:.1f}% (baseline was 100%) — **regression**")
        else:
            regression_notes.append(f"- `{t}`: GRU success = {gsr:.1f}% — no regression")
    regression_md = "\n".join(regression_notes)

    pick_interpretation = ""
    if pick_delta > 5:
        pick_interpretation = (
            f"Pick-place success improved by {pick_delta:+.1f}pp ({pick_mlp:.1f}% → {pick_gru:.1f}%), "
            "which is consistent with the hypothesis that temporal context helps compensate for "
            "positional drift near the contact boundary. This is suggestive but not conclusive: "
            "a controlled ablation varying history length would be needed to isolate the effect."
        )
    elif pick_delta < -5:
        pick_interpretation = (
            f"Pick-place success decreased by {pick_delta:.1f}pp ({pick_mlp:.1f}% → {pick_gru:.1f}%). "
            "This is inconsistent with the compounding-error hypothesis as the primary failure mode, "
            "or the GRU is not yet benefiting from its history due to training capacity or sequence construction."
        )
    else:
        pick_interpretation = (
            f"Pick-place success changed by only {pick_delta:+.1f}pp ({pick_mlp:.1f}% → {pick_gru:.1f}%), "
            "within noise for 50-episode evaluation. No conclusive evidence that temporal context "
            "addresses the observed failure mode at this history length and training budget."
        )

    section = f"""

---

## Model Comparison: MLP Baseline vs GRU (History={gru_history_len})

*Generated: {timestamp} | {episodes_per_task} closed-loop episodes per task*

The GRU model uses the last **{gru_history_len} observations** as context.
Both models were trained for the same number of epochs with identical
AdamW optimizer settings and the same train/val/test data splits.
Δ columns show GRU minus MLP (negative MAE Δ = GRU more accurate; positive Success Δ = GRU better).

{table_md}

### Pick-place-v3 (Primary Test of the Compounding-Error Hypothesis)

{pick_interpretation}

### Regression Check (Tasks at 100% MLP Baseline)

{regression_md}

### Interpretation note

Differences between a 50-episode evaluation are subject to stochastic noise
(95% CI for a binomial proportion at n=50 spans roughly ±14pp at p=0.5).
A result should be replicated with more episodes or multiple seeds before
being interpreted as a definitive improvement or regression.
"""

    with open(report_path, "a", encoding="utf-8") as f:
        f.write(section)
    print(f"[OK] Comparison section appended to '{report_path}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare MLP vs GRU BC policies.")
    parser.add_argument("--mlp-path",         type=str, default="models/best_bc_model.pt",     help="MLP checkpoint.")
    parser.add_argument("--gru-path",         type=str, default="models/best_bc_gru_model.pt", help="GRU checkpoint.")
    parser.add_argument("--test-file",        type=str, default="dataset/test.parquet",         help="Test parquet.")
    parser.add_argument("--episodes-per-task",type=int, default=50,                             help="Rollout episodes per task.")
    parser.add_argument("--max-steps",        type=int, default=500,                            help="Max steps per episode.")
    parser.add_argument("--output-report",    type=str, default="eval_report.md",               help="Report to append to.")
    parser.add_argument("--results-json",     type=str, default="models/comparison_results.json", help="JSON output path.")
    parser.add_argument("--device",           type=str, default="cpu",                          help="Compute device.")
    parser.add_argument("--seed",             type=int, default=2026,                           help="Random seed.")
    parser.add_argument("--skip-mlp-rollout", action="store_true",
                        help="Skip MLP rollout and read from existing eval_report.md numbers.")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # Load MLP
    # ------------------------------------------------------------------
    if not os.path.exists(args.mlp_path):
        print(f"ERROR: MLP checkpoint not found at '{args.mlp_path}'")
        sys.exit(1)
    mlp_ckpt = torch.load(args.mlp_path, map_location=device)
    mlp_model = BCPolicyMLP(
        obs_dim=mlp_ckpt.get("obs_dim", 39),
        act_dim=mlp_ckpt.get("act_dim", 4),
        hidden_dim=mlp_ckpt.get("hidden_dim", 256),
    )
    mlp_model.load_state_dict(mlp_ckpt["model_state_dict"])
    mlp_model.to(device).eval()
    mlp_obs_mean = mlp_ckpt["obs_mean"]
    mlp_obs_std  = mlp_ckpt["obs_std"]
    print(f"[OK] Loaded MLP from '{args.mlp_path}'")

    # ------------------------------------------------------------------
    # Load GRU
    # ------------------------------------------------------------------
    if not os.path.exists(args.gru_path):
        print(f"ERROR: GRU checkpoint not found at '{args.gru_path}'.")
        print("Run 'python train_bc_gru.py' first.")
        sys.exit(1)
    gru_ckpt = torch.load(args.gru_path, map_location=device)
    gru_history_len = gru_ckpt.get("history_len", 8)
    gru_model = BCPolicyGRU(
        obs_dim=gru_ckpt.get("obs_dim", 39),
        act_dim=gru_ckpt.get("act_dim", 4),
        hidden_dim=gru_ckpt.get("hidden_dim", 256),
    )
    gru_model.load_state_dict(gru_ckpt["model_state_dict"])
    gru_model.to(device).eval()
    gru_obs_mean = gru_ckpt["obs_mean"]
    gru_obs_std  = gru_ckpt["obs_std"]
    print(f"[OK] Loaded GRU (history_len={gru_history_len}) from '{args.gru_path}'")

    # ------------------------------------------------------------------
    # Step-wise offline metrics
    # ------------------------------------------------------------------
    print("\nComputing step-wise offline metrics...")
    mlp_offline = compute_stepwise_errors_mlp(mlp_model, mlp_obs_mean, mlp_obs_std, args.test_file, device)
    gru_offline = compute_stepwise_errors_gru(gru_model, gru_obs_mean, gru_obs_std, args.test_file, gru_history_len, device)

    # ------------------------------------------------------------------
    # Closed-loop rollouts
    # ------------------------------------------------------------------
    if args.skip_mlp_rollout:
        # Read baseline numbers from eval_baseline.json if available
        print("\n[--skip-mlp-rollout] Using baseline success rates from eval_baseline.json")
        baseline_path = "eval_baseline.json"
        if os.path.exists(baseline_path):
            with open(baseline_path) as f:
                bl = json.load(f)
            mlp_rollout = {
                task: {"episodes": args.episodes_per_task, "success_rate": vals["baseline_success_rate"],
                       "successes": int(vals["baseline_success_rate"] / 100 * args.episodes_per_task)}
                for task, vals in bl["tasks"].items()
            }
        else:
            print("  eval_baseline.json not found — running MLP rollout anyway.")
            args.skip_mlp_rollout = False

    if not args.skip_mlp_rollout:
        mlp_rollout = run_rollouts_mlp(
            mlp_model, mlp_obs_mean, mlp_obs_std,
            args.episodes_per_task, args.max_steps, device, args.seed
        )

    gru_rollout = run_rollouts_gru(
        gru_model, gru_obs_mean, gru_obs_std,
        args.episodes_per_task, args.max_steps, device, args.seed
    )

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    table_rows = []
    for task in TASK_NAMES:
        mo, mr = mlp_offline.get(task, {}), mlp_rollout.get(task, {})
        go, gr = gru_offline.get(task, {}), gru_rollout.get(task, {})
        table_rows.append([
            task,
            f"{mo.get('step_mae', float('nan')):.4f}",
            f"{mr.get('success_rate', float('nan')):.1f}%",
            f"{go.get('step_mae', float('nan')):.4f}",
            f"{gr.get('success_rate', float('nan')):.1f}%",
            f"{gr.get('success_rate', float('nan')) - mr.get('success_rate', float('nan')):+.1f}pp",
        ])
    headers = ["Task", "MLP MAE", "MLP Success", "GRU MAE", "GRU Success", "Success Δ"]
    print("\n" + "=" * 80)
    print("  MODEL COMPARISON: MLP BASELINE vs GRU")
    print("=" * 80)
    print(tabulate(table_rows, headers=headers, tablefmt="github"))
    print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # Append to eval_report.md
    # ------------------------------------------------------------------
    table_md = build_comparison_table(mlp_offline, mlp_rollout, gru_offline, gru_rollout)
    append_comparison_to_report(
        args.output_report,
        table_md,
        mlp_rollout,
        gru_rollout,
        gru_history_len,
        args.episodes_per_task,
    )

    # ------------------------------------------------------------------
    # Save raw results JSON
    # ------------------------------------------------------------------
    comparison_results = {
        "generated": datetime.datetime.now().isoformat(),
        "gru_history_len": gru_history_len,
        "episodes_per_task": args.episodes_per_task,
        "mlp": {task: {**mlp_offline.get(task, {}), **mlp_rollout.get(task, {})} for task in TASK_NAMES},
        "gru": {task: {**gru_offline.get(task, {}), **gru_rollout.get(task, {})} for task in TASK_NAMES},
    }
    os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
    with open(args.results_json, "w") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"[OK] Raw results -> '{args.results_json}'")


if __name__ == "__main__":
    main()
