#!/usr/bin/env python3
"""
eval_5task_gru_window.py
-------------------------
Re-evaluates the original 5-task GRU model under method=window (and method=step)
against the MLP baseline across the 5 MT10 tasks (50 episodes/task, seed=2026).
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

torch.set_num_threads(8)

TASK_NAMES = [
    "reach-v3",
    "pick-place-v3",
    "door-open-v3",
    "drawer-open-v3",
    "button-press-topdown-v3",
]


@torch.no_grad()
def compute_stepwise_errors_mlp(
    model: BCPolicyMLP,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    test_parquet_path: str,
    device: torch.device,
) -> Dict[str, Dict]:
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
def compute_stepwise_errors_gru_window(
    model: BCPolicyGRU,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    test_parquet_path: str,
    history_len: int,
    device: torch.device,
) -> Dict[str, Dict]:
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
        acts = np.array(ep_df["action"].tolist(), dtype=np.float32)

        # Sliding window with repeat-first padding
        buffer = [obs_raw[0].copy()] * history_len
        for t in range(len(obs_raw)):
            buffer.append(obs_raw[t].copy())
            if len(buffer) > history_len:
                buffer.pop(0)
            window = np.array(buffer, dtype=np.float32)
            window_norm = (window - obs_mean) / obs_std
            window_t = torch.from_numpy(window_norm).unsqueeze(0).to(device)
            pred, _ = model(window_t)
            task_preds[task].append(pred.squeeze(0).cpu().numpy())
            task_trues[task].append(acts[t])

    for task in TASK_NAMES:
        if not task_preds[task]:
            continue
        preds = np.stack(task_preds[task])
        trues = np.stack(task_trues[task])
        results[task] = {
            "step_mae": float(np.mean(np.abs(preds - trues))),
            "step_mse": float(np.mean((preds - trues) ** 2)),
            "test_steps": len(trues),
        }
    return results


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
        ep_records = []
        for ep in tqdm(range(episodes_per_task), desc=f"  MLP {task}", unit="ep"):
            env.set_task(tasks[ep % len(tasks)])
            ep_seed = seed + ep * 17
            obs, _ = env.reset(seed=ep_seed)
            ep_success = False
            for step in range(max_steps):
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
            ep_records.append(1 if ep_success else 0)
        results[task] = {
            "episodes": episodes_per_task,
            "successes": success_count,
            "success_rate": success_count / episodes_per_task * 100.0,
            "vector": ep_records,
        }
    return results


def run_rollouts_gru_window(
    model: BCPolicyGRU,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    history_len: int,
    episodes_per_task: int,
    max_steps: int,
    device: torch.device,
    seed: int,
) -> Dict[str, Dict]:
    mt10 = metaworld.MT10()
    results = {}
    print("\n[GRU window] Closed-loop rollouts (method=window)...")
    for task in TASK_NAMES:
        env_cls = mt10.train_classes[task]
        env = env_cls()
        tasks = [t for t in mt10.train_tasks if t.env_name == task]
        success_count = 0
        ep_records = []
        for ep in tqdm(range(episodes_per_task), desc=f"  GRU-win {task}", unit="ep"):
            env.set_task(tasks[ep % len(tasks)])
            ep_seed = seed + ep * 17
            obs, _ = env.reset(seed=ep_seed)
            ep_success = False

            # Sliding buffer of unnormalized observations, repeat first
            buffer = [obs.copy()] * history_len

            for step in range(max_steps):
                buffer.append(obs.copy())
                if len(buffer) > history_len:
                    buffer.pop(0)

                window = np.array(buffer, dtype=np.float32)
                window_norm = (window - obs_mean) / obs_std
                window_t = torch.from_numpy(window_norm).unsqueeze(0).to(device)

                with torch.no_grad():
                    action, _ = model(window_t)
                    action = action.squeeze(0).cpu().numpy()

                obs, _r, terminated, truncated, info = env.step(action)
                if float(info.get("success", 0.0)) > 0.5:
                    ep_success = True
                if terminated or truncated:
                    break

            if ep_success:
                success_count += 1
            ep_records.append(1 if ep_success else 0)

        results[task] = {
            "episodes": episodes_per_task,
            "successes": success_count,
            "success_rate": success_count / episodes_per_task * 100.0,
            "vector": ep_records,
        }
    return results


def run_rollouts_gru_step(
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
    print("\n[GRU step] Closed-loop rollouts (method=step)...")
    for task in TASK_NAMES:
        env_cls = mt10.train_classes[task]
        env = env_cls()
        tasks = [t for t in mt10.train_tasks if t.env_name == task]
        success_count = 0
        ep_records = []
        for ep in tqdm(range(episodes_per_task), desc=f"  GRU-step {task}", unit="ep"):
            env.set_task(tasks[ep % len(tasks)])
            ep_seed = seed + ep * 17
            obs, _ = env.reset(seed=ep_seed)
            model.reset_hidden(device)
            ep_success = False

            for step in range(max_steps):
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
            ep_records.append(1 if ep_success else 0)

        results[task] = {
            "episodes": episodes_per_task,
            "successes": success_count,
            "success_rate": success_count / episodes_per_task * 100.0,
            "vector": ep_records,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate 5-task GRU under window and step.")
    parser.add_argument("--mlp-path", default="models/best_bc_model.pt")
    parser.add_argument("--gru-path", default="models/best_bc_gru_model.pt")
    parser.add_argument("--test-file", default="dataset/test.parquet")
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-json", default="models/comparison_5tasks_window_results.json")
    parser.add_argument("--skip-mlp-rollout", action="store_true")
    parser.add_argument("--eval-step", action="store_true", help="Also evaluate method=step")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Load MLP
    print(f"Loading MLP from {args.mlp_path}...", flush=True)
    mlp_ckpt = torch.load(args.mlp_path, map_location=device)
    mlp_model = BCPolicyMLP(
        obs_dim=mlp_ckpt.get("obs_dim", 39),
        act_dim=mlp_ckpt.get("act_dim", 4),
        hidden_dim=mlp_ckpt.get("hidden_dim", 256),
    )
    mlp_model.load_state_dict(mlp_ckpt["model_state_dict"])
    mlp_model.to(device).eval()
    mlp_obs_mean = mlp_ckpt["obs_mean"]
    mlp_obs_std = mlp_ckpt["obs_std"]

    # 2. Load GRU
    print(f"Loading GRU from {args.gru_path}...", flush=True)
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
    gru_obs_std = gru_ckpt["obs_std"]

    # 3. Offline metrics
    print("\nComputing offline test metrics...", flush=True)
    mlp_offline = compute_stepwise_errors_mlp(mlp_model, mlp_obs_mean, mlp_obs_std, args.test_file, device)
    gru_offline = compute_stepwise_errors_gru_window(gru_model, gru_obs_mean, gru_obs_std, args.test_file, gru_history_len, device)

    # 4. Rollouts
    if args.skip_mlp_rollout:
        print("[--skip-mlp-rollout] Using locked baseline from eval_report.md (seed=2026)", flush=True)
        mlp_rollout = {
            "reach-v3": {"episodes": 50, "successes": 31, "success_rate": 62.0, "vector": []},
            "pick-place-v3": {"episodes": 50, "successes": 11, "success_rate": 22.0, "vector": []},
            "door-open-v3": {"episodes": 50, "successes": 50, "success_rate": 100.0, "vector": []},
            "drawer-open-v3": {"episodes": 50, "successes": 50, "success_rate": 100.0, "vector": []},
            "button-press-topdown-v3": {"episodes": 50, "successes": 50, "success_rate": 100.0, "vector": []},
        }
    else:
        mlp_rollout = run_rollouts_mlp(mlp_model, mlp_obs_mean, mlp_obs_std, args.episodes_per_task, args.max_steps, device, args.seed)

    gru_win_rollout = run_rollouts_gru_window(gru_model, gru_obs_mean, gru_obs_std, gru_history_len, args.episodes_per_task, args.max_steps, device, args.seed)

    if args.eval_step:
        gru_step_rollout = run_rollouts_gru_step(gru_model, gru_obs_mean, gru_obs_std, args.episodes_per_task, args.max_steps, device, args.seed)
    else:
        # Default step values from historical f3f3d21 run for reference
        gru_step_rollout = {
            "reach-v3": {"episodes": 50, "successes": 27, "success_rate": 54.0, "vector": []},
            "pick-place-v3": {"episodes": 50, "successes": 16, "success_rate": 32.0, "vector": []},
            "door-open-v3": {"episodes": 50, "successes": 50, "success_rate": 100.0, "vector": []},
            "drawer-open-v3": {"episodes": 50, "successes": 50, "success_rate": 100.0, "vector": []},
            "button-press-topdown-v3": {"episodes": 50, "successes": 50, "success_rate": 100.0, "vector": []},
        }

    # 5. Summary Table
    rows = []
    for t in TASK_NAMES:
        mlp_succ = mlp_rollout[t]["success_rate"]
        gwin_succ = gru_win_rollout[t]["success_rate"]
        gstep_succ = gru_step_rollout[t]["success_rate"]
        delta_win = gwin_succ - mlp_succ
        delta_step = gstep_succ - mlp_succ

        rows.append([
            t,
            f"{mlp_offline[t]['step_mae']:.4f}",
            f"{mlp_rollout[t]['successes']}/{args.episodes_per_task} ({mlp_succ:.1f}%)",
            f"{gru_offline[t]['step_mae']:.4f}",
            f"{gru_win_rollout[t]['successes']}/{args.episodes_per_task} ({gwin_succ:.1f}%)",
            f"{delta_win:+.1f}pp",
            f"{gru_step_rollout[t]['successes']}/{args.episodes_per_task} ({gstep_succ:.1f}%)",
            f"{delta_step:+.1f}pp",
        ])

    headers = [
        "Task",
        "MLP MAE",
        "MLP Success",
        "GRU MAE",
        "GRU (window) Success",
        "Win Δ",
        "GRU (step) Success",
        "Step Δ",
    ]

    print("\n" + "=" * 105)
    print("  ORIGINAL 5-TASK BENCHMARK: MLP vs GRU (method=window vs method=step)")
    print("=" * 105)
    print(tabulate(rows, headers=headers, tablefmt="github"))
    print("=" * 105 + "\n")

    # Save JSON
    out_data = {
        "generated": datetime.datetime.now().isoformat(),
        "episodes_per_task": args.episodes_per_task,
        "seed": args.seed,
        "mlp_offline": mlp_offline,
        "gru_offline": gru_offline,
        "mlp_rollout": mlp_rollout,
        "gru_win_rollout": gru_win_rollout,
        "gru_step_rollout": gru_step_rollout,
    }
    with open(args.output_json, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
