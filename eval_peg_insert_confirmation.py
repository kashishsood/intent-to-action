#!/usr/bin/env python3
"""
eval_peg_insert_confirmation.py
--------------------------------
Pre-registered confirmation benchmark on unused peg-insert-side-v3 configurations.
Meta-World ML1('peg-insert-side-v3', seed=42) has 50 task configurations.
The 12-task benchmark evaluated configurations 0..19.
This script evaluates exclusively on the 30 unused configurations (indices 20..49).

Models evaluated:
1. Feedforward: Kin MLP vs Force MLP
2. Recurrent (method=window):
   - Seed 42: Kin GRU vs Force GRU
   - Seed 43: Kin GRU vs Force GRU
   - Seed 44: Kin GRU vs Force GRU

Pre-registered prediction:
If force helps peg-insert, the paired difference is >= +10 percentage points for both architectures.
"""

import argparse
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
import mujoco
from tabulate import tabulate

warnings.filterwarnings("ignore", category=UserWarning, module="metaworld")
import metaworld

from train_12tasks_study import BCPolicyGRU, BCPolicyMLP
from eval_12tasks_paired_pickplace import (
    load_model_and_config,
    get_gripper_geom_ids,
    extract_genuine_contact_wrench,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

UNUSED_INDICES = list(range(20, 50))  # 30 unused task configurations (indices 20..49)

MODELS = [
    ("Kin MLP",        "models/bc_12tasks_mlp_kinematic.pt",   "mlp", "none"),
    ("Force MLP",      "models/bc_12tasks_mlp_force.pt",       "mlp", "none"),
    ("Kin GRU s42",    "models/bc_12tasks_gru_kinematic.pt",   "gru", "window"),
    ("Force GRU s42",  "models/bc_12tasks_gru_force.pt",       "gru", "window"),
    ("Kin GRU s43",    "models/bc_12tasks_gru_kinematic_s43.pt", "gru", "window"),
    ("Force GRU s43",  "models/bc_12tasks_gru_force_s43.pt",   "gru", "window"),
    ("Kin GRU s44",    "models/bc_12tasks_gru_kinematic_s44.pt", "gru", "window"),
    ("Force GRU s44",  "models/bc_12tasks_gru_force_s44.pt",   "gru", "window"),
]


def evaluate_model_on_unused_pegs(
    model_name: str,
    ckpt_path: str,
    arch_type: str,
    method: str,
    tasks_list: list,
    device: torch.device = torch.device("cpu"),
) -> Tuple[dict, str]:
    info = load_model_and_config(ckpt_path, device)
    model = info["model"]
    obs_dim = info["obs_dim"]
    obs_mean = info["obs_mean"]
    obs_std = info["obs_std"]
    is_force_model = (obs_dim == 45)

    log_lines = []
    header = f"================================================================================\n" \
             f"PEG-INSERT UNUSED CONFIG BENCHMARK: {model_name} (arch={arch_type}, method={method})\n" \
             f"================================================================================"
    print(header, flush=True)
    log_lines.append(header)

    t0 = time.time()
    succ_vector = []
    env_cls = metaworld.envs.sawyer_peg_insertion_side_v3.SawyerPegInsertionSideEnvV3

    for idx, cfg_idx in enumerate(UNUSED_INDICES):
        task_obj = tasks_list[cfg_idx]
        env = env_cls()
        env.set_task(task_obj)
        ep_seed = 42 + cfg_idx * 17
        torch.manual_seed(ep_seed)
        np.random.seed(ep_seed)
        obs, _ = env.reset(seed=ep_seed)

        model_internal = env.unwrapped.model
        data_internal = env.unwrapped.data
        gripper_geoms = get_gripper_geom_ids(model_internal) if is_force_model else None

        # History buffer for window method
        if is_force_model:
            wrench0 = extract_genuine_contact_wrench(model_internal, data_internal, gripper_geoms)
            feat0 = np.concatenate([obs, wrench0])
        else:
            feat0 = obs
        norm_feat0 = (feat0 - obs_mean) / obs_std
        hist = [norm_feat0]
        pad = [norm_feat0] * 7

        ep_succ = False
        for step in range(500):
            if is_force_model:
                wrench = extract_genuine_contact_wrench(model_internal, data_internal, gripper_geoms)
                raw_feat = np.concatenate([obs, wrench])
            else:
                raw_feat = obs

            norm_feat = (raw_feat - obs_mean) / obs_std

            with torch.no_grad():
                if arch_type == "mlp":
                    feat_t = torch.from_numpy(norm_feat).float().unsqueeze(0).to(device)
                    action = model(feat_t).squeeze(0).cpu().numpy()
                elif arch_type == "gru" and method == "window":
                    current_hist = pad + hist
                    win = np.array(current_hist[-8:], dtype=np.float32)
                    win_t = torch.from_numpy(win).float().unsqueeze(0).to(device)
                    action, _ = model(win_t, None)
                    action = action.squeeze(0).cpu().numpy()

            obs, reward, term, trunc, info = env.step(action)
            if is_force_model:
                next_wrench = extract_genuine_contact_wrench(model_internal, data_internal, gripper_geoms)
                next_raw = np.concatenate([obs, next_wrench])
            else:
                next_raw = obs
            hist.append((next_raw - obs_mean) / obs_std)

            if float(info.get("success", 0.0)) > 0.5:
                ep_succ = True
            if term or trunc:
                break

        succ_vector.append(1 if ep_succ else 0)
        line = f"[PEG_EP] Config #{cfg_idx:02d} ({idx+1:02d}/30) | Seed {ep_seed:5d} | Success: {ep_succ!s:<5} | Steps: {step+1:3d}"
        print(line, flush=True)
        log_lines.append(line)

    elapsed = time.time() - t0
    total_succ = sum(succ_vector)
    pct = (total_succ / len(UNUSED_INDICES)) * 100.0

    summary = f"\nSUMMARY: {model_name}\n" \
              f"  Success: {total_succ}/{len(UNUSED_INDICES)} ({pct:.1f}%)\n" \
              f"  Vector:  {succ_vector}\n" \
              f"  Elapsed: {elapsed:.1f}s\n" \
              f"--------------------------------------------------------------------------------\n"
    print(summary, flush=True)
    log_lines.append(summary)

    res = {
        "model_name": model_name,
        "ckpt_path": ckpt_path,
        "arch_type": arch_type,
        "method": method,
        "success_count": total_succ,
        "total_configs": len(UNUSED_INDICES),
        "success_rate": pct,
        "succ_vector": succ_vector,
        "elapsed_sec": elapsed,
    }
    return res, "\n".join(log_lines)


def main():
    print("Loading ML1('peg-insert-side-v3', seed=42)...")
    ml1 = metaworld.ML1("peg-insert-side-v3", seed=42)
    tasks = ml1.train_tasks

    print(f"Total task configurations available: {len(tasks)}")
    print(f"Evaluating on {len(UNUSED_INDICES)} unused configurations: {UNUSED_INDICES}\n")

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    all_results = {}
    for name, ckpt, arch, method in MODELS:
        res, full_log = evaluate_model_on_unused_pegs(name, ckpt, arch, method, tasks)
        all_results[name] = res

        # Write log file to logs/
        safe_name = name.lower().replace(" ", "_")
        log_path = f"logs/eval_peg_insert_unused_{safe_name}.log"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(full_log)
        print(f"Saved log to {log_path}\n")

    # Paired comparisons and hypothesis testing
    # 1. MLP: Force MLP vs Kin MLP
    mlp_kin_succ = all_results["Kin MLP"]["success_count"]
    mlp_force_succ = all_results["Force MLP"]["success_count"]
    mlp_delta_pp = all_results["Force MLP"]["success_rate"] - all_results["Kin MLP"]["success_rate"]

    # 2. GRU: Paired across 3 seeds
    gru_kin_succs = [all_results[f"Kin GRU {s}"]["success_count"] for s in ["s42", "s43", "s44"]]
    gru_force_succs = [all_results[f"Force GRU {s}"]["success_count"] for s in ["s42", "s43", "s44"]]

    mean_kin_gru = np.mean(gru_kin_succs)
    mean_force_gru = np.mean(gru_force_succs)
    gru_delta_pp = ((mean_force_gru - mean_kin_gru) / 30.0) * 100.0

    print("=" * 80)
    print("PRE-REGISTERED HYPOTHESIS TEST: PEG-INSERT-SIDE-V3 UNUSED CONFIGURATIONS")
    print("=" * 80)
    print(f"1. FEEDFORWARD (MLP):")
    print(f"   Kin MLP:   {mlp_kin_succ}/30 ({all_results['Kin MLP']['success_rate']:.1f}%)")
    print(f"   Force MLP: {mlp_force_succ}/30 ({all_results['Force MLP']['success_rate']:.1f}%)")
    print(f"   Delta:     {mlp_delta_pp:+.1f} percentage points (Prediction >= +10.0 pp: {mlp_delta_pp >= 10.0})")

    print(f"\n2. RECURRENT (GRU under method=window):")
    for s in ["s42", "s43", "s44"]:
        k_res = all_results[f"Kin GRU {s}"]
        f_res = all_results[f"Force GRU {s}"]
        d_pp = f_res["success_rate"] - k_res["success_rate"]
        print(f"   Seed {s}: Kin GRU {k_res['success_count']:2d}/30 ({k_res['success_rate']:5.1f}%) | "
              f"Force GRU {f_res['success_count']:2d}/30 ({f_res['success_rate']:5.1f}%) | Delta: {d_pp:+5.1f} pp")
    print(f"   Mean Kin GRU:   {mean_kin_gru:.2f}/30 ({mean_kin_gru/0.3:.1f}%)")
    print(f"   Mean Force GRU: {mean_force_gru:.2f}/30 ({mean_force_gru/0.3:.1f}%)")
    print(f"   Mean Delta:     {gru_delta_pp:+.1f} percentage points (Prediction >= +10.0 pp: {gru_delta_pp >= 10.0})")
    print("=" * 80)

    # Save to JSON
    json_path = "models/peg_insert_confirmation_results.json"
    summary_data = {
        "unused_indices": UNUSED_INDICES,
        "models": all_results,
        "comparisons": {
            "mlp": {
                "kin_succ": mlp_kin_succ,
                "force_succ": mlp_force_succ,
                "delta_pp": mlp_delta_pp,
                "prediction_held": bool(mlp_delta_pp >= 10.0),
            },
            "gru": {
                "per_seed": {
                    s: {
                        "kin_succ": all_results[f"Kin GRU {s}"]["success_count"],
                        "force_succ": all_results[f"Force GRU {s}"]["success_count"],
                        "delta_pp": all_results[f"Force GRU {s}"]["success_rate"] - all_results[f"Kin GRU {s}"]["success_rate"],
                    } for s in ["s42", "s43", "s44"]
                },
                "mean_kin_succ": mean_kin_gru,
                "mean_force_succ": mean_force_gru,
                "mean_delta_pp": gru_delta_pp,
                "prediction_held": bool(gru_delta_pp >= 10.0),
            }
        }
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
