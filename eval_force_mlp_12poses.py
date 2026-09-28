#!/usr/bin/env python3
"""
eval_force_mlp_12poses.py
--------------------------
Evaluates the 12-Task Force MLP (models/bc_12tasks_mlp_force.pt, 45-D) on the
locked 12-pose MT10(seed=42) paired benchmark:
12 poses x 5 seeds [1000, 2026, 3000, 4000, 5000] = 60 rollouts.

Compares against:
- Baseline (5-task MLP)
- 12-Task Kinematic MLP
- 12-Task Kinematic GRU
- 12-Task Force GRU
"""

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

from train_12tasks_study import BCPolicyMLP, BCPolicyGRU
from eval_12tasks_paired_pickplace import (
    get_gripper_geom_ids,
    extract_genuine_contact_wrench,
    load_model_and_config,
    evaluate_policy_on_locked_poses,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    print("=" * 100)
    print(" EVALUATING 12-TASK FORCE MLP ON LOCKED 12-POSE MT10(seed=42) BENCHMARK")
    print("=" * 100)

    device = torch.device("cpu")
    mt10 = metaworld.MT10(seed=42)
    env_cls = mt10.train_classes["pick-place-v3"]
    all_tasks = [t for t in mt10.train_tasks if t.env_name == "pick-place-v3"]

    selected_indices = [0, 4, 7, 10, 15, 20, 24, 31, 35, 40, 44, 47]
    seeds = [1000, 2026, 3000, 4000, 5000]

    # Evaluate Force MLP
    f_mlp_info = load_model_and_config("models/bc_12tasks_mlp_force.pt", device)
    f_mlp_results, f_mlp_rate = evaluate_policy_on_locked_poses(
        f_mlp_info, env_cls, all_tasks, selected_indices, seeds, device, "12-Task Force MLP"
    )

    # Reference data for other models on the exact same benchmark
    # We already know these bitwise deterministic results from previous runs:
    # Baseline 5-task MLP: 30/60 (50.0%)
    # 12-Task Kin MLP: 15/60 (25.0%)
    # 12-Task Kin GRU: 0/60 (0.0%)
    # 12-Task Force GRU: 30/60 (50.0%)
    
    # Let's load the exact pose success rates for the other models
    ref_eval = {
        "Baseline 5T MLP": {
            0: 0.0, 4: 100.0, 7: 0.0, 10: 100.0, 15: 100.0, 20: 0.0,
            24: 100.0, 31: 0.0, 35: 100.0, 40: 0.0, 44: 100.0, 47: 0.0
        },
        "12T Kin MLP": {
            0: 100.0, 4: 0.0, 7: 0.0, 10: 0.0, 15: 100.0, 20: 0.0,
            24: 0.0, 31: 100.0, 35: 0.0, 40: 0.0, 44: 0.0, 47: 0.0
        },
        "12T Kin GRU": {
            0: 0.0, 4: 0.0, 7: 0.0, 10: 0.0, 15: 0.0, 20: 0.0,
            24: 0.0, 31: 0.0, 35: 0.0, 40: 0.0, 44: 0.0, 47: 0.0
        },
        "12T Force GRU": {
            0: 100.0, 4: 0.0, 7: 0.0, 10: 0.0, 15: 0.0, 20: 100.0,
            24: 100.0, 31: 0.0, 35: 100.0, 40: 0.0, 44: 100.0, 47: 100.0
        }
    }

    easy_poses = [4, 10, 15, 24, 35, 44]
    hard_poses = [0, 7, 20, 31, 40, 47]

    print("\n" + "=" * 130)
    print(" 12-POSE POSE-BY-POSE COMPLETE BENCHMARK TABLE INCLUDING 12-TASK FORCE MLP")
    print("=" * 130)

    env_probe = env_cls()
    table_rows = []

    f_mlp_hard_flips = 0
    f_mlp_easy_held = 0

    for idx in selected_indices:
        env_probe.set_task(all_tasks[idx])
        probe_obs, _ = env_probe.reset(seed=2026)
        obj_xyz = f"[{probe_obs[3]:.2f},{probe_obs[4]:.2f},{probe_obs[5]:.2f}]"
        goal_xyz = f"[{probe_obs[36]:.2f},{probe_obs[37]:.2f},{probe_obs[38]:.2f}]"
        regime = "EASY" if idx in easy_poses else "HARD"

        r_base = ref_eval["Baseline 5T MLP"][idx]
        r_kmlp = ref_eval["12T Kin MLP"][idx]
        r_kgru = ref_eval["12T Kin GRU"][idx]
        r_fgru = ref_eval["12T Force GRU"][idx]
        r_fmlp = f_mlp_results[idx]["succ_rate"]

        if idx in hard_poses:
            if r_fmlp > 0.0:
                f_mlp_hard_flips += 1
                verdict = f"[FLIPPED] Hard -> {r_fmlp:.0f}%"
            else:
                verdict = "Still Hard (0%)"
        else:
            if r_fmlp == 100.0:
                f_mlp_easy_held += 1
                verdict = "Maintained Easy (100%)"
            else:
                verdict = f"[DEGRADED] Lost Easy ({r_fmlp:.0f}%)"

        table_rows.append([
            f"#{idx:02d} ({regime})",
            obj_xyz,
            goal_xyz,
            f"{r_base:.0f}%",
            f"{r_kmlp:.0f}%",
            f"{r_fmlp:.0f}%",
            f"{r_kgru:.0f}%",
            f"{r_fgru:.0f}%",
            verdict
        ])

    headers = [
        "Pose #",
        "Obj [X,Y,Z]",
        "Goal [X,Y,Z]",
        "Base 5T MLP",
        "12T Kin MLP",
        "12T Force MLP",
        "12T Kin GRU",
        "12T Force GRU",
        "Force MLP Verdict"
    ]
    print(tabulate(table_rows, headers=headers, tablefmt="github"))

    tot_succ = sum(f_mlp_results[idx]["successes"] for idx in selected_indices)
    tot_trials = len(selected_indices) * len(seeds)
    print("\n" + "=" * 90)
    print(" SUMMARY OF 12-TASK FORCE MLP ON 12-POSE BENCHMARK")
    print("=" * 90)
    print(f"  Total Closed-Loop Success : {tot_succ}/{tot_trials} ({tot_succ/tot_trials*100:.1f}%)")
    print(f"  Hard Poses Flipped (>0%)  : {f_mlp_hard_flips} / 6")
    print(f"  Easy Poses Maintained(100): {f_mlp_easy_held} / 6")
    print(f"  Comparison to Kin MLP     : {tot_succ/tot_trials*100:.1f}% vs 25.0% (15/60)")
    print(f"  Comparison to Force GRU   : {tot_succ/tot_trials*100:.1f}% vs 50.0% (30/60)")
    print("=" * 90)


if __name__ == "__main__":
    main()
