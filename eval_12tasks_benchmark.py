#!/usr/bin/env python3
"""
eval_12tasks_benchmark.py
--------------------------
Phase 4: Multi-Task Closed-Loop Benchmark Across All 12 Tasks.

Evaluates 3 policy models across all 12 Meta-World MT50 tasks:
  1. 12-Task Kinematic MLP (models/bc_12tasks_mlp_kinematic.pt)
  2. 12-Task Kinematic GRU (models/bc_12tasks_gru_kinematic.pt)
  3. 12-Task Force GRU     (models/bc_12tasks_gru_force.pt)

Computes per-task success rates, reward, and step lengths.
Flags performance on the 7 'hardstop hold' tasks vs. 4 true contact tasks vs. 1 free-space task.
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

from train_12tasks_study import BCPolicyMLP, BCPolicyGRU
from eval_12tasks_paired_pickplace import get_gripper_geom_ids, extract_genuine_contact_wrench, load_model_and_config

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TASK_NAMES = [
    # Contact-Rich Manipulation (Regime 1: True Contact Dynamics)
    "pick-place-v3",
    "pick-place-wall-v3",
    "peg-insert-side-v3",
    "assembly-v3",
    "hammer-v3",
    "sweep-into-v3",
    # Free-space (Regime 3: Zero Contact)
    "reach-v3",
    # Constrained / Mechanism (Regime 2: Hardstop Holding)
    "door-open-v3",
    "drawer-open-v3",
    "button-press-topdown-v3",
    "drawer-close-v3",
    "door-close-v3",
]

TASK_REGIMES = {
    "pick-place-v3": "True Contact",
    "pick-place-wall-v3": "True Contact",
    "peg-insert-side-v3": "Hardstop Hold",
    "assembly-v3": "True Contact",
    "hammer-v3": "Hardstop Hold",
    "sweep-into-v3": "True Contact",
    "reach-v3": "Free-Space (0N)",
    "door-open-v3": "Hardstop Hold",
    "drawer-open-v3": "Hardstop Hold",
    "button-press-topdown-v3": "Hardstop Hold",
    "drawer-close-v3": "Hardstop Hold",
    "door-close-v3": "Hardstop Hold",
}


def evaluate_model_on_12tasks(
    model_info: dict,
    mt50: metaworld.MT50,
    episodes_per_task: int = 20,
    seed: int = 42,
    max_steps: int = 500,
    device: torch.device = torch.device("cpu"),
    model_name: str = "Model",
) -> Dict[str, dict]:
    model = model_info["model"]
    arch_type = model_info["arch_type"]
    obs_dim = model_info["obs_dim"]
    obs_mean = model_info["obs_mean"]
    obs_std = model_info["obs_std"]
    is_force_model = (obs_dim == 45)

    print(f"\nEvaluating '{model_name}' on 12 tasks ({episodes_per_task} eps/task)...")
    t0 = time.time()
    task_results = {}

    for t_idx, task_name in enumerate(TASK_NAMES):
        env_cls = mt50.train_classes[task_name]
        tasks = [t for t in mt50.train_tasks if t.env_name == task_name]
        succ_count = 0
        rewards = []

        for ep_i in range(episodes_per_task):
            task_obj = tasks[ep_i % len(tasks)]
            env = env_cls()
            env.set_task(task_obj)
            ep_seed = seed + ep_i * 17
            torch.manual_seed(ep_seed)
            np.random.seed(ep_seed)
            obs, _ = env.reset(seed=ep_seed)

            if arch_type == "gru":
                model.reset_hidden(device)

            model_internal = env.unwrapped.model
            data_internal = env.unwrapped.data
            gripper_geoms = get_gripper_geom_ids(model_internal) if is_force_model else None

            ep_succ = False
            ep_rew = 0.0
            for step in range(max_steps):
                if is_force_model:
                    wrench = extract_genuine_contact_wrench(model_internal, data_internal, gripper_geoms)
                    raw_feat = np.concatenate([obs, wrench])
                else:
                    raw_feat = obs

                norm_feat = (raw_feat - obs_mean) / obs_std
                feat_t = torch.from_numpy(norm_feat).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    if arch_type == "mlp":
                        action = model(feat_t).squeeze(0).cpu().numpy()
                    elif arch_type == "gru":
                        action = model.step(feat_t).cpu().numpy()

                obs, reward, terminated, truncated, info = env.step(action)
                ep_rew += float(reward)
                if float(info.get("success", 0.0)) > 0.5:
                    ep_succ = True
                if terminated or truncated:
                    break

            if ep_succ:
                succ_count += 1
            rewards.append(ep_rew)

        succ_rate = (succ_count / episodes_per_task) * 100.0
        mean_rew = float(np.mean(rewards))
        task_results[task_name] = {
            "successes": succ_count,
            "episodes": episodes_per_task,
            "success_rate": succ_rate,
            "mean_reward": mean_rew,
        }
        print(f"  [{t_idx+1:02d}/12] {task_name:<25}: {succ_count}/{episodes_per_task} ({succ_rate:.1f}%) | Mean Reward: {mean_rew:.1f}")

    total_succ = sum(r["successes"] for r in task_results.values())
    total_eps = len(TASK_NAMES) * episodes_per_task
    overall_rate = (total_succ / total_eps) * 100.0
    print(f">> {model_name} Overall Benchmark: {total_succ}/{total_eps} ({overall_rate:.1f}%) in {time.time() - t0:.1f}s")
    return task_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate 12-task benchmark.")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per task (default: 20).")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation random seed.")
    args = parser.parse_args()

    print("=" * 100)
    print(f" PHASE 4: MULTI-TASK CLOSED-LOOP BENCHMARK ({len(TASK_NAMES)} TASKS x {args.episodes} EPS)")
    print("=" * 100)

    device = torch.device("cpu")
    mt50 = metaworld.MT50()

    models = [
        ("12-Task Kinematic MLP", "models/bc_12tasks_mlp_kinematic.pt"),
        ("12-Task Kinematic GRU", "models/bc_12tasks_gru_kinematic.pt"),
        ("12-Task Force GRU",     "models/bc_12tasks_gru_force.pt"),
    ]

    all_results = {}
    for name, path in models:
        info = load_model_and_config(path, device)
        res = evaluate_model_on_12tasks(
            info, mt50, episodes_per_task=args.episodes, seed=args.seed, device=device, model_name=name
        )
        all_results[name] = res

    # Summary table
    table_rows = []
    for t_name in TASK_NAMES:
        regime = TASK_REGIMES[t_name]
        r_mlp = all_results["12-Task Kinematic MLP"][t_name]["success_rate"]
        r_gru_kin = all_results["12-Task Kinematic GRU"][t_name]["success_rate"]
        r_gru_force = all_results["12-Task Force GRU"][t_name]["success_rate"]
        delta_force = r_gru_force - r_gru_kin

        table_rows.append([
            t_name,
            regime,
            f"{r_mlp:.1f}%",
            f"{r_gru_kin:.1f}%",
            f"{r_gru_force:.1f}%",
            f"{delta_force:+.1f}%",
        ])

    headers = [
        "Task Name",
        "Regime Classification",
        "12T Kin MLP",
        "12T Kin GRU",
        "12T Force GRU",
        "Force Delta (GRU)"
    ]
    print("\n" + "=" * 120)
    print(" MULTI-TASK BENCHMARK: PER-TASK SUCCESS RATES")
    print("=" * 120)
    print(tabulate(table_rows, headers=headers, tablefmt="github"))

    # Overall Summary
    print("\n" + "=" * 80)
    print(" AGGREGATE SUMMARY BY REGIME")
    print("=" * 80)
    for model_name in models:
        m_name = model_name[0]
        # Overall
        tot_succ = sum(all_results[m_name][t]["successes"] for t in TASK_NAMES)
        tot_eps = len(TASK_NAMES) * args.episodes
        # True Contact
        tc_tasks = [t for t in TASK_NAMES if TASK_REGIMES[t] == "True Contact"]
        tc_succ = sum(all_results[m_name][t]["successes"] for t in tc_tasks)
        tc_eps = len(tc_tasks) * args.episodes
        # Hardstop Hold
        hh_tasks = [t for t in TASK_NAMES if TASK_REGIMES[t] == "Hardstop Hold"]
        hh_succ = sum(all_results[m_name][t]["successes"] for t in hh_tasks)
        hh_eps = len(hh_tasks) * args.episodes
        # Free Space
        fs_succ = all_results[m_name]["reach-v3"]["successes"]
        fs_eps = args.episodes

        print(f"\n{m_name}:")
        print(f"  Overall:         {tot_succ}/{tot_eps} ({100.0*tot_succ/tot_eps:.1f}%)")
        print(f"  True Contact (4):{tc_succ}/{tc_eps} ({100.0*tc_succ/tc_eps:.1f}%)")
        print(f"  Hardstop (7):    {hh_succ}/{hh_eps} ({100.0*hh_succ/hh_eps:.1f}%)")
        print(f"  Free-Space (1):  {fs_succ}/{fs_eps} ({100.0*fs_succ/fs_eps:.1f}%)")

    # Save benchmark results to JSON
    benchmark_save_path = "models/benchmark_12tasks_results.json"
    with open(benchmark_save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Saved benchmark results to '{benchmark_save_path}'")


if __name__ == "__main__":
    main()
