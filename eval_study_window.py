#!/usr/bin/env python3
"""
eval_study_window.py
---------------------
Unified evaluation script for GRU models (Kinematic & Force) using method='window'.
Evaluates on:
1. 12-pose paired pick-place benchmark (12 poses x 5 seeds = 60 rollouts)
2. 12-task MT50(seed=42) benchmark (12 tasks x 20 episodes = 240 rollouts)

Under method='window':
At step t:
  - sliding window of length 8 of normalized features [o_{t-7}, ..., o_t]
  - model(win_t, None) where initial hidden state is 0 (matching training)
  - no recurrent state carried across environment steps.
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

TASK_NAMES = [
    "pick-place-v3",
    "pick-place-wall-v3",
    "peg-insert-side-v3",
    "assembly-v3",
    "hammer-v3",
    "sweep-into-v3",
    "reach-v3",
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

POSE_INDICES = [0, 4, 7, 10, 15, 20, 24, 31, 35, 40, 44, 47]
POSE_SEEDS = [1000, 2026, 3000, 4000, 5000]


def run_12poses_eval(model_info: dict, device: torch.device = torch.device("cpu"), method: str = "window"):
    model = model_info["model"]
    obs_dim = model_info["obs_dim"]
    obs_mean = model_info["obs_mean"]
    obs_std = model_info["obs_std"]
    is_force_model = (obs_dim == 45)
    model_name = model_info.get("model_name", "model")

    mt10 = metaworld.MT10(seed=42)
    env_cls = mt10.train_classes["pick-place-v3"]
    all_tasks = [t for t in mt10.train_tasks if t.env_name == "pick-place-v3"]

    print("=" * 80)
    print(f"12-POSE PAIRED BENCHMARK (pick-place-v3): {model_name} (method={method})")
    print("=" * 80)
    t0 = time.time()
    pose_results = {}
    ep_count = 0
    total_trials = len(POSE_INDICES) * len(POSE_SEEDS)

    for pose_idx in POSE_INDICES:
        task_obj = all_tasks[pose_idx]
        succ_for_pose = 0

        for r_seed in POSE_SEEDS:
            ep_count += 1
            env = env_cls()
            env.set_task(task_obj)
            torch.manual_seed(r_seed)
            np.random.seed(r_seed)
            obs, _ = env.reset(seed=r_seed)

            model_internal = env.unwrapped.model
            data_internal = env.unwrapped.data
            gripper_geoms = get_gripper_geom_ids(model_internal) if is_force_model else None

            if method == "step":
                model.reset_hidden(device)

            # Initialize history buffer for window method
            if is_force_model:
                wrench0 = extract_genuine_contact_wrench(model_internal, data_internal, gripper_geoms)
                feat0 = np.concatenate([obs, wrench0])
            else:
                feat0 = obs
            norm_feat0 = (feat0 - obs_mean) / obs_std
            hist = [norm_feat0]
            pad = [norm_feat0] * 7  # 7 pads for history length 8

            ep_succ = False
            for step in range(500):
                if is_force_model:
                    wrench = extract_genuine_contact_wrench(model_internal, data_internal, gripper_geoms)
                    raw_feat = np.concatenate([obs, wrench])
                else:
                    raw_feat = obs

                norm_feat = (raw_feat - obs_mean) / obs_std

                with torch.no_grad():
                    if method == "step":
                        feat_t = torch.from_numpy(norm_feat).float().unsqueeze(0).to(device)
                        action = model.step(feat_t).cpu().numpy()
                    elif method == "window":
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

            if ep_succ:
                succ_for_pose += 1

            print(
                f"[POSE_EP] Ep {ep_count:02d}/{total_trials} | Pose #{pose_idx:02d} | Seed {r_seed:4d} | Success: {ep_succ!s:<5} | Steps: {step+1:3d}",
                flush=True,
            )

        pose_results[pose_idx] = succ_for_pose

    elapsed = time.time() - t0
    total_succ_eps = sum(pose_results.values())
    overall_pct = (total_succ_eps / total_trials) * 100.0

    # A pose is solved if 5/5 rollouts succeed (exact consistency)
    solved_poses_5of5 = [p for p, s in pose_results.items() if s == 5]
    solved_poses_any = [p for p, s in pose_results.items() if s > 0]

    # Binary vector across 12 poses (1 if solved 5/5, 0 otherwise)
    vector_5of5 = [1 if pose_results[p] == 5 else 0 for p in POSE_INDICES]
    # Raw counts per pose
    counts_vector = [pose_results[p] for p in POSE_INDICES]

    print("\n" + "-" * 80)
    print(f"12-POSE BENCHMARK SUMMARY: {model_name} (method={method})")
    print("-" * 80)
    print(f"Elapsed Time: {elapsed:.1f}s")
    print(f"Rollouts Succeeded: {total_succ_eps}/{total_trials} ({overall_pct:.1f}%)")
    print(f"Poses Solved (5/5 seeds): {len(solved_poses_5of5)}/12 (Indices: {solved_poses_5of5})")
    print(f"Poses Solved (>=1 seed):  {len(solved_poses_any)}/12 (Indices: {solved_poses_any})")
    print(f"Pose Index Order: {POSE_INDICES}")
    print(f"Per-Pose Success Counts (out of 5): {counts_vector}")
    print(f"Per-Pose Binary Solved (5/5):       {vector_5of5}")
    print("-" * 80 + "\n")

    return {
        "model_name": model_name,
        "method": method,
        "pose_results": pose_results,
        "total_succ_eps": total_succ_eps,
        "total_trials": total_trials,
        "pct_succ_eps": overall_pct,
        "solved_5of5_count": len(solved_poses_5of5),
        "solved_5of5_poses": solved_poses_5of5,
        "vector_5of5": vector_5of5,
        "counts_vector": counts_vector,
    }


def run_12tasks_eval(
    model_info: dict,
    episodes_per_task: int = 20,
    seed: int = 42,
    max_steps: int = 500,
    device: torch.device = torch.device("cpu"),
    method: str = "window",
):
    model = model_info["model"]
    obs_dim = model_info["obs_dim"]
    obs_mean = model_info["obs_mean"]
    obs_std = model_info["obs_std"]
    is_force_model = (obs_dim == 45)
    model_name = model_info.get("model_name", "model")

    mt50 = metaworld.MT50(seed=seed)

    print("=" * 80)
    print(f"12-TASK BENCHMARK MT50(seed={seed}): {model_name} (method={method}, {episodes_per_task} eps/task)")
    print("=" * 80)
    t0 = time.time()
    task_results = {}
    total_episodes = len(TASK_NAMES) * episodes_per_task
    global_ep = 0

    for task_name in TASK_NAMES:
        env_cls = mt50.train_classes[task_name]
        tasks = [t for t in mt50.train_tasks if t.env_name == task_name]
        succ_count = 0
        ep_vectors = []

        for ep_i in range(episodes_per_task):
            global_ep += 1
            task_obj = tasks[ep_i % len(tasks)]
            env = env_cls()
            env.set_task(task_obj)
            ep_seed = seed + ep_i * 17
            torch.manual_seed(ep_seed)
            np.random.seed(ep_seed)
            obs, _ = env.reset(seed=ep_seed)

            model_internal = env.unwrapped.model
            data_internal = env.unwrapped.data
            gripper_geoms = get_gripper_geom_ids(model_internal) if is_force_model else None

            if method == "step":
                model.reset_hidden(device)

            if is_force_model:
                wrench0 = extract_genuine_contact_wrench(model_internal, data_internal, gripper_geoms)
                feat0 = np.concatenate([obs, wrench0])
            else:
                feat0 = obs
            norm_feat0 = (feat0 - obs_mean) / obs_std
            hist = [norm_feat0]
            pad = [norm_feat0] * 7

            ep_succ = False
            for step in range(max_steps):
                if is_force_model:
                    wrench = extract_genuine_contact_wrench(model_internal, data_internal, gripper_geoms)
                    raw_feat = np.concatenate([obs, wrench])
                else:
                    raw_feat = obs

                norm_feat = (raw_feat - obs_mean) / obs_std

                with torch.no_grad():
                    if method == "step":
                        feat_t = torch.from_numpy(norm_feat).float().unsqueeze(0).to(device)
                        action = model.step(feat_t).cpu().numpy()
                    elif method == "window":
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

            if ep_succ:
                succ_count += 1
            ep_vectors.append(1 if ep_succ else 0)

            print(
                f"[TASK_EP] Ep {global_ep:03d}/{total_episodes} | Task: {task_name:<24} | EpSeed {ep_seed:5d} | Success: {ep_succ!s:<5} | Steps: {step+1:3d}",
                flush=True,
            )

        task_results[task_name] = {
            "success_count": succ_count,
            "episodes": episodes_per_task,
            "success_rate": (succ_count / episodes_per_task) * 100.0,
            "ep_vector": ep_vectors,
            "regime": TASK_REGIMES[task_name],
        }

    elapsed = time.time() - t0
    total_succ = sum(r["success_count"] for r in task_results.values())
    total_rate = (total_succ / total_episodes) * 100.0

    print("\n" + "=" * 80)
    print(f"12-TASK BENCHMARK SUMMARY: {model_name} (method={method})")
    print("=" * 80)
    table_rows = []
    for t_name in TASK_NAMES:
        r = task_results[t_name]
        table_rows.append([t_name, r["regime"], f"{r['success_count']}/{r['episodes']}", f"{r['success_rate']:.1f}%"])
    table_rows.append(["TOTAL", "--", f"{total_succ}/{total_episodes}", f"{total_rate:.1f}%"])

    print(tabulate(table_rows, headers=["Task", "Regime", "Success", "Rate (%)"], tablefmt="grid"))
    print(f"Elapsed Time: {elapsed:.1f}s\n")

    return {
        "model_name": model_name,
        "method": method,
        "task_results": task_results,
        "total_succ": total_succ,
        "total_episodes": total_episodes,
        "total_rate": total_rate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--benchmark", type=str, default="both", choices=["poses", "12tasks", "both"])
    parser.add_argument("--method", type=str, default="window", choices=["window", "step"])
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--out-json", type=str, default="", help="Path to json file to record results")
    args = parser.parse_args()

    info = load_model_and_config(args.ckpt, torch.device("cpu"))
    if args.model_name:
        info["model_name"] = args.model_name
    else:
        info["model_name"] = os.path.basename(args.ckpt).replace(".pt", "")

    saved_data = {}
    if args.out_json and os.path.exists(args.out_json):
        try:
            with open(args.out_json, "r") as f:
                saved_data = json.load(f)
        except Exception:
            saved_data = {}

    m_key = f"{info['model_name']}_{args.method}"
    if m_key not in saved_data:
        saved_data[m_key] = {}

    if args.benchmark in ["poses", "both"]:
        poses_res = run_12poses_eval(info, method=args.method)
        saved_data[m_key]["12poses"] = poses_res

    if args.benchmark in ["12tasks", "both"]:
        tasks_res = run_12tasks_eval(info, method=args.method)
        saved_data[m_key]["12tasks"] = tasks_res

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(saved_data, f, indent=2)
        print(f"Results saved to {args.out_json}")


if __name__ == "__main__":
    main()
