#!/usr/bin/env python3
"""
eval_12tasks_paired_pickplace.py
---------------------------------
Phase 4: Primary Hypothesis Test on pick-place-v3 Locked 12-Pose Paired Benchmark.

Directly evaluates 4 models on the exact locked 12 poses from Phase 2 / Phase 3:
  1. Original 5-task Baseline MLP (models/best_bc_model.pt)
  2. 12-task Kinematic MLP (models/bc_12tasks_mlp_kinematic.pt)
  3. 12-task Kinematic GRU (models/bc_12tasks_gru_kinematic.pt)
  4. 12-task Kinematic + Contact Force GRU (models/bc_12tasks_gru_force.pt)

Using metaworld.MT10(seed=42) on task indices [0, 4, 7, 10, 15, 20, 24, 31, 35, 40, 44, 47]
with 5 fixed random seeds [1000, 2026, 3000, 4000, 5000] per pose (60 rollouts per model).
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

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_gripper_geom_ids(model):
    body_names = [model.body(i).name for i in range(model.nbody)]
    target_bodies = ["hand", "rightclaw", "rightpad", "leftclaw", "leftpad"]
    target_bids = {body_names.index(n) for n in target_bodies if n in body_names}
    
    gripper_geoms = set()
    for g_idx in range(model.ngeom):
        if model.geom_bodyid[g_idx] in target_bids:
            gripper_geoms.add(g_idx)
    return gripper_geoms


def extract_genuine_contact_wrench(model, data, gripper_geoms):
    net_force = np.zeros(3, dtype=np.float64)
    net_torque = np.zeros(3, dtype=np.float64)
    
    hand_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    hand_pos = data.xpos[hand_bid] if hand_bid != -1 else np.zeros(3)

    c_force_6d = np.zeros(6, dtype=np.float64)

    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = con.geom1, con.geom2
        
        is_g1 = g1 in gripper_geoms
        is_g2 = g2 in gripper_geoms
        
        if not (is_g1 or is_g2):
            continue
            
        mujoco.mj_contactForce(model, data, i, c_force_6d)
        R = con.frame.reshape(3, 3)
        f_contact_world = R.T @ c_force_6d[0:3]
        
        if is_g1:
            f_on_gripper = -f_contact_world
        else:
            f_on_gripper = f_contact_world
            
        net_force += f_on_gripper
        r = con.pos - hand_pos
        net_torque += np.cross(r, f_on_gripper)
        
    return np.concatenate([net_force, net_torque]).astype(np.float32)


def load_model_and_config(ckpt_path: str, device: torch.device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: '{ckpt_path}'")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    arch_type = ckpt.get("arch_type", "mlp")
    obs_dim = ckpt.get("obs_dim", 39)
    act_dim = ckpt.get("act_dim", 4)
    hidden_dim = ckpt.get("hidden_dim", 256)
    
    if arch_type == "mlp":
        model = BCPolicyMLP(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim).to(device)
    elif arch_type == "gru":
        history_len = ckpt.get("history_len", 8)
        model = BCPolicyGRU(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim).to(device)
    else:
        raise ValueError(f"Unknown arch_type: {arch_type}")
        
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    return {
        "model": model,
        "arch_type": arch_type,
        "obs_dim": obs_dim,
        "obs_mean": ckpt["obs_mean"],
        "obs_std": ckpt["obs_std"],
    }


def evaluate_policy_on_locked_poses(
    model_info: dict,
    env_cls,
    all_tasks,
    selected_indices: List[int],
    seeds: List[int],
    device: torch.device,
    model_name: str,
) -> Tuple[Dict[int, dict], float]:
    model = model_info["model"]
    arch_type = model_info["arch_type"]
    obs_dim = model_info["obs_dim"]
    obs_mean = model_info["obs_mean"]
    obs_std = model_info["obs_std"]
    is_force_model = (obs_dim == 45)

    print(f"\nEvaluating '{model_name}' (arch: {arch_type}, obs_dim: {obs_dim}) on 12 locked poses...")
    t0 = time.time()
    results = {}
    from tqdm import tqdm
    pbar = tqdm(total=len(selected_indices) * len(seeds), desc=f"  {model_name}", unit="ep")
    ep_count = 0

    for pose_idx in selected_indices:
        task_obj = all_tasks[pose_idx]
        succ_count = 0

        for r_seed in seeds:
            ep_count += 1
            env = env_cls()
            env.set_task(task_obj)
            torch.manual_seed(r_seed)
            np.random.seed(r_seed)
            obs, _ = env.reset(seed=r_seed)

            if arch_type == "gru":
                model.reset_hidden(device)

            model_internal = env.unwrapped.model
            data_internal = env.unwrapped.data
            gripper_geoms = get_gripper_geom_ids(model_internal) if is_force_model else None

            ep_succ = False
            for step in range(500):
                if is_force_model:
                    wrench = extract_genuine_contact_wrench(model_internal, data_internal, gripper_geoms)
                    raw_feature = np.concatenate([obs, wrench])
                else:
                    raw_feature = obs

                norm_feature = (raw_feature - obs_mean) / obs_std
                
                with torch.no_grad():
                    if arch_type == "mlp":
                        feat_t = torch.from_numpy(norm_feature).float().unsqueeze(0).to(device)
                        action = model(feat_t).squeeze(0).cpu().numpy()
                    elif arch_type == "gru":
                        feat_t = torch.from_numpy(norm_feature).float().unsqueeze(0).to(device)
                        action = model.step(feat_t).cpu().numpy()

                obs, reward, terminated, truncated, info = env.step(action)
                if float(info.get("success", 0.0)) > 0.5:
                    ep_succ = True
                if terminated or truncated:
                    break

            if ep_succ:
                succ_count += 1

            pbar.update(1)
            print(f"[EP_LOG] Model: {model_name:<22} | Ep {ep_count:02d}/60 | Pose #{pose_idx:02d} | Seed {r_seed:4d} | Success: {ep_succ!s:<5} | Steps: {step+1:3d}", flush=True)

        succ_rate = (succ_count / len(seeds)) * 100.0
        results[pose_idx] = {
            "successes": succ_count,
            "repeats": len(seeds),
            "succ_rate": succ_rate,
        }

    pbar.close()
    total_succ = sum(r["successes"] for r in results.values())
    total_trials = len(selected_indices) * len(seeds)
    overall_rate = (total_succ / total_trials) * 100.0
    elapsed = time.time() - t0
    print(f"  [DONE] {model_name} Overall: {total_succ}/{total_trials} ({overall_rate:.1f}%) in {elapsed:.1f}s\n", flush=True)
    return results, overall_rate


def main():
    print("=" * 100)
    print(" PHASE 4: PAIRED EVALUATION ON LOCKED 12-POSE PICK-PLACE BENCHMARK")
    print("=" * 100)

    device = torch.device("cpu")
    mt10 = metaworld.MT10(seed=42)
    env_cls = mt10.train_classes["pick-place-v3"]
    all_tasks = [t for t in mt10.train_tasks if t.env_name == "pick-place-v3"]

    selected_indices = [0, 4, 7, 10, 15, 20, 24, 31, 35, 40, 44, 47]
    seeds = [1000, 2026, 3000, 4000, 5000]

    model_configs = [
        ("Baseline (5-task MLP)", "models/best_bc_model.pt"),
        ("12-Task Kinematic MLP", "models/bc_12tasks_mlp_kinematic.pt"),
        ("12-Task Kinematic GRU", "models/bc_12tasks_gru_kinematic.pt"),
        ("12-Task Force GRU", "models/bc_12tasks_gru_force.pt"),
    ]

    all_eval_results = {}
    overall_rates = {}

    for name, path in model_configs:
        info = load_model_and_config(path, device)
        res, rate = evaluate_policy_on_locked_poses(
            info, env_cls, all_tasks, selected_indices, seeds, device, name
        )
        all_eval_results[name] = res
        overall_rates[name] = rate

    # Print Pose Metadata and Comparison Table
    env_probe = env_cls()
    table_rows = []
    
    # Track metrics for hypothesis test
    easy_poses = [4, 10, 15, 24, 35, 44]
    hard_poses = [0, 7, 20, 31, 40, 47]

    print("\n" + "=" * 120)
    print(" PAIRED 12-POSE BENCHMARK: FORCE-AUGMENTATION HYPOTHESIS TEST RESULTS")
    print("=" * 120)

    for idx in selected_indices:
        env_probe.set_task(all_tasks[idx])
        probe_obs, _ = env_probe.reset(seed=2026)
        obj_xyz = f"[{probe_obs[3]:.2f},{probe_obs[4]:.2f},{probe_obs[5]:.2f}]"
        goal_xyz = f"[{probe_obs[36]:.2f},{probe_obs[37]:.2f},{probe_obs[38]:.2f}]"
        regime = "EASY" if idx in easy_poses else "HARD"

        r_base = all_eval_results["Baseline (5-task MLP)"][idx]["succ_rate"]
        r_mlp12 = all_eval_results["12-Task Kinematic MLP"][idx]["succ_rate"]
        r_gru_kin = all_eval_results["12-Task Kinematic GRU"][idx]["succ_rate"]
        r_gru_force = all_eval_results["12-Task Force GRU"][idx]["succ_rate"]

        # Did Force GRU flip from hard (0%) to success?
        if idx in hard_poses:
            if r_gru_force > 0.0:
                verdict = f"[FLIPPED] Hard -> Success ({r_gru_force:.0f}%)"
            else:
                verdict = "Still Hard (0%)"
        else:
            if r_gru_force == 100.0:
                verdict = "Maintained Easy (100%)"
            else:
                verdict = f"[DEGRADED] Lost Easy ({r_gru_force:.0f}%)"

        table_rows.append([
            f"#{idx:02d} ({regime})",
            obj_xyz,
            goal_xyz,
            f"{r_base:.0f}%",
            f"{r_mlp12:.0f}%",
            f"{r_gru_kin:.0f}%",
            f"{r_gru_force:.0f}%",
            verdict
        ])

    headers = [
        "Pose #",
        "Obj [X,Y,Z]",
        "Goal [X,Y,Z]",
        "Base 5-task",
        "12T Kin MLP",
        "12T Kin GRU",
        "12T Force GRU",
        "Force vs Hypothesis"
    ]
    print(tabulate(table_rows, headers=headers, tablefmt="github"))

    print("\n" + "=" * 80)
    print(" OVERALL SUCCESS RATES ACROSS 60 ROLLOUTS")
    print("=" * 80)
    for name in all_eval_results:
        print(f"  {name:<30}: {overall_rates[name]:.1f}%")


if __name__ == "__main__":
    main()
