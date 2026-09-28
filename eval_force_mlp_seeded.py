#!/usr/bin/env python3
"""
eval_force_mlp_seeded.py
------------------------
Evaluates the newly trained 12-Task Force MLP (models/bc_12tasks_mlp_force.pt, 45-D)
on the exact same seeded benchmark: MT50(seed=42), 20 fixed seeds per task (42 + 17*i).
Directly tests whether Force MLP > Kin MLP on pick-place and pick-place-wall.
"""

import sys
import time
import json
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="metaworld")
import metaworld
from tabulate import tabulate

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


def main():
    print("=" * 100)
    print(" EVALUATING 12-TASK FORCE MLP ON LOCKED BENCHMARK (MT50 seed=42, 20 eps/task)")
    print("=" * 100)

    device = torch.device("cpu")
    mt50 = metaworld.MT50(seed=42)
    seeds = [42 + i * 17 for i in range(20)]

    model_path = "models/bc_12tasks_mlp_force.pt"
    m_info = load_model_and_config(model_path, device)
    model = m_info["model"]
    obs_dim = m_info["obs_dim"]
    obs_mean = m_info["obs_mean"]
    obs_std = m_info["obs_std"]

    print(f"Loaded {model_path}: obs_dim={obs_dim}, arch=mlp\n")

    t0_all = time.time()
    results = {}

    for t_idx, task_name in enumerate(TASK_NAMES):
        env_cls = mt50.train_classes[task_name]
        tasks = [t for t in mt50.train_tasks if t.env_name == task_name]
        succ_count = 0
        rewards = []
        episode_details = []

        t0_task = time.time()
        for ep_i, ep_seed in enumerate(seeds):
            task_obj = tasks[ep_i % len(tasks)]
            env = env_cls()
            env.set_task(task_obj)
            torch.manual_seed(ep_seed)
            np.random.seed(ep_seed)
            obs, _ = env.reset(seed=ep_seed)

            m_int = env.unwrapped.model
            d_int = env.unwrapped.data
            g_geoms = get_gripper_geom_ids(m_int)

            ep_succ = False
            ep_rew = 0.0

            with torch.no_grad():
                for step in range(500):
                    wrench = extract_genuine_contact_wrench(m_int, d_int, g_geoms)
                    raw_feat = np.concatenate([obs, wrench])
                    norm_feat = (raw_feat - obs_mean) / obs_std
                    feat_t = torch.from_numpy(norm_feat).float().unsqueeze(0).to(device)
                    action = model(feat_t).squeeze(0).cpu().numpy()

                    obs, reward, terminated, truncated, info = env.step(action)
                    ep_rew += float(reward)
                    if float(info.get("success", 0.0)) > 0.5:
                        ep_succ = True
                    if terminated or truncated:
                        break

            if ep_succ:
                succ_count += 1
            rewards.append(ep_rew)
            episode_details.append({
                "ep": ep_i,
                "seed": ep_seed,
                "succ": bool(ep_succ),
                "rew": round(float(ep_rew), 2),
            })

        succ_rate = (succ_count / len(seeds)) * 100.0
        mean_rew = float(np.mean(rewards))
        results[task_name] = {
            "successes": succ_count,
            "episodes": len(seeds),
            "success_rate": succ_rate,
            "mean_reward": mean_rew,
            "episode_details": episode_details,
        }
        print(f"  [{t_idx+1:02d}/12] {task_name:<25}: {succ_count:2d}/20 ({succ_rate:5.1f}%) | Mean Rew: {mean_rew:6.1f} ({time.time()-t0_task:.1f}s)")

    tot_succ = sum(r["successes"] for r in results.values())
    tot_eps = len(TASK_NAMES) * 20
    print(f"\n>> 12-Task Force MLP Overall Benchmark: {tot_succ}/{tot_eps} ({tot_succ/tot_eps*100:.1f}%) in {time.time()-t0_all:.1f}s")

    # Save to JSON
    out_path = "models/benchmark_force_mlp_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved Force MLP benchmark results to '{out_path}'")

    # Comparison table against Kinematic MLP
    bench_data = json.load(open("models/benchmark_12tasks_results.json"))
    kin_mlp_data = bench_data["12-Task Kinematic MLP"]

    print("\n" + "=" * 110)
    print(" HEAD-TO-HEAD COMPARISON: 12-TASK KINEMATIC MLP vs 12-TASK FORCE MLP")
    print("=" * 110)

    rows = []
    for t_name in TASK_NAMES:
        k_s = kin_mlp_data[t_name]["successes"]
        f_s = results[t_name]["successes"]
        r_k = k_s / 20 * 100
        r_f = f_s / 20 * 100
        delta = r_f - r_k
        v_k = [int(ep["succ"]) for ep in kin_mlp_data[t_name]["episode_details"]]
        v_f = [int(ep["succ"]) for ep in results[t_name]["episode_details"]]
        diffs = [f - k for f, k in zip(v_f, v_k)]
        var = sum((d - delta/100)**2 for d in diffs) / 19
        se = (var / 20) ** 0.5 * 100
        z = 1.64485
        ci = (delta - z * se, delta + z * se)
        rows.append([
            t_name,
            f"{k_s}/20 ({r_k:.1f}%)",
            f"{f_s}/20 ({r_f:.1f}%)",
            f"{delta:+.1f}%",
            f"[{ci[0]:+.1f}%, {ci[1]:+.1f}%]",
        ])

    headers = ["Task Name", "12T Kin MLP", "12T Force MLP", "Delta (Force - Kin)", "Paired 90% CI"]
    print(tabulate(rows, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()
