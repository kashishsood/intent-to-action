#!/usr/bin/env python3
"""
eval_gru_seeds_12poses.py
Evaluates a GRU policy checkpoint on the 12 locked pick-place-v3 poses with 5 seeds each (60 rollouts).
Supports both evaluation methods:
  - 'step': standard closed-loop step() with recurrent hidden state carryover
  - 'window': sliding window of length 8 with hidden state reset to 0 (matching training)
"""

import argparse
import sys
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning, module="metaworld")
import metaworld

from eval_12tasks_paired_pickplace import load_model_and_config

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def evaluate_gru_checkpoint(
    ckpt_path: str,
    method: str = "step",
    selected_indices: List[int] = [0, 4, 7, 10, 15, 20, 24, 31, 35, 40, 44, 47],
    seeds: List[int] = [1000, 2026, 3000, 4000, 5000],
    device: torch.device = torch.device("cpu"),
):
    info = load_model_and_config(ckpt_path, device)
    model = info["model"]
    obs_mean = info["obs_mean"]
    obs_std = info["obs_std"]
    obs_dim = info["obs_dim"]

    mt10 = metaworld.MT10(seed=42)
    env_cls = mt10.train_classes["pick-place-v3"]
    all_tasks = [t for t in mt10.train_tasks if t.env_name == "pick-place-v3"]

    print(f"\nEvaluating '{ckpt_path}' using method='{method}' on {len(selected_indices)} poses x {len(seeds)} seeds...")
    t0 = time.time()
    results = {}
    ep_count = 0
    total_trials = len(selected_indices) * len(seeds)

    for pose_idx in selected_indices:
        t_obj = all_tasks[pose_idx]
        p_succ = 0
        for r_seed in seeds:
            ep_count += 1
            env = env_cls()
            env.set_task(t_obj)
            torch.manual_seed(r_seed)
            np.random.seed(r_seed)
            obs, _ = env.reset(seed=r_seed)
            
            model.reset_hidden(device)
            ep_succ = False
            
            norm_obs = (obs - obs_mean) / obs_std
            hist = [norm_obs]
            pad = [norm_obs] * 7  # 7 pads for H=8
            
            for step in range(500):
                norm_feat = (obs - obs_mean) / obs_std
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

                obs, reward, term, trunc, info_step = env.step(action)
                hist.append((obs - obs_mean) / obs_std)
                
                if float(info_step.get("success", 0.0)) > 0.5:
                    ep_succ = True
                if term or trunc:
                    break
                    
            if ep_succ:
                p_succ += 1
            print(f"[EP_LOG] Ckpt: {ckpt_path:<32} | Method: {method:<6} | Ep {ep_count:02d}/{total_trials} | Pose #{pose_idx:02d} | Seed {r_seed:4d} | Success: {ep_succ!s:<5} | Steps: {step+1:3d}", flush=True)

        results[pose_idx] = p_succ

    total_succ = sum(results.values())
    rate = (total_succ / total_trials) * 100.0
    elapsed = time.time() - t0
    print(f"\n[SUMMARY] {ckpt_path} (method={method}): {total_succ}/{total_trials} ({rate:.1f}%) in {elapsed:.1f}s")
    print(f"Pose breakdown: {results}\n")
    return results, total_succ, rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--method", type=str, default="step", choices=["step", "window"])
    args = parser.parse_args()
    
    evaluate_gru_checkpoint(args.ckpt, method=args.method)
