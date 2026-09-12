#!/usr/bin/env python3
"""
Closed-Loop Policy Evaluator for Trained Behavior Cloning Model in Meta-World

Deploys the trained BC policy into Meta-World simulation environments across the
selected tasks and evaluates closed-loop rollouts and task completion rates.
"""

import argparse
import os
import sys
import warnings
from typing import List

import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="metaworld")

import metaworld
from train_bc import BCPolicyMLP

# Ensure UTF-8 output on Windows consoles
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


def evaluate_closed_loop(
    model_path: str = "models/best_bc_model.pt",
    episodes_per_task: int = 10,
    max_steps: int = 500,
    device_str: str = "cpu",
    seed: int = 100,
):
    print(f"\n==========================================")
    print(f"CLOSED-LOOP SIMULATION EVALUATION OF BC POLICY")
    print(f"Model Checkpoint: '{model_path}'")
    print(f"==========================================")

    device = torch.device(device_str)
    checkpoint = torch.load(model_path, map_location=device)

    obs_dim = checkpoint.get("obs_dim", 39)
    act_dim = checkpoint.get("act_dim", 4)
    hidden_dim = checkpoint.get("hidden_dim", 256)
    obs_mean = checkpoint["obs_mean"]
    obs_std = checkpoint["obs_std"]

    model = BCPolicyMLP(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    mt10 = metaworld.MT10()
    results = []

    for task_name in TASK_NAMES:
        env_cls = mt10.train_classes[task_name]
        env = env_cls()
        tasks = [t for t in mt10.train_tasks if t.env_name == task_name]

        # Use the latter portion of tasks (test split domain)
        test_tasks = tasks[-episodes_per_task:]

        successes = 0
        total_rewards = []
        episode_lengths = []

        for ep_idx, task in enumerate(tqdm(test_tasks, desc=f"  Eval {task_name}", unit="ep")):
            env.set_task(task)
            obs, info = env.reset(seed=seed + ep_idx)
            ep_reward = 0.0
            ep_success = False

            for step in range(max_steps):
                # Standardize observation with training distribution statistics
                norm_obs = (obs - obs_mean) / obs_std
                obs_tensor = torch.from_numpy(norm_obs).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    action_tensor = model(obs_tensor)
                    action = action_tensor.squeeze(0).cpu().numpy()

                obs, reward, terminated, truncated, step_info = env.step(action)
                ep_reward += float(reward)

                if float(step_info.get("success", 0.0)) > 0.5:
                    ep_success = True

                if terminated or truncated:
                    break

            if ep_success:
                successes += 1
            total_rewards.append(ep_reward)
            episode_lengths.append(step + 1)

        success_rate = (successes / episodes_per_task) * 100.0
        results.append(
            {
                "task_name": task_name,
                "episodes": episodes_per_task,
                "successful": successes,
                "success_rate": f"{success_rate:.1f}%",
                "mean_reward": f"{np.mean(total_rewards):.1f}",
                "mean_steps": f"{np.mean(episode_lengths):.1f}",
            }
        )

    headers = ["Task Name", "Episodes", "Successful", "Success Rate", "Mean Reward", "Mean Steps"]
    table_rows = [
        [r["task_name"], r["episodes"], r["successful"], r["success_rate"], r["mean_reward"], r["mean_steps"]]
        for r in results
    ]

    total_eps = len(results) * episodes_per_task
    total_succ = sum(r["successful"] for r in results)
    overall_rate = (total_succ / total_eps) * 100.0 if total_eps > 0 else 0.0
    table_rows.append(["TOTAL / OVERALL", total_eps, total_succ, f"{overall_rate:.1f}%", "-", "-"])

    print("\n" + "=" * 70)
    print("           CLOSED-LOOP SIMULATION EVALUATION RESULTS")
    print("=" * 70)
    try:
        print(tabulate(table_rows, headers=headers, tablefmt="github"))
    except Exception:
        print(tabulate(table_rows, headers=headers, tablefmt="simple"))
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Closed-loop evaluation of BC model on Meta-World.")
    parser.add_argument("--model-path", type=str, default="models/best_bc_model.pt", help="Path to model checkpoint.")
    parser.add_argument("--episodes-per-task", type=int, default=10, help="Episodes per task for evaluation.")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode.")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device.")
    parser.add_argument("--seed", type=int, default=999, help="Evaluation random seed.")
    args = parser.parse_args()

    evaluate_closed_loop(
        model_path=args.model_path,
        episodes_per_task=args.episodes_per_task,
        max_steps=args.max_steps,
        device_str=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
