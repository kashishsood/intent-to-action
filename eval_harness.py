#!/usr/bin/env python3
"""
Evaluation Harness for Meta-World Imitation Learning Policies (eval_harness.py)

Consolidated evaluation pipeline that:
1. Computes step-wise action MAE and MSE per task on the held-out test split.
2. Runs closed-loop rollouts (e.g. 50 episodes per task) in Meta-World simulation.
3. Automatically generates an evaluation markdown report (eval_report.md) with
   the unified comparison table and an automated "Flagged Gaps" analysis section.
"""

import argparse
import datetime
import os
import sys
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from tabulate import tabulate
import torch
import torch.nn as nn
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


def evaluate_stepwise_test_errors(
    model: nn.Module,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    test_parquet_path: str,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Computes step-wise action MSE and MAE per task on the held-out test set.
    """
    if not os.path.exists(test_parquet_path):
        raise FileNotFoundError(f"Test dataset not found at '{test_parquet_path}'")

    df = pd.read_parquet(test_parquet_path)
    model.eval()

    task_metrics = {}

    for task_name in TASK_NAMES:
        group = df[df["task_name"] == task_name]
        if len(group) == 0:
            continue

        obs_raw = np.array(group["obs"].tolist(), dtype=np.float32)
        act_true = np.array(group["action"].tolist(), dtype=np.float32)

        # Standardize using training partition statistics
        obs_norm = (obs_raw - obs_mean) / obs_std

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_norm).to(device)
            act_pred = model(obs_tensor).cpu().numpy()

        mse = float(np.mean((act_pred - act_true) ** 2))
        mae = float(np.mean(np.abs(act_pred - act_true)))

        task_metrics[task_name] = {
            "step_mae": mae,
            "step_mse": mse,
            "test_steps": len(group),
            "test_episodes": group["global_episode_id"].nunique(),
        }

    return task_metrics


def run_closed_loop_rollouts(
    model: nn.Module,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    episodes_per_task: int,
    max_steps: int,
    device: torch.device,
    seed: int,
    return_trajectories: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Runs closed-loop simulation rollouts in Meta-World across all tasks.
    If return_trajectories is True, also returns a dict mapping each task name to a dict
    containing a list of action trajectories (list of np.ndarray) for each episode.
    """
    mt10 = metaworld.MT10()
    results = {}
    trajectories = {} if return_trajectories else None

    print(f"\nRunning closed-loop simulation rollouts ({episodes_per_task} episodes/task)...")

    for task_id, task_name in enumerate(TASK_NAMES):
        env_cls = mt10.train_classes[task_name]
        tasks = [t for t in mt10.train_tasks if t.env_name == task_name]

        success_count = 0
        rewards = []
        lengths = []
        task_actions = [] if return_trajectories else None

        desc = f"  [{task_id + 1}/{len(TASK_NAMES)}] {task_name}"
        for ep_idx in tqdm(range(episodes_per_task), desc=desc, unit="ep"):
            env = env_cls()  # fresh env for each episode to avoid RNG leakage
            env.set_task(tasks[ep_idx % len(tasks)])
            ep_seed = seed + ep_idx * 17
            torch.manual_seed(ep_seed)
            np.random.seed(ep_seed)
            obs, info = env.reset(seed=ep_seed)
            ep_reward = 0.0
            ep_success = False
            episode_actions = []
            # Log episode information (task, episode index, seed, success) after the episode ends
            # We'll compute success after the rollout loop and print here.

            for step in range(max_steps):
                norm_obs = (obs - obs_mean) / obs_std
                obs_t = torch.from_numpy(norm_obs).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    action = model(obs_t).squeeze(0).cpu().numpy()

                # Record action if needed
                if return_trajectories:
                    episode_actions.append(action.copy())

                obs, reward, terminated, truncated, step_info = env.step(action)
                ep_reward += float(reward)

                if float(step_info.get("success", 0.0)) > 0.5:
                    ep_success = True

                if terminated or truncated:
                    break

            if ep_success:
                success_count += 1
                # Log per‑episode outcome (success)
                print(f"[EP_LOG] Task {task_name} Episode {ep_idx} Seed {seed + ep_idx * 17} Success {ep_success}")
            else:
                # Log per‑episode outcome (failure) – do NOT increment success_count
                print(f"[EP_LOG] Task {task_name} Episode {ep_idx} Seed {seed + ep_idx * 17} Success {ep_success}")
            # (no increment on failure)
            rewards.append(ep_reward)
            lengths.append(step + 1)
            if return_trajectories:
                task_actions.append(np.stack(episode_actions))

        succ_rate = (success_count / episodes_per_task) * 100.0
        results[task_name] = {
            "episodes": episodes_per_task,
            "successes": success_count,
            "success_rate": succ_rate,
            "mean_reward": float(np.mean(rewards)),
            "mean_steps": float(np.mean(lengths)),
        }
        if return_trajectories:
            trajectories[task_name] = {"actions": task_actions}

    if return_trajectories:
        return {"metrics": results, "trajectories": trajectories}
    return results


def detect_flagged_gaps(
    combined_results: List[dict],
    success_threshold: float = 50.0,
) -> List[dict]:
    """
    Identifies tasks where step-wise accuracy is in the best half of tasks (low MAE)
    yet task-success rate falls below the specified threshold.
    """
    maes = [r["step_mae"] for r in combined_results]
    median_mae = float(np.median(maes))

    flagged = []
    for r in combined_results:
        is_best_half = r["step_mae"] <= median_mae
        is_poor_success = r["success_rate"] < success_threshold

        if is_best_half and is_poor_success:
            flagged.append(
                {
                    "task_name": r["task_name"],
                    "step_mae": r["step_mae"],
                    "step_mse": r["step_mse"],
                    "success_rate": r["success_rate"],
                    "median_mae": median_mae,
                    "threshold": success_threshold,
                }
            )

    return flagged


def generate_markdown_report(
    combined_results: List[dict],
    flagged_gaps: List[dict],
    total_summary: dict,
    output_report_path: str,
    model_path: str,
    episodes_per_task: int,
    device_str: str,
    quality_gate_results: List[dict] = None,
    quality_gate_passed: bool = True,
):
    """
    Dynamically formats and writes the comprehensive evaluation markdown report.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Table markdown
    table_lines = [
        "| Task Name | Test Steps | Step-wise MAE | Step-wise MSE | Rollout Episodes | Rollout Successes | Task Success Rate |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in combined_results:
        table_lines.append(
            f"| `{r['task_name']}` | {r['test_steps']:,} | **{r['step_mae']:.4f}** | {r['step_mse']:.5f} | "
            f"{r['episodes']} | {r['successes']} | **{r['success_rate']:.1f}%** |"
        )
    table_lines.append(
        f"| **TOTAL / OVERALL** | **{total_summary['test_steps']:,}** | **{total_summary['step_mae']:.4f}** | "
        f"**{total_summary['step_mse']:.5f}** | **{total_summary['episodes']}** | "
        f"**{total_summary['successes']}** | **{total_summary['success_rate']:.1f}%** |"
    )
    table_md = "\n".join(table_lines)

    # Quality gate markdown
    qg_md = ""
    if quality_gate_results:
        status_banner = "### [PASS] Quality Gate Passed" if quality_gate_passed else "### [FAIL] Quality Gate Failed (Merge Blocked)"
        qg_lines = [
            f"## Quality Gate Verification\n",
            f"{status_banner}\n",
            "| Task Name | Baseline Success | Min Allowed Threshold | Achieved Success | Delta | Gate Status |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for q in quality_gate_results:
            qg_lines.append(
                f"| `{q['task_name']}` | {q['baseline']:.1f}% | {q['min_allowed']:.1f}% | "
                f"**{q['achieved']:.1f}%** | {q['delta']:+.1f}% | **{q['status']}** |"
            )
        qg_md = "\n".join(qg_lines) + "\n\n---\n"

    # Flagged gaps markdown
    if flagged_gaps:
        gaps_sections = []
        for g in flagged_gaps:
            task = g["task_name"]
            gaps_sections.append(
                f"### > [!WARNING]\n"
                f"### Flagged Discrepancy: `{task}`\n\n"
                f"- **Step-wise MAE**: `{g['step_mae']:.4f}` (Ranked in best half of tasks, below median `{g['median_mae']:.4f}`)\n"
                f"- **Step-wise MSE**: `{g['step_mse']:.5f}`\n"
                f"- **Closed-Loop Task Success**: `{g['success_rate']:.1f}%` (Below threshold of `{g['threshold']:.1f}%`)\n\n"
                f"**Root Cause Analysis for `{task}`**:\n"
                f"- The model predicts actions with high numerical precision on state distributions seen in demonstrations.\n"
                f"- However, in closed loop, small uncorrected positional drifts compound before the contact boundary, "
                f"causing the policy to stall near the object without securing physical contact or completing the task.\n"
                f"- Demonstrates the classical **compounding error / covariate shift** phenomenon where low offline loss fails to guarantee high closed-loop task execution."
            )
        gaps_md = "\n\n".join(gaps_sections)
    else:
        gaps_md = "_No tasks breached the accuracy-vs-success gap threshold._"

    report_content = f"""# Behavioral Cloning Evaluation Report

**Generated on**: {timestamp}  
**Model Checkpoint**: `{model_path}`  
**Evaluation Environment**: Meta-World MT10  
**Rollout Configuration**: {episodes_per_task} episodes per task ({total_summary['episodes']} total rollouts)  
**Compute Device**: `{device_str}`  

---

{qg_md}## 1. Unified Benchmark: Step-Wise Accuracy vs. Task Success

This table directly pairs offline imitation accuracy on the held-out test split with active closed-loop rollout success rates in the Meta-World simulation:

{table_md}

---

## 2. Flagged Gaps: Accuracy-to-Success Discrepancies

Tasks where offline prediction accuracy is in the **best half of tasks** (Step-wise MAE $\\le$ median `{np.median([r['step_mae'] for r in combined_results]):.4f}$) yet closed-loop task success is poor ($< {flagged_gaps[0]['threshold'] if flagged_gaps else 50.0:.1f}\\%$):

{gaps_md}

---

## 3. Methodological Implications

1. **Compounding Error vs. Physical Funneling**:
   - Tasks with narrow contact tolerance (e.g. `pick-place-v3`) demand sub-centimeter alignment; small offline errors compound over time, preventing task completion.
   - Constrained kinematics tasks (e.g. `door-open-v3`, `button-press-topdown-v3`) tolerate larger action errors because physical barriers funnel the robot toward the goal state.
2. **Evaluation Principle**:
   - Step-wise MSE/MAE on static demonstration data is insufficient as a standalone model selection metric for robot manipulation; closed-loop rollout validation is indispensable.
"""

    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\n[OK] Successfully written evaluation report to '{output_report_path}'")

    # If running inside GitHub Actions, publish report to GITHUB_STEP_SUMMARY
    gh_step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh_step_summary:
        try:
            with open(gh_step_summary, "a", encoding="utf-8") as f:
                f.write(report_content + "\n")
            print(f"[OK] Published evaluation report to GitHub Actions Step Summary")
        except Exception as e:
            print(f"Warning: Failed to write to GITHUB_STEP_SUMMARY: {e}")



def main():
    parser = argparse.ArgumentParser(description="Consolidated Evaluation Harness for Behavior Cloning.")
    parser.add_argument("--model-path", type=str, default="models/best_bc_model.pt", help="Trained model checkpoint.")
    parser.add_argument("--test-file", type=str, default="dataset/test.parquet", help="Held-out test dataset file.")
    parser.add_argument("--episodes-per-task", type=int, default=50, help="Closed-loop rollout episodes per task.")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per closed-loop episode.")
    parser.add_argument("--output-report", type=str, default="eval_report.md", help="Markdown report output path.")
    parser.add_argument(
        "--gap-success-threshold",
        type=float,
        default=50.0,
        help="Success rate threshold (%%) below which low-MAE tasks are flagged.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda').")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for reproducibility.")
    parser.add_argument("--repro-episodes", type=int, default=None, help="Limit number of episodes compared in reproducibility test (default: all).")
    parser.add_argument("--test-repro", action="store_true", help="Run reproducibility test: execute rollouts twice with same seed and compare trajectories.")
    parser.add_argument("--test-repro-single", action="store_true", help="Run reproducibility test in isolated subprocesses to avoid in-process RNG leakage.")
    parser.add_argument("--task", type=str, default=None, help="Specific task name to evaluate (default: all MT10 tasks).")
    parser.add_argument("--repro-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    global TASK_NAMES
    if args.task:
        TASK_NAMES = [args.task]

    print("\n" + "=" * 80)
    print("      BEHAVIOR CLONING EVALUATION HARNESS (eval_harness.py)")
    print("=" * 80)

    device = torch.device(args.device)
    if not os.path.exists(args.model_path):
        print(f"Error: model checkpoint not found at '{args.model_path}'")
        sys.exit(1)

    checkpoint = torch.load(args.model_path, map_location=device)
    obs_mean = checkpoint["obs_mean"]
    obs_std = checkpoint["obs_std"]

    model = BCPolicyMLP(
        obs_dim=checkpoint.get("obs_dim", 39),
        act_dim=checkpoint.get("act_dim", 4),
        hidden_dim=checkpoint.get("hidden_dim", 256),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Step 1: Step-wise offline errors on test set
    print("\nComputing per-task step-wise test errors...")
    offline_metrics = evaluate_stepwise_test_errors(
        model=model,
        obs_mean=obs_mean,
        obs_std=obs_std,
        test_parquet_path=args.test_file,
        device=device,
    )

    # Step 2: Closed-loop simulation rollouts
    closed_loop_metrics = run_closed_loop_rollouts(
        model=model,
        obs_mean=obs_mean,
        obs_std=obs_std,
        episodes_per_task=args.episodes_per_task,
        max_steps=args.max_steps,
        device=device,
        seed=args.seed,
    )

    # Optional reproducibility test
    if args.test_repro:
        print("\nRunning reproducibility test (two runs with same seed)...")
        # First run
        trajs1 = run_closed_loop_rollouts(
            model=model,
            obs_mean=obs_mean,
            obs_std=obs_std,
            episodes_per_task=args.episodes_per_task,
            max_steps=args.max_steps,
            device=device,
            seed=args.seed,
            return_trajectories=True,
        )
        # Second run
        trajs2 = run_closed_loop_rollouts(
            model=model,
            obs_mean=obs_mean,
            obs_std=obs_std,
            episodes_per_task=args.episodes_per_task,
            max_steps=args.max_steps,
            device=device,
            seed=args.seed,
            return_trajectories=True,
        )
        # Compare per task
        for task_name in TASK_NAMES:
            actions1 = trajs1["trajectories"][task_name]["actions"]
            actions2 = trajs2["trajectories"][task_name]["actions"]
            n_eps = args.repro_episodes if args.repro_episodes is not None else len(actions1)
            for ep_idx in range(n_eps):
                seed_used = args.seed + ep_idx * 17
                # Compare trajectories for this episode
                arr1 = np.array(actions1[ep_idx])
                arr2 = np.array(actions2[ep_idx])
                if np.array_equal(arr1, arr2):
                    result = 'exactly equal'
                    diff_step = 'N/A'
                elif np.allclose(arr1, arr2, atol=1e-6):
                    result = 'numerically close (<=1e-6)'
                    diff_step = 'N/A'
                else:
                    if arr1.shape != arr2.shape:
                        min_len = min(len(arr1), len(arr2))
                        per_step_equal = np.isclose(arr1[:min_len], arr2[:min_len], atol=1e-6).all(axis=1)
                        diff_idxs = np.where(~per_step_equal)[0]
                        diff_step = diff_idxs[0] if diff_idxs.size > 0 else f"length mismatch ({len(arr1)} vs {len(arr2)})"
                    else:
                        per_step_equal = np.isclose(arr1, arr2, atol=1e-6).all(axis=1)
                        diff_idxs = np.where(~per_step_equal)[0]
                        diff_step = diff_idxs[0] if diff_idxs.size > 0 else 'N/A'
                    result = 'different'
                print(f"[REPRO] Task {task_name:<20} Episode {ep_idx+1:2d} Seed {seed_used:5d}: {result}; First divergence step: {diff_step}")


    if args.test_repro_single:
        import subprocess, tempfile
        print("\nRunning reproducibility test in isolated subprocesses (two independent runs)...")
        eps = args.repro_episodes if args.repro_episodes is not None else args.episodes_per_task
        # Temporary files to store trajectories
        tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".npz")
        tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".npz")
        tmp1_path = tmp1.name
        tmp2_path = tmp2.name
        tmp1.close()
        tmp2.close()
        base_cmd = [
            sys.executable, sys.argv[0],
            "--model-path", args.model_path,
            "--test-file", args.test_file,
            "--episodes-per-task", str(eps),
            "--max-steps", str(args.max_steps),
            "--device", args.device,
            "--seed", str(args.seed),
            "--repro-run",
            "--output", ""
        ]
        if args.task:
            base_cmd.extend(["--task", args.task])
        # First subprocess
        cmd1 = base_cmd.copy()
        cmd1[-1] = tmp1_path
        subprocess.run(cmd1, check=True)
        # Second subprocess
        cmd2 = base_cmd.copy()
        cmd2[-1] = tmp2_path
        subprocess.run(cmd2, check=True)
        # Load trajectories saved by subprocesses
        trajs1 = np.load(tmp1_path, allow_pickle=True)['trajectories'].item()
        trajs2 = np.load(tmp2_path, allow_pickle=True)['trajectories'].item()
        for task_name in TASK_NAMES:
            actions1 = trajs1[task_name]["actions"]
            actions2 = trajs2[task_name]["actions"]
            n_eps = eps
            for ep_idx in range(n_eps):
                seed_used = args.seed + ep_idx * 17
                arr1 = np.array(actions1[ep_idx])
                arr2 = np.array(actions2[ep_idx])
                equal = np.array_equal(arr1, arr2)
                close = np.allclose(arr1, arr2, atol=1e-6)
                if equal:
                    result = "exactly equal"
                    diff = "N/A"
                elif close:
                    result = "numerically close (≤1e-6)"
                    diff = "N/A"
                else:
                    if arr1.shape != arr2.shape:
                        min_len = min(len(arr1), len(arr2))
                        per_step_equal = np.isclose(arr1[:min_len], arr2[:min_len], atol=1e-6).all(axis=1)
                        diff_idxs = np.where(~per_step_equal)[0]
                        diff_step = diff_idxs[0] if diff_idxs.size > 0 else f"length mismatch ({len(arr1)} vs {len(arr2)})"
                    else:
                        per_step_equal = np.isclose(arr1, arr2, atol=1e-6).all(axis=1)
                        diff_idxs = np.where(~per_step_equal)[0]
                        diff_step = diff_idxs[0] if diff_idxs.size > 0 else 'N/A'
                    result = 'different'
                print(f"[REPRO] Task {task_name:<20} Episode {ep_idx+1:2d} Seed {seed_used:5d}: {result}; First divergence step: {diff_step}")
        os.remove(tmp1_path)
        os.remove(tmp2_path)
    # Combine metrics per task
    combined_results = []
    total_test_steps = 0
    weighted_mae_sum = 0.0
    weighted_mse_sum = 0.0
    total_episodes = 0
    total_successes = 0

    for task_name in TASK_NAMES:
        off = offline_metrics.get(task_name, {"step_mae": 0.0, "step_mse": 0.0, "test_steps": 0})
        cl = closed_loop_metrics.get(task_name, {"episodes": 0, "successes": 0, "success_rate": 0.0})

        combined_results.append(
            {
                "task_name": task_name,
                "test_steps": off["test_steps"],
                "step_mae": off["step_mae"],
                "step_mse": off["step_mse"],
                "episodes": cl["episodes"],
                "successes": cl["successes"],
                "success_rate": cl["success_rate"],
            }
        )

        total_test_steps += off["test_steps"]
        weighted_mae_sum += off["step_mae"] * off["test_steps"]
        weighted_mse_sum += off["step_mse"] * off["test_steps"]
        total_episodes += cl["episodes"]
        total_successes += cl["successes"]

    total_summary = {
        "test_steps": total_test_steps,
        "step_mae": weighted_mae_sum / total_test_steps if total_test_steps > 0 else 0.0,
        "step_mse": weighted_mse_sum / total_test_steps if total_test_steps > 0 else 0.0,
        "episodes": total_episodes,
        "successes": total_successes,
        "success_rate": (total_successes / total_episodes) * 100.0 if total_episodes > 0 else 0.0,
    }

    # Step 3: Flagged Gaps
    flagged_gaps = detect_flagged_gaps(combined_results, success_threshold=args.gap_success_threshold)

    # Step 4: Write eval_report.md
    generate_markdown_report(
        combined_results=combined_results,
        flagged_gaps=flagged_gaps,
        total_summary=total_summary,
        output_report_path=args.output_report,
        model_path=args.model_path,
        episodes_per_task=args.episodes_per_task,
        device_str=args.device,
    )

    # Step 5: Print table to console
    table_headers = [
        "Task Name",
        "Test Steps",
        "Step-wise MAE",
        "Step-wise MSE",
        "Rollout Episodes",
        "Rollout Successes",
        "Task Success Rate",
    ]
    table_rows = [
        [
            r["task_name"],
            f"{r['test_steps']:,}",
            f"{r['step_mae']:.4f}",
            f"{r['step_mse']:.5f}",
            r["episodes"],
            r["successes"],
            f"{r['success_rate']:.1f}%",
        ]
        for r in combined_results
    ]
    table_rows.append(
        [
            "TOTAL / OVERALL",
            f"{total_summary['test_steps']:,}",
            f"{total_summary['step_mae']:.4f}",
            f"{total_summary['step_mse']:.5f}",
            total_summary["episodes"],
            total_summary["successes"],
            f"{total_summary['success_rate']:.1f}%",
        ]
    )

    print("\n" + "=" * 92)
    print(" UNIFIED BENCHMARK: STEP-WISE PREDICTION ACCURACY VS CLOSED-LOOP TASK SUCCESS")
    print("=" * 92)
    try:
        print(tabulate(table_rows, headers=table_headers, tablefmt="github"))
    except Exception:
        print(tabulate(table_rows, headers=table_headers, tablefmt="simple"))
    print("=" * 92 + "\n")

    if flagged_gaps:
        print(">> FLAGGED GAPS (Low MAE, Low Success Rate):")
        for g in flagged_gaps:
            print(
                f"   * [{g['task_name']}] Step MAE: {g['step_mae']:.4f} (best half <= {g['median_mae']:.4f}) | "
                f"Success Rate: {g['success_rate']:.1f}% (< {g['threshold']:.1f}%)"
            )
        print()


if __name__ == "__main__":
    main()
