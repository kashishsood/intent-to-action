#!/usr/bin/env python3
"""
Meta-World MT10 Demonstration Data Collection & Dataset Builder

Selects 5 manipulation tasks from Meta-World's MT10 benchmark, runs the built-in
scripted policies for 50 episodes per task, records observations, actions, and
success flags at each timestep, and saves structured Parquet and NPZ datasets split
by episode (train/val/test). Finally, prints a summary table of policy performance.
"""

import argparse
import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Suppress MetaWorld constant clipping warnings for cleaner logging
warnings.filterwarnings("ignore", category=UserWarning, module="metaworld")

import metaworld
import metaworld.policies as policies


# Selected 5 diverse manipulation tasks from MT10 and their scripted policies
SELECTED_TASKS = {
    "reach-v3": policies.SawyerReachV3Policy,
    "pick-place-v3": policies.SawyerPickPlaceV3Policy,
    "door-open-v3": policies.SawyerDoorOpenV3Policy,
    "drawer-open-v3": policies.SawyerDrawerOpenV3Policy,
    "button-press-topdown-v3": policies.SawyerButtonPressTopdownV3Policy,
}


def get_episode_splits(
    num_episodes: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[int, str]:
    """
    Partitions episode indices strictly into train, val, and test splits.
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(num_episodes)
    rng.shuffle(indices)

    n_train = int(num_episodes * train_ratio)
    n_val = int(num_episodes * val_ratio)

    train_eps = set(indices[:n_train])
    val_eps = set(indices[n_train : n_train + n_val])
    test_eps = set(indices[n_train + n_val :])

    split_map = {}
    for ep in range(num_episodes):
        if ep in train_eps:
            split_map[ep] = "train"
        elif ep in val_eps:
            split_map[ep] = "val"
        else:
            split_map[ep] = "test"

    return split_map


def collect_demonstrations(
    episodes_per_task: int = 50,
    max_steps_per_episode: int = 500,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """
    Runs scripted policies on the selected MT10 tasks and records transition data.
    """
    print(f"\nInitializing MT10 Benchmark...")
    mt10 = metaworld.MT10()

    all_transitions = []
    task_summaries = []

    global_ep_id = 0

    for task_id, (task_name, policy_cls) in enumerate(SELECTED_TASKS.items()):
        print(f"\n[{task_id + 1}/{len(SELECTED_TASKS)}] Collecting task: '{task_name}'...")

        env_cls = mt10.train_classes[task_name]
        env = env_cls()
        task_instances = [t for t in mt10.train_tasks if t.env_name == task_name]

        if len(task_instances) < episodes_per_task:
            raise ValueError(
                f"Task '{task_name}' has only {len(task_instances)} tasks, but {episodes_per_task} requested."
            )

        # Generate split map for this task's episodes
        split_map = get_episode_splits(
            episodes_per_task,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed + task_id * 1000,
        )

        successful_episodes = 0
        task_total_steps = 0
        split_counts = {"train": 0, "val": 0, "test": 0}

        for ep_idx in tqdm(range(episodes_per_task), desc=f"  {task_name}", unit="ep"):
            env.set_task(task_instances[ep_idx])
            obs, info = env.reset(seed=seed + ep_idx)
            policy = policy_cls()

            split_name = split_map[ep_idx]
            split_counts[split_name] += 1

            ep_transitions = []
            ep_success = False

            for step in range(max_steps_per_episode):
                action = policy.get_action(obs)
                next_obs, reward, terminated, truncated, step_info = env.step(action)

                step_success = float(step_info.get("success", 0.0))
                if step_success > 0.5:
                    ep_success = True

                ep_transitions.append(
                    {
                        "task_name": task_name,
                        "task_id": task_id,
                        "global_episode_id": global_ep_id,
                        "task_episode_id": ep_idx,
                        "timestep": step,
                        "obs": obs.astype(np.float32),
                        "action": action.astype(np.float32),
                        "reward": float(reward),
                        "success": step_success,
                        "split": split_name,
                    }
                )

                obs = next_obs
                task_total_steps += 1

                if terminated or truncated:
                    break

            if ep_success:
                successful_episodes += 1

            # Annotate transition with overall episode success flag
            for t in ep_transitions:
                t["episode_success"] = 1.0 if ep_success else 0.0

            all_transitions.extend(ep_transitions)
            global_ep_id += 1

        success_rate = (successful_episodes / episodes_per_task) * 100.0
        task_summaries.append(
            {
                "task_name": task_name,
                "total_episodes": episodes_per_task,
                "train_eps": split_counts["train"],
                "val_eps": split_counts["val"],
                "test_eps": split_counts["test"],
                "successful_eps": successful_episodes,
                "success_rate": success_rate,
                "total_steps": task_total_steps,
            }
        )

    return all_transitions, task_summaries


def save_datasets(transitions: List[dict], output_dir: str):
    """
    Saves dataset into Parquet and NPZ formats partitioned strictly by episode.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nFormatting and saving dataset to '{output_dir}'...")

    # Build primary pandas DataFrame
    rows = []
    obs_len = len(transitions[0]["obs"])
    act_len = len(transitions[0]["action"])

    for t in transitions:
        row = {
            "task_name": t["task_name"],
            "task_id": t["task_id"],
            "global_episode_id": t["global_episode_id"],
            "task_episode_id": t["task_episode_id"],
            "timestep": t["timestep"],
            "reward": t["reward"],
            "success": t["success"],
            "episode_success": t["episode_success"],
            "split": t["split"],
            "obs": t["obs"].tolist(),
            "action": t["action"].tolist(),
        }
        # Flattened fields for fast column queries
        for i in range(obs_len):
            row[f"obs_{i}"] = float(t["obs"][i])
        for j in range(act_len):
            row[f"action_{j}"] = float(t["action"][j])
        rows.append(row)

    df = pd.DataFrame(rows)

    # 1. Save unified and split Parquet files
    full_parquet_path = os.path.join(output_dir, "metaworld_mt10_dataset.parquet")
    df.to_parquet(full_parquet_path, index=False, engine="pyarrow")
    print(f"  Saved unified Parquet: {full_parquet_path} ({len(df):,} timesteps)")

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        split_parquet_path = os.path.join(output_dir, f"{split}.parquet")
        split_df.to_parquet(split_parquet_path, index=False, engine="pyarrow")
        n_eps = split_df["global_episode_id"].nunique()
        print(f"  Saved {split} Parquet: {split_parquet_path} ({n_eps} episodes, {len(split_df):,} timesteps)")

    # 2. Save NPZ arrays for vectorized loading
    for split in ["train", "val", "test"]:
        split_records = [t for t in transitions if t["split"] == split]
        if not split_records:
            continue

        obs_arr = np.array([t["obs"] for t in split_records], dtype=np.float32)
        act_arr = np.array([t["action"] for t in split_records], dtype=np.float32)
        rew_arr = np.array([t["reward"] for t in split_records], dtype=np.float32)
        succ_arr = np.array([t["success"] for t in split_records], dtype=np.float32)
        ep_ids = np.array([t["global_episode_id"] for t in split_records], dtype=np.int32)
        task_ids = np.array([t["task_id"] for t in split_records], dtype=np.int32)

        # Episode boundary markers
        ep_change = np.concatenate(([True], ep_ids[1:] != ep_ids[:-1]))
        episode_starts = np.where(ep_change)[0].astype(np.int32)
        episode_ends = np.concatenate((episode_starts[1:], [len(split_records)])).astype(np.int32)

        npz_path = os.path.join(output_dir, f"{split}.npz")
        np.savez_compressed(
            npz_path,
            obs=obs_arr,
            actions=act_arr,
            rewards=rew_arr,
            successes=succ_arr,
            episode_ids=ep_ids,
            task_ids=task_ids,
            episode_starts=episode_starts,
            episode_ends=episode_ends,
        )
        print(f"  Saved {split} NPZ: {npz_path} (obs: {obs_arr.shape}, actions: {act_arr.shape})")


def print_summary_table(summaries: List[dict]):
    """
    Renders a formatted table of dataset counts and scripted policy success rates.
    """
    headers = [
        "Task Name",
        "Total Eps",
        "Train Eps",
        "Val Eps",
        "Test Eps",
        "Successful Eps",
        "Success Rate",
        "Total Steps",
    ]
    table_rows = []

    total_eps = sum(s["total_episodes"] for s in summaries)
    total_train = sum(s["train_eps"] for s in summaries)
    total_val = sum(s["val_eps"] for s in summaries)
    total_test = sum(s["test_eps"] for s in summaries)
    total_succ = sum(s["successful_eps"] for s in summaries)
    total_steps = sum(s["total_steps"] for s in summaries)
    overall_rate = (total_succ / total_eps) * 100.0 if total_eps > 0 else 0.0

    for s in summaries:
        table_rows.append(
            [
                s["task_name"],
                s["total_episodes"],
                s["train_eps"],
                s["val_eps"],
                s["test_eps"],
                s["successful_eps"],
                f"{s['success_rate']:.1f}%",
                f"{s['total_steps']:,}",
            ]
        )

    # Append summary totals
    table_rows.append(
        [
            "TOTAL / OVERALL",
            total_eps,
            total_train,
            total_val,
            total_test,
            total_succ,
            f"{overall_rate:.1f}%",
            f"{total_steps:,}",
        ]
    )

    # Use ASCII / github table formatting to avoid Windows CP1252 encoding issues
    print("\n" + "=" * 80)
    print("      META-WORLD MT10 SCRIPTED POLICY BENCHMARK & DEMONSTRATION SUMMARY")
    print("=" * 80)
    try:
        print(tabulate(table_rows, headers=headers, tablefmt="github"))
    except Exception:
        print(tabulate(table_rows, headers=headers, tablefmt="simple"))
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Record Meta-World MT10 scripted policy demonstrations."
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=50,
        help="Number of episodes per manipulation task (default: 50).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Max timesteps per episode (default: 500).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset",
        help="Directory where structured parquet and npz datasets will be saved.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proportion of episodes for training (default: 0.70).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of episodes for validation (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Read existing dataset and print the summary table without re-running simulation.",
    )
    args = parser.parse_args()

    full_parquet = os.path.join(args.output_dir, "metaworld_mt10_dataset.parquet")
    if args.summary_only:
        if not os.path.exists(full_parquet):
            print(f"Error: dataset not found at '{full_parquet}'. Run without --summary-only first.")
            sys.exit(1)
        summarize_from_dataset(args.output_dir)
        return

    transitions, summaries = collect_demonstrations(
        episodes_per_task=args.episodes_per_task,
        max_steps_per_episode=args.max_steps,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    save_datasets(transitions, args.output_dir)
    print_summary_table(summaries)


def summarize_from_dataset(output_dir: str = "dataset"):
    """
    Computes and prints the summary table from existing dataset files.
    """
    full_parquet = os.path.join(output_dir, "metaworld_mt10_dataset.parquet")
    df = pd.read_parquet(full_parquet)

    summaries = []
    for task_name, group in df.groupby("task_name", sort=False):
        ep_df = group.groupby("global_episode_id").first()
        total_eps = len(ep_df)
        train_eps = (ep_df["split"] == "train").sum()
        val_eps = (ep_df["split"] == "val").sum()
        test_eps = (ep_df["split"] == "test").sum()
        successful_eps = int(ep_df["episode_success"].sum())
        success_rate = (successful_eps / total_eps) * 100.0 if total_eps > 0 else 0.0
        total_steps = len(group)

        summaries.append(
            {
                "task_name": task_name,
                "total_episodes": total_eps,
                "train_eps": int(train_eps),
                "val_eps": int(val_eps),
                "test_eps": int(test_eps),
                "successful_eps": successful_eps,
                "success_rate": success_rate,
                "total_steps": total_steps,
            }
        )

    print_summary_table(summaries)


if __name__ == "__main__":
    main()

