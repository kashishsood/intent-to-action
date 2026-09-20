"""
Inspect and verify saved dataset_12tasks files:
1. Verify NPZ files (train.npz, val.npz, test.npz) array shapes and keys.
2. Verify Parquet file columns.
3. Compute per-task force/torque statistics directly from saved parquet/npz data.
"""
import numpy as np
import pandas as pd
from tabulate import tabulate

print("=== 1. VERIFYING NPZ FILES ON DISK ===")
splits = ["train", "val", "test"]
for s in splits:
    path = f"dataset_12tasks/{s}.npz"
    data = np.load(path)
    print(f"\n--- {s}.npz ({path}) ---")
    for k in data.files:
        arr = data[k]
        print(f"  {k:15s}: shape={arr.shape}, dtype={arr.dtype}")
    # Quick checks
    assert data["obs"].shape[1] == 39, f"Expected 39 obs dims, got {data['obs'].shape[1]}"
    assert data["wrenches"].shape[1] == 6, f"Expected 6 wrench dims, got {data['wrenches'].shape[1]}"
    assert data["obs_force"].shape[1] == 45, f"Expected 45 obs_force dims, got {data['obs_force'].shape[1]}"
    print(f"  [OK] Shape assertions passed for {s}.npz")

print("\n=== 2. VERIFYING UNIFIED PARQUET DATASET ON DISK ===")
pq_path = "dataset_12tasks/metaworld_12tasks_dataset.parquet"
df = pd.read_parquet(pq_path)
print(f"Loaded {pq_path}: {len(df):,} total rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")
print(f"Unique tasks ({df['task_name'].nunique()}): {df['task_name'].unique().tolist()}")
print(f"Unique episodes: {df['global_episode_id'].nunique()} (expected 600)")

print("\n=== 3. PER-TASK FORCE/TORQUE SIGNAL VERIFICATION FROM DISK ===")
# wrench_0..2 are Fx, Fy, Fz; wrench_3..5 are Tx, Ty, Tz
df["force_norm"] = np.sqrt(df["wrench_0"]**2 + df["wrench_1"]**2 + df["wrench_2"]**2)
df["torque_norm"] = np.sqrt(df["wrench_3"]**2 + df["wrench_4"]**2 + df["wrench_5"]**2)

table_rows = []
for task, group in df.groupby("task_name", sort=False):
    f_norm = group["force_norm"]
    t_norm = group["torque_norm"]
    contact_active_steps = (f_norm > 1e-4).sum()
    pct_contact = 100.0 * contact_active_steps / len(group)
    mean_f = f_norm.mean()
    max_f = f_norm.max()
    mean_t = t_norm.mean()
    max_t = t_norm.max()
    
    table_rows.append([
        task,
        f"{len(group):,}",
        f"{group['global_episode_id'].nunique()}",
        f"{pct_contact:.1f}%",
        f"{mean_f:.3f} N",
        f"{max_f:.2f} N",
        f"{mean_t:.4f} Nm",
        f"{max_t:.3f} Nm"
    ])

headers = [
    "Task Name",
    "Steps",
    "Eps",
    "% In Contact",
    "Mean Force",
    "Max Force",
    "Mean Torque",
    "Max Torque"
]
print(tabulate(table_rows, headers=headers, tablefmt="github"))
