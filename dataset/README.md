# dataset/

This directory is excluded from version control (see `.gitignore`).

## Contents (when generated)

| File | Size (approx.) | Description |
|---|---|---|
| `train.parquet` | ~19 MB | Training split (70% of episodes) |
| `val.parquet` | ~4 MB | Validation split (15% of episodes) |
| `test.parquet` | ~5 MB | Test split (15% of episodes) |
| `train.npz` | ~4 MB | NPZ copy of training split |
| `val.npz` | ~800 KB | NPZ copy of validation split |
| `test.npz` | ~900 KB | NPZ copy of test split |
| `metaworld_mt10_dataset.parquet` | ~27 MB | Full unsplit dataset |

## How to regenerate

```bash
pip install -r requirements.txt
python collect_demonstrations.py
```

This runs the five built-in Meta-World scripted oracle policies for 50 episodes
each (250 total rollouts) and writes all splits to this directory. Wall-clock
time is approximately 10 minutes on CPU.

## Schema

Each Parquet file contains the following columns:

| Column | dtype | Description |
|---|---|---|
| `task_name` | string | MT10 task identifier, e.g. `reach-v3` |
| `episode_id` | int | Episode index within task (0–49) |
| `timestep` | int | Step index within episode |
| `obs` | list[float] | 39-dimensional observation vector |
| `action` | list[float] | 4-dimensional action vector (Δx, Δy, Δz, gripper) |
| `success` | bool | `info['success']` from the simulator at this timestep |

The episode-level train/val/test split is stratified at the episode level
(not the timestep level) to prevent data leakage.
