#!/usr/bin/env python3
"""
Verification suite for collected Meta-World demonstration dataset.
Validates file integrity, strict episode-level partitioning, and tensor dimensions.
"""

import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def verify_dataset(dataset_dir: str = "dataset"):
    print(f"\n==========================================")
    print(f"VERIFYING DATASET INTEGRITY: '{dataset_dir}'")
    print(f"==========================================")

    # 1. Check required files
    expected_files = [
        "metaworld_mt10_dataset.parquet",
        "train.parquet",
        "val.parquet",
        "test.parquet",
        "train.npz",
        "val.npz",
        "test.npz",
    ]
    for fname in expected_files:
        fpath = os.path.join(dataset_dir, fname)
        assert os.path.exists(fpath), f"Missing expected dataset file: {fpath}"
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"[OK] Found: {fname:<32} ({size_mb:.2f} MB)")

    # 2. Check Parquet episode partitioning
    train_df = pd.read_parquet(os.path.join(dataset_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(dataset_dir, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(dataset_dir, "test.parquet"))
    full_df = pd.read_parquet(os.path.join(dataset_dir, "metaworld_mt10_dataset.parquet"))

    train_eps = set(train_df["global_episode_id"].unique())
    val_eps = set(val_df["global_episode_id"].unique())
    test_eps = set(test_df["global_episode_id"].unique())

    print("\nChecking episode partition disjointness...")
    assert train_eps.isdisjoint(val_eps), "Leakage detected between train and val episodes!"
    assert train_eps.isdisjoint(test_eps), "Leakage detected between train and test episodes!"
    assert val_eps.isdisjoint(test_eps), "Leakage detected between val and test episodes!"
    print(f"[OK] Episode sets are strictly disjoint!")
    print(f"  Train episodes: {len(train_eps)} | Val episodes: {len(val_eps)} | Test episodes: {len(test_eps)}")
    print(f"  Total unique episodes: {len(train_eps | val_eps | test_eps)}")

    # Check that total rows match
    total_split_rows = len(train_df) + len(val_df) + len(test_df)
    assert total_split_rows == len(full_df), (
        f"Row count mismatch: split sum {total_split_rows} != full {len(full_df)}"
    )
    print(f"[OK] Total timesteps verified: {len(full_df):,} steps across all splits.")

    # 3. Check observations and actions
    print("\nChecking tensor shapes and null values...")
    assert not full_df.isnull().any().any(), "Found null values in dataset!"

    sample_obs = np.array(full_df["obs"].iloc[0])
    sample_act = np.array(full_df["action"].iloc[0])
    assert sample_obs.shape == (39,), f"Unexpected obs shape: {sample_obs.shape}"
    assert sample_act.shape == (4,), f"Unexpected action shape: {sample_act.shape}"
    print(f"[OK] Observation dim: {sample_obs.shape} | Action dim: {sample_act.shape}")

    # 4. Check NPZ files
    print("\nChecking NPZ archives...")
    for split in ["train", "val", "test"]:
        npz = np.load(os.path.join(dataset_dir, f"{split}.npz"))
        assert "obs" in npz and "actions" in npz and "successes" in npz
        obs = npz["obs"]
        actions = npz["actions"]
        assert obs.ndim == 2 and obs.shape[1] == 39
        assert actions.ndim == 2 and actions.shape[1] == 4
        print(f"[OK] {split}.npz: obs {obs.shape}, actions {actions.shape}, starts {len(npz['episode_starts'])}")

    print("\n==========================================")
    print("ALL INTEGRITY CHECKS PASSED SUCCESSFULLY!")
    print("==========================================\n")


if __name__ == "__main__":
    verify_dataset()
