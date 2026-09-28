#!/usr/bin/env python3
"""
compare_gru_tensors.py
Directly compares the GRU input window tensor at t=0, 3, and 8 between:
1. Training sample (from build_sequences_vectorized on dataset)
2. Closed-loop rollout sliding window buffer
"""

import numpy as np
import torch
import metaworld
from train_12tasks_study import build_sequences_vectorized

def main():
    train_npz = np.load("dataset_12tasks/train.npz")
    train_raw = train_npz["obs"].astype(np.float32)
    obs_mean = train_raw.mean(axis=0)
    obs_std = train_raw.std(axis=0)
    obs_std = np.where(obs_std < 1e-6, 1.0, obs_std)
    
    train_norm = (train_raw - obs_mean) / obs_std
    
    # 1. Training sequences for Episode 0 (steps 0 to 500)
    history_len = 8
    train_seq = build_sequences_vectorized(
        train_norm[:500],
        np.array([0]),
        np.array([500]),
        history_len=history_len
    ) # (500, 8, 39)
    
    # Let's inspect Episode 0 from training data as if it were a rollout:
    ep_obs_norm = train_norm[:500]
    
    # 2. Rollout sliding window construction:
    # At t=0, rollout receives obs_norm[0].
    # Initial padding: (H - 1) copies of obs_norm[0].
    pad = [ep_obs_norm[0]] * (history_len - 1)
    rollout_history = [ep_obs_norm[0]]
    
    rollout_windows = {}
    for t in range(10):
        if t > 0:
            rollout_history.append(ep_obs_norm[t])
        
        current_padded = pad + rollout_history
        win = np.array(current_padded[-history_len:], dtype=np.float32)
        rollout_windows[t] = win

    print("=" * 80)
    print("VERIFICATION OF GRU INPUT WINDOW TENSORS (SHAPE 8 x 39)")
    print("=" * 80)
    
    for t in [0, 3, 8]:
        t_seq = train_seq[t]
        r_seq = rollout_windows[t]
        
        diff = np.max(np.abs(t_seq - r_seq))
        print(f"\n--- TIMESTEP t = {t} ---")
        print(f"Max absolute difference between train window and rollout window: {diff:.8f}")
        print(f"Identical arrays: {np.array_equal(t_seq, r_seq)}")
        print(f"Train window shape: {t_seq.shape} | Rollout window shape: {r_seq.shape}")
        
        # Print summary of features across the 8 time steps (showing first 4 obs dimensions)
        print("Train Window [timesteps 0..7, dims 0..3]:")
        for step_i in range(8):
            row_str = " ".join([f"{v:8.4f}" for v in t_seq[step_i, :4]])
            print(f"  step {step_i}: [{row_str} ...]")
            
        print("Rollout Window [timesteps 0..7, dims 0..3]:")
        for step_i in range(8):
            row_str = " ".join([f"{v:8.4f}" for v in r_seq[step_i, :4]])
            print(f"  step {step_i}: [{row_str} ...]")

if __name__ == "__main__":
    main()
