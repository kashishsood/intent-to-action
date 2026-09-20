#!/usr/bin/env python3
"""
generate_rollout_gifs.py
------------------------
Generates clean, high-quality demonstration GIFs for the three key experimental cases:
1. baseline_success_pose10.gif: Baseline model on consistently easy pose #10 (Outcome: SUCCESS)
2. baseline_failure_pose00.gif: Baseline model on consistently hard pose #00 (Outcome: FAILURE)
3. weighted_failure_pose04.gif: Weighted model on previously-solvable pose #04 (Outcome: FAILURE)
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="metaworld")
import metaworld
from train_bc import BCPolicyMLP

def render_episode(model, obs_mean, obs_std, task_obj, env_cls, seed, device, save_path, capture_every=4, frame_duration=80):
    env = env_cls(render_mode="rgb_array")
    env.set_task(task_obj)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    
    frames = []
    success = False
    first_success_step = None
    
    # Capture initial frame
    f0 = env.render()
    if f0 is not None:
        frames.append(f0)
        
    for step in range(1, 501):
        norm_obs = (obs - obs_mean) / obs_std
        obs_t = torch.from_numpy(norm_obs).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = model(obs_t).squeeze(0).cpu().numpy()
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        if float(info.get("success", 0.0)) > 0.5:
            if not success:
                first_success_step = step
            success = True
            
        if step % capture_every == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
                
        if terminated or truncated:
            break
            
    # Save as clean GIF using PIL
    pil_frames = []
    for f in frames:
        im = Image.fromarray(f)
        # Resize to 360x360 for high visual quality and compact web size
        im_resized = im.resize((360, 360), Image.Resampling.LANCZOS)
        # Quantize to 128 colors with palette optimization
        im_q = im_resized.convert("RGB").quantize(colors=128)
        pil_frames.append(im_q)
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pil_frames[0].save(
        save_path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=frame_duration,
        optimize=True
    )
    
    size_kb = os.path.getsize(save_path) / 1024
    return {
        "success": success,
        "first_success_step": first_success_step,
        "total_frames": len(pil_frames),
        "file_size_kb": size_kb
    }


def main():
    print("=" * 80)
    print(" GENERATING ROLLOUT ANIMATIONS FOR THE THREE KEY CASES")
    print("=" * 80)
    
    device = torch.device("cpu")
    mt10 = metaworld.MT10(seed=42)
    env_cls = mt10.train_classes["pick-place-v3"]
    all_tasks = [t for t in mt10.train_tasks if t.env_name == "pick-place-v3"]
    
    # 1. Load Baseline Model
    base_ckpt = torch.load("models/best_bc_model.pt", map_location=device)
    base_model = BCPolicyMLP(obs_dim=39, act_dim=4, hidden_dim=256).to(device)
    base_model.load_state_dict(base_ckpt["model_state_dict"])
    base_model.eval()
    
    # 2. Load Weighted Model
    weight_ckpt = torch.load("models/best_bc_weighted_model.pt", map_location=device)
    weight_model = BCPolicyMLP(obs_dim=39, act_dim=4, hidden_dim=256).to(device)
    weight_model.load_state_dict(weight_ckpt["model_state_dict"])
    weight_model.eval()
    
    seed = 2026
    
    cases = [
        {
            "name": "Case 1: Baseline Success",
            "model_name": "models/best_bc_model.pt",
            "model": base_model,
            "obs_mean": base_ckpt["obs_mean"],
            "obs_std": base_ckpt["obs_std"],
            "pose_idx": 10,
            "filename": "models/baseline_success_pose10.gif",
            "expected": "SUCCESS"
        },
        {
            "name": "Case 2: Baseline Failure",
            "model_name": "models/best_bc_model.pt",
            "model": base_model,
            "obs_mean": base_ckpt["obs_mean"],
            "obs_std": base_ckpt["obs_std"],
            "pose_idx": 0,
            "filename": "models/baseline_failure_pose00.gif",
            "expected": "FAILURE"
        },
        {
            "name": "Case 3: Weighted-Model Failure (Degraded Pose)",
            "model_name": "models/best_bc_weighted_model.pt",
            "model": weight_model,
            "obs_mean": weight_ckpt["obs_mean"],
            "obs_std": weight_ckpt["obs_std"],
            "pose_idx": 4,
            "filename": "models/weighted_failure_pose04.gif",
            "expected": "FAILURE"
        }
    ]
    
    results = []
    for c in cases:
        print(f"\nRendering {c['name']} (Pose #{c['pose_idx']:02d}, Seed {seed})...")
        task_obj = all_tasks[c["pose_idx"] % len(all_tasks)]
        res = render_episode(
            model=c["model"],
            obs_mean=c["obs_mean"],
            obs_std=c["obs_std"],
            task_obj=task_obj,
            env_cls=env_cls,
            seed=seed,
            device=device,
            save_path=c["filename"],
            capture_every=4,
            frame_duration=80
        )
        outcome_str = "SUCCESS" if res["success"] else "FAILURE"
        succ_step_str = f"at step {res['first_success_step']}" if res["success"] else "never"
        print(f"  Model Checkpoint: {c['model_name']}")
        print(f"  Pose Index:       Pose #{c['pose_idx']:02d}")
        print(f"  Episode Outcome:  {outcome_str} ({succ_step_str})")
        print(f"  Saved GIF:        {c['filename']} ({res['total_frames']} frames, {res['file_size_kb']:.1f} KB)")
        
        results.append({
            "case": c["name"],
            "model": c["model_name"],
            "pose": c["pose_idx"],
            "outcome": outcome_str,
            "frames": res["total_frames"],
            "size_kb": res["file_size_kb"],
            "path": c["filename"]
        })
        
    print("\n" + "=" * 80)
    print(" ALL RENDERS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    for r in results:
        print(f"  - [{r['outcome']}] {r['path']}: {r['case']} | {r['frames']} frames, {r['size_kb']:.1f} KB")

if __name__ == "__main__":
    main()
