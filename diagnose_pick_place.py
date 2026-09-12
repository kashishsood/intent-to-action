#!/usr/bin/env python3
"""
Diagnostic script for pick-place-v3 rollouts:
Records step-by-step telemetry, renders animated GIFs, and generates trajectory plots
to determine whether failures occur early (pre-grasp) or late (post-grasp).
"""

import argparse
import os
import sys
import warnings

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from tabulate import tabulate

warnings.filterwarnings("ignore", category=UserWarning, module="metaworld")

import metaworld
from train_bc import BCPolicyMLP

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def diagnose_pick_place(
    model_path: str = "models/best_bc_model.pt",
    output_dir: str = "models",
    num_rollouts: int = 3,
    max_steps: int = 500,
    seed: int = 42,
    device_str: str = "cpu",
):
    device = torch.device(device_str)
    checkpoint = torch.load(model_path, map_location=device)

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

    mt10 = metaworld.MT10()
    env = mt10.train_classes["pick-place-v3"](render_mode="rgb_array")
    tasks = [t for t in mt10.train_tasks if t.env_name == "pick-place-v3"]

    os.makedirs(output_dir, exist_ok=True)
    rollouts = []

    print(f"\nRunning {num_rollouts} diagnostic rollouts for 'pick-place-v3'...")

    for idx in range(num_rollouts):
        env.set_task(tasks[idx])
        obs, _ = env.reset(seed=seed + idx * 77)

        frames = []
        steps = []
        rewards = []
        grip_obj_dists = []
        obj_tgt_dists = []
        gripper_openings = []
        grasp_succs = []
        near_objs = []

        for step in range(max_steps):
            if step % 4 == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            gripper_pos = obs[0:3]
            gripper_open = obs[3]
            obj_pos = obs[4:7]
            target_pos = obs[-3:]

            d_grip_obj = float(np.linalg.norm(gripper_pos - obj_pos))
            d_obj_tgt = float(np.linalg.norm(obj_pos - target_pos))

            norm_obs = (obs - obs_mean) / obs_std
            obs_t = torch.from_numpy(norm_obs).float().unsqueeze(0).to(device)

            with torch.no_grad():
                action = model(obs_t).squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)

            steps.append(step)
            rewards.append(float(reward))
            grip_obj_dists.append(d_grip_obj)
            obj_tgt_dists.append(d_obj_tgt)
            gripper_openings.append(float(gripper_open))
            grasp_succs.append(float(info.get("grasp_success", 0.0)))
            near_objs.append(float(info.get("near_object", 0.0)))

            if terminated or truncated:
                break

        gif_path = os.path.join(output_dir, f"pick_place_rollout_{idx + 1}.gif")
        if frames:
            imageio.mimsave(gif_path, frames, fps=25, loop=0)
            print(f"[OK] Saved GIF: '{gif_path}'")

        rollouts.append(
            {
                "id": idx + 1,
                "init_grip_obj": grip_obj_dists[0],
                "min_grip_obj": min(grip_obj_dists),
                "final_grip_obj": grip_obj_dists[-1],
                "ever_near": max(near_objs) > 0.5,
                "ever_grasped": max(grasp_succs) > 0.5,
                "init_obj_tgt": obj_tgt_dists[0],
                "final_obj_tgt": obj_tgt_dists[-1],
                "total_reward": sum(rewards),
                "steps": steps,
                "rewards": rewards,
                "grip_obj_dists": grip_obj_dists,
                "obj_tgt_dists": obj_tgt_dists,
                "gripper_openings": gripper_openings,
            }
        )

    # Plot
    plot_path = os.path.join(output_dir, "pick_place_analysis.png")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=300)
    colors = ["#2563EB", "#DC2626", "#16A34A"]

    for i, r in enumerate(rollouts):
        axes[0, 0].plot(r["steps"], r["rewards"], label=f"Rollout {i+1}", color=colors[i], lw=1.8)
        axes[0, 1].plot(r["steps"], r["grip_obj_dists"], label=f"Rollout {i+1}", color=colors[i], lw=1.8)
        axes[1, 0].plot(r["steps"], r["obj_tgt_dists"], label=f"Rollout {i+1}", color=colors[i], lw=1.8)
        axes[1, 1].plot(r["steps"], r["gripper_openings"], label=f"Rollout {i+1}", color=colors[i], lw=1.8)

    axes[0, 0].set_title("Reward per Timestep", fontweight="bold")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend()

    axes[0, 1].axhline(0.03, color="gray", linestyle="--", label="Contact Threshold (~0.03m)")
    axes[0, 1].set_title("Gripper-to-Object Distance (m)", fontweight="bold")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend()

    axes[1, 0].set_title("Object-to-Target Distance (m)", fontweight="bold")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend()

    axes[1, 1].set_title("Gripper Opening State", fontweight="bold")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved plot: '{plot_path}'")

    # Table
    table = [
        [
            f"Rollout #{r['id']}",
            f"{r['init_grip_obj']:.3f}m",
            f"{r['min_grip_obj']:.3f}m",
            f"{r['final_grip_obj']:.3f}m",
            "YES" if r["ever_near"] else "NO",
            "YES" if r["ever_grasped"] else "NO",
            f"{r['init_obj_tgt']:.3f}m",
            f"{r['final_obj_tgt']:.3f}m",
            f"{r['total_reward']:.1f}",
        ]
        for r in rollouts
    ]
    headers = ["Rollout", "Init Grip-Obj", "Min Grip-Obj", "Final Grip-Obj", "Near?", "Grasped?", "Init Obj-Tgt", "Final Obj-Tgt", "Total Reward"]
    print("\n" + tabulate(table, headers=headers, tablefmt="github") + "\n")


def main():
    parser = argparse.ArgumentParser(description="Diagnose pick-place-v3 rollouts.")
    parser.add_argument("--model-path", type=str, default="models/best_bc_model.pt")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--num-rollouts", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    diagnose_pick_place(
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_rollouts=args.num_rollouts,
        seed=args.seed,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
