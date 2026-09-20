import os
import sys
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="metaworld")

import metaworld
from train_bc import BCPolicyMLP

def evaluate_model_on_poses(model, obs_mean, obs_std, all_tasks, selected_indices, seeds, device, env_cls, model_name="Model"):
    results = {}

    print(f"\n--- Evaluating {model_name} on {len(selected_indices)} fixed poses ({len(seeds)} repeats per pose) ---")
    for idx in selected_indices:
        task_obj = all_tasks[idx % len(all_tasks)]
        succ_count = 0
        
        for r_seed in seeds:
            env = env_cls()
            env.set_task(task_obj)
            torch.manual_seed(r_seed)
            np.random.seed(r_seed)
            obs, _ = env.reset(seed=r_seed)
            
            ep_succ = False
            for step in range(500):
                norm_obs = (obs - obs_mean) / obs_std
                obs_t = torch.from_numpy(norm_obs).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    action = model(obs_t).squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action)
                if float(info.get("success", 0.0)) > 0.5:
                    ep_succ = True
                if terminated or truncated:
                    break
            if ep_succ:
                succ_count += 1
        
        succ_rate = (succ_count / len(seeds)) * 100.0
        results[idx] = {
            "successes": succ_count,
            "repeats": len(seeds),
            "succ_rate": succ_rate,
        }
        print(f"  Pose #{idx:02d}: Success = {succ_count}/{len(seeds)} ({succ_rate:.1f}%)")
    
    total_succ = sum(r["successes"] for r in results.values())
    total_trials = len(selected_indices) * len(seeds)
    overall_rate = (total_succ / total_trials) * 100.0
    print(f">> {model_name} Overall Pose Set Success Rate: {total_succ}/{total_trials} ({overall_rate:.1f}%)")
    return results, overall_rate


def main():
    print("=" * 80)
    print(" PHASE 3: PAIRED EVALUATION (BASELINE VS. PROXIMITY-WEIGHTED)")
    print("=" * 80)

    device = torch.device("cpu")
    mt10 = metaworld.MT10(seed=42)
    env_cls = mt10.train_classes["pick-place-v3"]
    all_tasks = [t for t in mt10.train_tasks if t.env_name == "pick-place-v3"]
    selected_indices = [0, 4, 7, 10, 15, 20, 24, 31, 35, 40, 44, 47]
    seeds = [1000, 2026, 3000, 4000, 5000]

    # 1. Load Baseline Model
    base_ckpt_path = "models/best_bc_model.pt"
    base_ckpt = torch.load(base_ckpt_path, map_location=device)
    base_model = BCPolicyMLP(obs_dim=39, act_dim=4, hidden_dim=256).to(device)
    base_model.load_state_dict(base_ckpt["model_state_dict"])
    base_model.eval()

    # 2. Load Weighted Model
    weight_ckpt_path = "models/best_bc_weighted_model.pt"
    if not os.path.exists(weight_ckpt_path):
        print(f"Error: weighted model checkpoint '{weight_ckpt_path}' not found! Run train_bc_weighted.py first.")
        sys.exit(1)
    weight_ckpt = torch.load(weight_ckpt_path, map_location=device)
    weight_model = BCPolicyMLP(obs_dim=39, act_dim=4, hidden_dim=256).to(device)
    weight_model.load_state_dict(weight_ckpt["model_state_dict"])
    weight_model.eval()

    # Evaluate both arms
    base_results, base_overall = evaluate_model_on_poses(
        base_model, base_ckpt["obs_mean"], base_ckpt["obs_std"],
        all_tasks, selected_indices, seeds, device, env_cls, "Baseline Model"
    )
    
    weight_results, weight_overall = evaluate_model_on_poses(
        weight_model, weight_ckpt["obs_mean"], weight_ckpt["obs_std"],
        all_tasks, selected_indices, seeds, device, env_cls, "Weighted Model"
    )

    # Comparison Table
    print("\n" + "=" * 95)
    print(" PAIRED COMPARISON TABLE: BASELINE VS. PROXIMITY-WEIGHTED ON FIXED POSES")
    print("=" * 95)
    print(f"{'Pose #':<8} | {'Phase 2 Baseline':<20} | {'Weighted Model':<20} | {'Delta':<10} | {'Outcome Verdict'}")
    print("-" * 95)

    flipped_to_success = 0
    degraded_easy = 0
    maintained_easy = 0
    maintained_hard = 0

    for idx in selected_indices:
        b = base_results[idx]["succ_rate"]
        w = weight_results[idx]["succ_rate"]
        delta = w - b

        if b == 0.0 and w > 0.0:
            verdict = f"[IMPROVED] Flipped to Success (+{delta:.0f}%)"
            flipped_to_success += 1
        elif b == 100.0 and w < 100.0:
            verdict = f"[DEGRADED] Lost Easy Pose ({delta:.0f}%)"
            degraded_easy += 1
        elif b == 100.0 and w == 100.0:
            verdict = "[MAINTAINED] Consistently Easy (100%)"
            maintained_easy += 1
        else:
            verdict = "[MAINTAINED] Still Hard (0%)"
            maintained_hard += 1

        print(f"Pose #{idx:02d}  | {b:>6.1f}% ({base_results[idx]['successes']}/{base_results[idx]['repeats']})        | {w:>6.1f}% ({weight_results[idx]['successes']}/{weight_results[idx]['repeats']})        | {delta:>+6.1f}%    | {verdict}")

    print("-" * 95)
    print(f"OVERALL  | {base_overall:>6.1f}%               | {weight_overall:>6.1f}%               | {weight_overall - base_overall:>+6.1f}%    | Net Change: {weight_overall - base_overall:+.1f}%")
    print("=" * 95)
    print(f"\nSummary of Pose Dynamics:")
    print(f"  - Hard Poses Flipped to Success:     {flipped_to_success} / 8")
    print(f"  - Easy Poses Degraded:               {degraded_easy} / 4")
    print(f"  - Easy Poses Maintained (100%):      {maintained_easy} / 4")
    print(f"  - Hard Poses Remaining Hard (0%):    {maintained_hard} / 8")


if __name__ == "__main__":
    main()
