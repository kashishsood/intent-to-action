import os
import sys
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="metaworld")

import metaworld
from train_bc import BCPolicyMLP

def main():
    print("=" * 80)
    print(" PHASE 2: POSE-DIFFICULTY VS STOCHASTICITY DECOMPOSITION")
    print("=" * 80)

    device = torch.device("cpu")
    ckpt_path = "models/best_bc_model.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: model checkpoint not found at '{ckpt_path}'")
        sys.exit(1)

    checkpoint = torch.load(ckpt_path, map_location=device)
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
    env_cls = mt10.train_classes["pick-place-v3"]
    all_tasks = [t for t in mt10.train_tasks if t.env_name == "pick-place-v3"]

    # Select 12 representative task indices spanning successful and failing poses from benchmark
    # From 50-ep benchmark: Succeeded at ep 4, 10, 20, 22, 24, 31, 32, 33, 38, 41, 44, 47
    selected_ep_indices = [0, 4, 7, 10, 15, 20, 24, 31, 35, 40, 44, 47]

    repeats = 5
    base_seeds = [1000, 2026, 3000, 4000, 5000]

    results_table = []

    print("\nExecuting 5 repeats for each of the 12 selected benchmark poses...")
    print("-" * 80)

    for ep_idx in selected_ep_indices:
        task_obj = all_tasks[ep_idx % len(all_tasks)]
        
        # Get reference pose information from env
        env = env_cls()
        env.set_task(task_obj)
        ref_obs, _ = env.reset(seed=2026)
        
        obj_pos = ref_obs[3:6]
        goal_pos = ref_obs[36:39]
        pose_str = f"Obj:[{obj_pos[0]:.2f},{obj_pos[1]:.2f},{obj_pos[2]:.2f}] Goal:[{goal_pos[0]:.2f},{goal_pos[1]:.2f},{goal_pos[2]:.2f}]"

        print(f"\nPose #{ep_idx:02d} | {pose_str}")
        
        successes = 0
        repeat_outcomes = []

        for rep_i, r_seed in enumerate(base_seeds):
            env_rep = env_cls()
            env_rep.set_task(task_obj)
            
            torch.manual_seed(r_seed)
            np.random.seed(r_seed)
            obs, info = env_rep.reset(seed=r_seed)
            
            ep_success = False
            for step in range(500):
                norm_obs = (obs - obs_mean) / obs_std
                obs_t = torch.from_numpy(norm_obs).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    action = model(obs_t).squeeze(0).cpu().numpy()

                obs, reward, terminated, truncated, step_info = env_rep.step(action)
                if float(step_info.get("success", 0.0)) > 0.5:
                    ep_success = True
                if terminated or truncated:
                    break
            
            if ep_success:
                successes += 1
            repeat_outcomes.append(ep_success)
            print(f"  Repeat {rep_i+1}/5 (Seed {r_seed:4d}): Success = {ep_success}")
        
        succ_rate = (successes / repeats) * 100.0
        results_table.append({
            "ep_idx": ep_idx,
            "obj_pos": obj_pos,
            "goal_pos": goal_pos,
            "successes": successes,
            "repeats": repeats,
            "succ_rate": succ_rate,
            "outcomes": repeat_outcomes
        })

    print("\n" + "=" * 80)
    print(" SUMMARY TABLE: PER-POSE DECOMPOSITION")
    print("=" * 80)
    print(f"{'Pose #':<8} | {'Obj Pos (x,y,z)':<22} | {'Goal Pos (x,y,z)':<22} | {'Successes':<10} | {'Success Rate':<12} | {'Variance Class'}")
    print("-" * 80)
    
    consistent_easy = 0
    consistent_hard = 0
    stochastic_mixed = 0

    for r in results_table:
        obj_s = f"[{r['obj_pos'][0]:.2f}, {r['obj_pos'][1]:.2f}, {r['obj_pos'][2]:.2f}]"
        goal_s = f"[{r['goal_pos'][0]:.2f}, {r['goal_pos'][1]:.2f}, {r['goal_pos'][2]:.2f}]"
        rate = r['succ_rate']
        if rate == 100.0:
            var_class = "Consistently Easy (100%)"
            consistent_easy += 1
        elif rate == 0.0:
            var_class = "Consistently Hard (0%)"
            consistent_hard += 1
        else:
            var_class = f"Stochastic Mixed ({rate:.0f}%)"
            stochastic_mixed += 1
            
        print(f"Pose #{r['ep_idx']:<02d}  | {obj_s:<22} | {goal_s:<22} | {r['successes']}/{r['repeats']:<8} | {rate:>11.1f}% | {var_class}")

    print("-" * 80)
    print(f"Total Poses Evaluated: {len(results_table)}")
    print(f"  - Consistently Easy (100%):  {consistent_easy}")
    print(f"  - Consistently Hard (0%):    {consistent_hard}")
    print(f"  - Stochastic / Mixed (20-80%): {stochastic_mixed}")
    print("=" * 80)

if __name__ == "__main__":
    main()
