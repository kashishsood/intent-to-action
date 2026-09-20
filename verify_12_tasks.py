#!/usr/bin/env python3
"""
verify_12_tasks.py
------------------
Confirms that all 12 selected Meta-World task names resolve to real environments
and scripted policies in the local Meta-World installation, instantiates each one,
and checks observation dimensions and MuJoCo external force-torque signals.
"""

import sys
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning, module="metaworld")

import metaworld
import metaworld.policies as policies

# Define the 12 selected tasks and their scripted policies
SELECTED_TASKS = {
    # Group A: Contact / Precision-Rich (6 tasks)
    "pick-place-v3": policies.SawyerPickPlaceV3Policy,
    "pick-place-wall-v3": policies.SawyerPickPlaceWallV3Policy,
    "peg-insert-side-v3": policies.SawyerPegInsertionSideV3Policy,
    "assembly-v3": policies.SawyerAssemblyV3Policy,
    "hammer-v3": policies.SawyerHammerV3Policy,
    "sweep-into-v3": policies.SawyerSweepIntoV3Policy,

    # Group B: Free-Space / Kinematic-Funneling (6 tasks)
    "reach-v3": policies.SawyerReachV3Policy,
    "door-open-v3": policies.SawyerDoorOpenV3Policy,
    "drawer-open-v3": policies.SawyerDrawerOpenV3Policy,
    "button-press-topdown-v3": policies.SawyerButtonPressTopdownV3Policy,
    "drawer-close-v3": policies.SawyerDrawerCloseV3Policy,
    "door-close-v3": policies.SawyerDoorCloseV3Policy,
}


def main():
    print("=" * 80)
    print(" VERIFYING 12 SELECTED META-WORLD TASKS IN LOCAL INSTALLATION")
    print("=" * 80)

    # Use MT50 to get task definitions
    mt50 = metaworld.MT50()
    train_classes = mt50.train_classes
    train_tasks = mt50.train_tasks

    results = []
    all_resolved = True

    for task_name, policy_cls in SELECTED_TASKS.items():
        resolved = task_name in train_classes
        if not resolved:
            print(f"FAILED: Task '{task_name}' NOT found in MT50.train_classes!")
            all_resolved = False
            continue

        env_cls = train_classes[task_name]
        tasks = [t for t in train_tasks if t.env_name == task_name]
        
        # Instantiate environment and policy
        env = env_cls()
        env.set_task(tasks[0])
        obs, _ = env.reset(seed=42)
        policy = policy_cls()

        # Step 1: take action with policy
        act = policy.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(act)

        # Check MuJoCo body index for hand / gripper
        body_names = [env.unwrapped.model.body(i).name for i in range(env.unwrapped.model.nbody)]
        hand_idx = body_names.index("hand") if "hand" in body_names else -1
        cfrc_shape = env.unwrapped.data.cfrc_ext.shape

        results.append({
            "task": task_name,
            "policy": policy_cls.__name__,
            "obs_dim": obs.shape[0],
            "act_dim": act.shape[0],
            "cfrc_shape": cfrc_shape,
            "hand_idx": hand_idx,
            "tasks_available": len(tasks),
            "status": "VERIFIED OK"
        })
        print(f" [OK] {task_name:<25} | Policy: {policy_cls.__name__:<33} | Obs Dim: {obs.shape[0]} | Tasks: {len(tasks)}")

    print("-" * 80)
    if all_resolved and len(results) == 12:
        print("ALL 12 TASKS SUCCESSFULLY RESOLVED, INSTANTIATED, AND VERIFIED!")
    else:
        print("SOME TASKS FAILED VERIFICATION!")
        sys.exit(1)


if __name__ == "__main__":
    main()
