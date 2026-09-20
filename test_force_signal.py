#!/usr/bin/env python3
"""
test_force_signal.py
--------------------
Verifies the MuJoCo contact-force/torque signals during actual oracle rollouts.
Instantiates MT50 once and probes representative tasks with unbuffered output.
"""

import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="metaworld")

import metaworld
import metaworld.policies as policies

# Unbuffered stdout
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass


def get_gripper_body_indices(env):
    """Finds body indices related to the end-effector and gripper fingers."""
    body_names = [env.unwrapped.model.body(i).name for i in range(env.unwrapped.model.nbody)]
    target_names = ["hand", "rightclaw", "rightpad", "leftclaw", "leftpad", "right_wrist"]
    indices = [body_names.index(name) for name in target_names if name in body_names]
    return indices, [body_names[i] for i in indices]


def extract_gripper_wrench(env, gripper_indices):
    """
    Extracts the net 6D external force/torque (wrench) acting on the gripper/fingers.
    cfrc_ext shape is (nbody, 6): [torque_x, torque_y, torque_z, force_x, force_y, force_z].
    Returns [fx, fy, fz, tx, ty, tz].
    """
    cfrc = env.unwrapped.data.cfrc_ext
    net_wrench = np.sum(cfrc[gripper_indices, :], axis=0)
    torques = net_wrench[0:3]
    forces = net_wrench[3:6]
    return np.concatenate([forces, torques])


def run_probe(task_name, policy_cls, mt50, seed=42, max_steps=120):
    env_cls = mt50.train_classes[task_name]
    tasks = [t for t in mt50.train_tasks if t.env_name == task_name]
    
    env = env_cls()
    env.set_task(tasks[0])
    obs, _ = env.reset(seed=seed)
    policy = policy_cls()
    
    gripper_indices, names = get_gripper_body_indices(env)
    
    wrenches = []
    
    print(f"\n--- Probing Force/Torque on '{task_name}' (Gripper bodies: {names}) ---", flush=True)
    for step in range(max_steps):
        act = policy.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(act)
        
        wrench = extract_gripper_wrench(env, gripper_indices)
        wrenches.append(wrench)
        
        # Sample steps to observe dynamics
        if step in [5, 20, 35, 50, 70, 95]:
            f_norm = np.linalg.norm(wrench[0:3])
            t_norm = np.linalg.norm(wrench[3:6])
            print(f"  Step {step:03d} | Forces [Fx, Fy, Fz]: [{wrench[0]:6.2f}, {wrench[1]:6.2f}, {wrench[2]:6.2f}] N (||F||={f_norm:5.2f}N) | "
                  f"Torques [Tx, Ty, Tz]: [{wrench[3]:6.3f}, {wrench[4]:6.3f}, {wrench[5]:6.3f}] N*m", flush=True)
            
    wrenches = np.array(wrenches)
    f_norms = np.linalg.norm(wrenches[:, 0:3], axis=1)
    t_norms = np.linalg.norm(wrenches[:, 3:6], axis=1)
    
    print(f">> Task '{task_name}' Force/Torque Stats over {max_steps} steps:", flush=True)
    print(f"   Force Norm (N):   Min={f_norms.min():.2f}, Max={f_norms.max():.2f}, Mean={f_norms.mean():.2f}, Std={f_norms.std():.2f}", flush=True)
    print(f"   Torque Norm (Nm): Min={t_norms.min():.3f}, Max={t_norms.max():.3f}, Mean={t_norms.mean():.3f}, Std={t_norms.std():.3f}", flush=True)
    
    is_varying = f_norms.std() > 1e-3 and f_norms.max() > 0.1
    print(f"   Signal Validity Check: {'[CONFIRMED DYNAMICALLY VARYING]' if is_varying else '[STATIC/ZERO]'}", flush=True)
    return wrenches


def main():
    print("=" * 85, flush=True)
    print(" PHASE 2 VERIFICATION: MUJOCO CONTACT FORCE & TORQUE SENSOR SIGNAL VALIDATION", flush=True)
    print("=" * 85, flush=True)
    
    print("Initializing MT50 once...", flush=True)
    mt50 = metaworld.MT50(seed=42)
    print("MT50 initialized successfully. Running task probes...", flush=True)
    
    # 1. Free-space control task: reach-v3
    run_probe("reach-v3", policies.SawyerReachV3Policy, mt50)
    
    # 2. Key contact manipulation task: pick-place-v3
    run_probe("pick-place-v3", policies.SawyerPickPlaceV3Policy, mt50)
    
    # 3. Mechanically constrained contact task: door-open-v3
    run_probe("door-open-v3", policies.SawyerDoorOpenV3Policy, mt50)
    
    # 4. Precision insertion task: peg-insert-side-v3
    run_probe("peg-insert-side-v3", policies.SawyerPegInsertionSideV3Policy, mt50)
    
    print("\n" + "=" * 85, flush=True)
    print(" FORCE/TORQUE SENSOR VALIDATION COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
