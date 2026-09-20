#!/usr/bin/env python3
"""
verify_contact_force_fixed.py
-----------------------------
Corrects the contact-force extraction using MuJoCo's native collision contact API
(`data.contact` and `mujoco.mj_contactForce`), strictly isolating genuine physical
contact events on the gripper and filtering out internal actuation/gravity bias.
"""

import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="metaworld")

import metaworld
import metaworld.policies as policies
import mujoco

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass


def get_gripper_geom_ids(model):
    """Finds all geom IDs associated with the gripper/hand and fingers."""
    body_names = [model.body(i).name for i in range(model.nbody)]
    target_bodies = ["hand", "rightclaw", "rightpad", "leftclaw", "leftpad"]
    target_bids = {body_names.index(n) for n in target_bodies if n in body_names}
    
    gripper_geoms = set()
    for g_idx in range(model.ngeom):
        if model.geom_bodyid[g_idx] in target_bids:
            gripper_geoms.add(g_idx)
    return gripper_geoms


def extract_genuine_contact_wrench(model, data, gripper_geoms):
    """
    Computes the net external contact force and torque (6D wrench) acting on the
    gripper geoms using MuJoCo's mj_contactForce.
    Returns: [Fx, Fy, Fz, Tx, Ty, Tz] in world coordinates.
    If no physical contact exists on the gripper, returns EXACTLY [0, 0, 0, 0, 0, 0].
    """
    net_force = np.zeros(3, dtype=np.float64)
    net_torque = np.zeros(3, dtype=np.float64)
    
    # End-effector reference position (hand position) for torque lever arm
    hand_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    hand_pos = data.xpos[hand_bid] if hand_bid != -1 else np.zeros(3)

    c_force_6d = np.zeros(6, dtype=np.float64)
    active_gripper_contacts = 0

    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = con.geom1, con.geom2
        
        is_g1 = g1 in gripper_geoms
        is_g2 = g2 in gripper_geoms
        
        if not (is_g1 or is_g2):
            continue  # Contact does not involve the robot gripper
            
        active_gripper_contacts += 1
        
        # Extract contact force in contact frame: [fn, ft1, ft2, tn, tt1, tt2]
        mujoco.mj_contactForce(model, data, i, c_force_6d)
        
        # Contact frame is con.frame: 3x3 orientation matrix [normal, tangent1, tangent2]
        R = con.frame.reshape(3, 3)
        # Force in world frame: R.T @ [fn, ft1, ft2]
        # In MuJoCo: force on geom2 is along normal, force on geom1 is opposite
        f_contact_world = R.T @ c_force_6d[0:3]
        
        if is_g1:
            # Force exerted ON geom1 by geom2
            f_on_gripper = -f_contact_world
        else:
            # Force exerted ON geom2 by geom1
            f_on_gripper = f_contact_world
            
        net_force += f_on_gripper
        
        # Torque about hand position: r x F
        r = con.pos - hand_pos
        net_torque += np.cross(r, f_on_gripper)
        
    return np.concatenate([net_force, net_torque]), active_gripper_contacts


def run_probe(task_name, policy_cls, mt50, seed=42, max_steps=120):
    env_cls = mt50.train_classes[task_name]
    tasks = [t for t in mt50.train_tasks if t.env_name == task_name]
    
    env = env_cls()
    env.set_task(tasks[0])
    obs, _ = env.reset(seed=seed)
    policy = policy_cls()
    
    model = env.unwrapped.model
    data = env.unwrapped.data
    gripper_geoms = get_gripper_geom_ids(model)
    
    wrenches = []
    contact_counts = []
    
    print(f"\n--- Probing Genuine Contact Force/Torque on '{task_name}' ---", flush=True)
    
    # Step 0 check (before any action)
    w0, c0 = extract_genuine_contact_wrench(model, data, gripper_geoms)
    f0_norm = np.linalg.norm(w0[0:3])
    print(f"  Step 000 (Initial reset) | Contacts={c0:d} | ||F_contact||={f0_norm:6.2f} N | F_xyz={np.round(w0[0:3], 2)}", flush=True)
    
    for step in range(1, max_steps + 1):
        act = policy.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(act)
        
        wrench, n_contacts = extract_genuine_contact_wrench(model, data, gripper_geoms)
        wrenches.append(wrench)
        contact_counts.append(n_contacts)
        
        if step in [5, 20, 35, 50, 70, 95]:
            f_norm = np.linalg.norm(wrench[0:3])
            t_norm = np.linalg.norm(wrench[3:6])
            print(f"  Step {step:03d} | Contacts={n_contacts:d} | Forces [Fx, Fy, Fz]: [{wrench[0]:6.2f}, {wrench[1]:6.2f}, {wrench[2]:6.2f}] N (||F||={f_norm:5.2f}N) | "
                  f"Torques [Tx, Ty, Tz]: [{wrench[3]:6.3f}, {wrench[4]:6.3f}, {wrench[5]:6.3f}] N*m", flush=True)
            
    wrenches = np.array(wrenches)
    contact_counts = np.array(contact_counts)
    f_norms = np.linalg.norm(wrenches[:, 0:3], axis=1)
    t_norms = np.linalg.norm(wrenches[:, 3:6], axis=1)
    
    print(f">> Task '{task_name}' Summary ({max_steps} steps):", flush=True)
    print(f"   Steps with Active Gripper Contact: {np.sum(contact_counts > 0)} / {max_steps} ({np.mean(contact_counts > 0)*100:.1f}%)", flush=True)
    print(f"   Contact Force Norm (N):   Min={f_norms.min():.2f}, Max={f_norms.max():.2f}, Mean={f_norms.mean():.2f}, Std={f_norms.std():.2f}", flush=True)
    print(f"   Contact Torque Norm (Nm): Min={t_norms.min():.3f}, Max={t_norms.max():.3f}, Mean={t_norms.mean():.3f}, Std={t_norms.std():.3f}", flush=True)
    
    if task_name == "reach-v3":
        assert f_norms.max() < 1e-4, f"FAIL: reach-v3 should have zero contact force, got max {f_norms.max()} N!"
        print("   >> VERIFICATION PASSED: reach-v3 is strictly ZERO contact force throughout!", flush=True)
    else:
        assert f_norms.max() > 0.5, f"FAIL: contact task {task_name} should have non-zero contact force, got max {f_norms.max()} N!"
        print(f"   >> VERIFICATION PASSED: {task_name} exhibits clear physical contact forces (peak {f_norms.max():.2f} N)!", flush=True)


def main():
    print("=" * 85, flush=True)
    print(" MUJOCO GENUINE CONTACT-FORCE SENSOR VALIDATION (FILTERED ON mjContact)", flush=True)
    print("=" * 85, flush=True)
    
    print("Initializing MT50 once...", flush=True)
    mt50 = metaworld.MT50(seed=42)
    print("MT50 initialized. Testing contact force filtering...", flush=True)
    
    # 1. Free space task: MUST BE ZERO throughout
    run_probe("reach-v3", policies.SawyerReachV3Policy, mt50)
    
    # 2. Contact grasp task: pick-place-v3
    run_probe("pick-place-v3", policies.SawyerPickPlaceV3Policy, mt50)
    
    # 3. Constrained contact task: door-open-v3
    run_probe("door-open-v3", policies.SawyerDoorOpenV3Policy, mt50)
    
    # 4. Insertion contact task: peg-insert-side-v3
    run_probe("peg-insert-side-v3", policies.SawyerPegInsertionSideV3Policy, mt50)
    
    print("\n" + "=" * 85, flush=True)
    print(" ALL CONTACT-FORCE SENSOR VALIDATIONS COMPLETED AND CONFIRMED CORRECT", flush=True)
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
