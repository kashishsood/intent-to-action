"""
Deep-dive inspection into button-press-topdown-v3 force profile:
1. Pull 10-15 timesteps across an episode from saved parquet dataset.
2. Inspect the geoms involved in contact during button-press-topdown-v3.
3. Check the MuJoCo XML properties of the button (joint limits, stiffness, contact params).
"""
import pandas as pd
import numpy as np
import metaworld
from metaworld.policies import SawyerButtonPressTopdownV3Policy
import mujoco

# 1. Inspect saved dataset episode
df = pd.read_parquet("dataset_12tasks/metaworld_12tasks_dataset.parquet")
bp_df = df[df["task_name"] == "button-press-topdown-v3"]
ep0 = bp_df[bp_df["task_episode_id"] == 0]

print("=== 1. TIMESTEP FORCE PROFILE FOR EPISODE 0 (FROM SAVED DATASET) ===")
# Compute force norm
ep0_forces = ep0[["wrench_0", "wrench_1", "wrench_2", "wrench_3", "wrench_4", "wrench_5"]].values
f_norms = np.linalg.norm(ep0_forces[:, :3], axis=1)

# Find first contact timestep
contact_idx = np.where(f_norms > 1e-3)[0]
first_contact = contact_idx[0] if len(contact_idx) > 0 else -1
print(f"Total steps in episode: {len(ep0)}")
print(f"First contact timestep: {first_contact}")

# Sample key timesteps: start, approach, first contact, 5 steps after contact, middle, bottomed out / end
sample_steps = [0, max(0, first_contact - 5), first_contact, first_contact + 2, first_contact + 5, first_contact + 15, first_contact + 50, 250, 400, 499]
sample_steps = sorted(list(set([s for s in sample_steps if 0 <= s < len(ep0)])))

print(f"{'Step':<6} | {'Fx (N)':<10} | {'Fy (N)':<10} | {'Fz (N)':<10} | {'|F| (N)':<10} | {'|tau| (Nm)':<10} | {'Reward':<8} | {'Success':<8}")
print("-" * 80)
for s in sample_steps:
    row = ep0.iloc[s]
    fx, fy, fz = row["wrench_0"], row["wrench_1"], row["wrench_2"]
    fnorm = f_norms[s]
    tx, ty, tz = row["wrench_3"], row["wrench_4"], row["wrench_5"]
    tnorm = np.sqrt(tx**2 + ty**2 + tz**2)
    rew = row["reward"]
    succ = row["success"]
    print(f"{s:<6} | {fx:<10.3f} | {fy:<10.3f} | {fz:<10.3f} | {fnorm:<10.3f} | {tnorm:<10.4f} | {rew:<8.2f} | {succ:<8.0f}")

# 2. Inspect contact pairs and MuJoCo XML properties live
print("\n=== 2. LIVE MUJOCO CONTACT PAIR & XML INSPECTION ===")
mt50 = metaworld.MT50()
env_cls = mt50.train_classes["button-press-topdown-v3"]
env = env_cls()
tasks = [t for t in mt50.train_tasks if t.env_name == "button-press-topdown-v3"]
env.set_task(tasks[0])
env.reset()
policy = SawyerButtonPressTopdownV3Policy()

# Check XML for button joint/geoms
model = env.unwrapped.model
data = env.unwrapped.data

# List all geoms with 'btn' or 'button' or gripper
print("\nRelevant Model Geoms:")
for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    if name and any(k in name.lower() for k in ["btn", "button", "box", "claw", "pad", "hand"]):
        body_id = model.geom_bodyid[i]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        print(f"  Geom {i:2d}: '{name}' (Body: '{bname}')")

# Step environment with policy until contact and see which geoms collide
print("\nTracking contact geom pairs across rollout:")
contact_events_logged = 0
for t in range(500):
    obs = env._get_obs()
    action = policy.get_action(obs)
    env.step(action)
    
    # Check contacts
    gripper_geom_ids = set()
    for g_id in range(model.ngeom):
        b_id = model.geom_bodyid[g_id]
        b_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b_id)
        if b_name and any(p in b_name.lower() for p in ['hand', 'claw', 'pad', 'gripper']):
            gripper_geom_ids.add(g_id)
            
    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = con.geom1, con.geom2
        if g1 in gripper_geom_ids or g2 in gripper_geom_ids:
            g1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1)
            g2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)
            c_force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, c_force)
            fnorm = np.linalg.norm(c_force[:3])
            if fnorm > 10.0 and contact_events_logged < 8:
                print(f"  Step {t:3d}: Contact between '{g1_name}' and '{g2_name}' | Normal force = {c_force[0]:.2f}N | total 3D = {fnorm:.2f}N")
                contact_events_logged += 1

# Check button joint limits / stiffness
for j in range(model.njnt):
    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    if jname and "btn" in jname.lower():
        stiffness = model.jnt_stiffness[j]
        jrange = model.jnt_range[j]
        print(f"\nButton Joint: '{jname}'")
        print(f"  Stiffness: {stiffness}")
        print(f"  Range: {jrange}")
