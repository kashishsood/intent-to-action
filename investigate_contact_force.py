import sys
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning, module="metaworld")
import metaworld
import mujoco

# 1. Investigate reach-v3 step 0
mt50 = metaworld.MT50(seed=42)
env = mt50.train_classes["reach-v3"]()
tasks = [t for t in mt50.train_tasks if t.env_name == "reach-v3"]
env.set_task(tasks[0])
obs, _ = env.reset(seed=42)

model = env.unwrapped.model
data = env.unwrapped.data

body_names = [model.body(i).name for i in range(model.nbody)]
hand_indices = [body_names.index(n) for n in ["hand", "rightclaw", "rightpad", "leftclaw", "leftpad", "right_wrist"] if n in body_names]

raw_cfrc_step0 = np.sum(data.cfrc_ext[hand_indices, :], axis=0)
print(f"reach-v3 Step 0 (before any step/action):")
print(f"  raw cfrc_ext forces: {np.round(raw_cfrc_step0[3:6], 2)} N (norm: {np.linalg.norm(raw_cfrc_step0[3:6]):.2f} N)")
print(f"  raw cfrc_ext torques: {np.round(raw_cfrc_step0[0:3], 3)} Nm")
print(f"  Active contacts in scene (data.ncon): {data.ncon}")

# Check contacts in reach-v3 step 0
gripper_geom_ids = set()
for b_idx in hand_indices:
    for g_idx in range(model.ngeom):
        if model.geom_bodyid[g_idx] == b_idx:
            gripper_geom_ids.add(g_idx)

print(f"  Gripper geom IDs: {gripper_geom_ids}")

gripper_contacts = 0
for i in range(data.ncon):
    con = data.contact[i]
    if con.geom1 in gripper_geom_ids or con.geom2 in gripper_geom_ids:
        gripper_contacts += 1

print(f"  Active contacts involving gripper in reach-v3 step 0: {gripper_contacts}")
