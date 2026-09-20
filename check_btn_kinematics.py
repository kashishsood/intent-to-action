import metaworld
from metaworld.policies import SawyerButtonPressTopdownV3Policy
import numpy as np

mt50 = metaworld.MT50()
env_cls = mt50.train_classes['button-press-topdown-v3']
env = env_cls()
tasks = [t for t in mt50.train_tasks if t.env_name == 'button-press-topdown-v3']
env.set_task(tasks[0])
env.reset()
policy = SawyerButtonPressTopdownV3Policy()

print("Step | Hand Z  | Btn Z   | Error Z | Act Z (cmd) | Joint Qpos")
print("-" * 60)
for t in range(120):
    obs = env._get_obs()
    action = policy.get_action(obs)
    od = policy._parse_obs(obs)
    hand_z = od["hand_pos"][2]
    btn_z = od["button_pos"][2]
    err_z = btn_z - hand_z
    j_qpos = env.unwrapped.data.qpos[env.unwrapped.model.jnt_qposadr[env.unwrapped.model.joint("btnbox_joint").id]]
    if t in [0, 50, 58, 60, 63, 65, 70, 75, 80, 100]:
        print(f"{t:4d} | {hand_z:7.4f} | {btn_z:7.4f} | {err_z:7.4f} | {action[2]:11.4f} | {j_qpos:10.4f}")
    env.step(action)
