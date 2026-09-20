import sys
import numpy as np, torch, os
import metaworld
from train_bc import BCPolicyMLP

if len(sys.argv) < 2:
    print('Usage: python trace_seed.py <seed>')
    sys.exit(1)
seed = int(sys.argv[1])

ckpt_path = 'models/best_bc_model.pt'
if not os.path.exists(ckpt_path):
    print(f'Checkpoint not found: {ckpt_path}')
    sys.exit(1)
checkpoint = torch.load(ckpt_path, map_location='cpu')
obs_mean = checkpoint['obs_mean']
obs_std = checkpoint['obs_std']
model = BCPolicyMLP(obs_dim=checkpoint.get('obs_dim', 39),
                    act_dim=checkpoint.get('act_dim', 4),
                    hidden_dim=checkpoint.get('hidden_dim', 256))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

mt10 = metaworld.MT10()
env_cls = mt10.train_classes['pick-place-v3']
tasks = [t for t in mt10.train_tasks if t.env_name == 'pick-place-v3']

env = env_cls()
env.set_task(tasks[0])
obs, _ = env.reset(seed=seed)
print(f'=== Seed {seed} initial observation ===')
print('obs0:', obs)

norm = (obs - obs_mean) / obs_std
obs_t = torch.from_numpy(norm).float().unsqueeze(0)
with torch.no_grad():
    act = model(obs_t).squeeze(0).cpu().numpy()
print('act0:', act)

max_steps = 500
for step_idx in range(1, max_steps + 1):
    next_obs, rew, term, trunc, info = env.step(act)
    print(f'=== Step {step_idx} ===')
    print('obs:', next_obs)
    print('reward:', rew)
    print('terminated:', term, 'truncated:', trunc)
    if term or trunc:
        print('Episode ended at step', step_idx)
        break
    norm = (next_obs - obs_mean) / obs_std
    obs_t = torch.from_numpy(norm).float().unsqueeze(0)
    with torch.no_grad():
        act = model(obs_t).squeeze(0).cpu().numpy()
    print('act:', act)
