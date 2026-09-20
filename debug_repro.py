import numpy as np, torch, sys, os
import metaworld
from train_bc import BCPolicyMLP

# Load model checkpoint
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
seed = 2026

# Initialize two fresh environments
env1 = env_cls()
env1.set_task(tasks[0])
obs1, _ = env1.reset(seed=seed)

env2 = env_cls()
env2.set_task(tasks[0])
obs2, _ = env2.reset(seed=seed)

print('=== Observation arrays (raw) ===')
print('obs1:', obs1)
print('obs2:', obs2)
print('obs equal?', np.array_equal(obs1, obs2))

# Compute identical action from model
norm1 = (obs1 - obs_mean) / obs_std
norm2 = (obs2 - obs_mean) / obs_std
obs_t1 = torch.from_numpy(norm1).float().unsqueeze(0)
obs_t2 = torch.from_numpy(norm2).float().unsqueeze(0)
with torch.no_grad():
    act1 = model(obs_t1).squeeze(0).cpu().numpy()
    act2 = model(obs_t2).squeeze(0).cpu().numpy()
print('=== Action arrays (model output) ===')
print('act1:', act1)
print('act2:', act2)
print('actions equal?', np.array_equal(act1, act2))

# Step environments and compare up to 10 steps
max_steps = 30
for step_idx in range(1, max_steps + 1):
    # Step both environments with the same action
    next1 = env1.step(act1)
    next2 = env2.step(act2)
    # Unpack full return tuple (obs, reward, done, info)
    obs1_next, rew1, term1, trunc1, info1 = next1
    obs2_next, rew2, term2, trunc2, info2 = next2
    print(f'=== Step {step_idx} ===')
    print('obs1_next:', obs1_next)
    print('obs2_next:', obs2_next)
    print('obs equal?', np.array_equal(obs1_next, obs2_next))
    print('reward1:', rew1, 'reward2:', rew2, 'equal?', np.isclose(rew1, rew2))
    # Gymnasium may include a 'truncated' flag in the info dict
    trunc1 = info1.get('truncated', False) if isinstance(info1, dict) else False
    trunc2 = info2.get('truncated', False) if isinstance(info2, dict) else False
    print('terminated1:', term1, 'truncated1:', trunc1)
    print('terminated2:', term2, 'truncated2:', trunc2)
    if not np.array_equal(obs1_next, obs2_next) or not np.isclose(rew1, rew2):
        print(f'Divergence detected at step {step_idx}')
        break
    if term1 or term2:
        print('Episode terminated at step', step_idx)
        break
    # Prepare next action
    norm1 = (obs1_next - obs_mean) / obs_std
    norm2 = (obs2_next - obs_mean) / obs_std
    obs_t1 = torch.from_numpy(norm1).float().unsqueeze(0)
    obs_t2 = torch.from_numpy(norm2).float().unsqueeze(0)
    with torch.no_grad():
        act1 = model(obs_t1).squeeze(0).cpu().numpy()
        act2 = model(obs_t2).squeeze(0).cpu().numpy()
    print('act1 (step):', act1)
    print('act2 (step):', act2)



