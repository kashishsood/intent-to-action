import json
import math

def wilson_exact(k, n, z=1.6448536):
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (center - margin) * 100, (center + margin) * 100

def paired_diff_ci(v_x, v_y, z=1.6448536):
    n = len(v_x)
    diffs = [x - y for x, y in zip(v_x, v_y)]
    d_mean = sum(diffs) / n
    var = sum((d - d_mean)**2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    return d_mean * 100, (d_mean - z * se) * 100, (d_mean + z * se) * 100

data = json.load(open('models/benchmark_12tasks_results.json'))

TASK_NAMES = [
    "pick-place-v3",
    "pick-place-wall-v3",
    "sweep-into-v3",
    "assembly-v3",
    "hammer-v3",
    "button-press-topdown-v3",
    "door-open-v3",
    "drawer-open-v3",
    "drawer-close-v3",
    "door-close-v3",
    "peg-insert-side-v3",
    "reach-v3",
]

TASK_REGIMES = {
    "pick-place-v3": "True Contact",
    "pick-place-wall-v3": "True Contact",
    "sweep-into-v3": "True Contact",
    "assembly-v3": "True Contact",
    "hammer-v3": "Hardstop Mechanism",
    "button-press-topdown-v3": "Hardstop Mechanism",
    "door-open-v3": "Hardstop Mechanism",
    "drawer-open-v3": "Hardstop Mechanism",
    "drawer-close-v3": "Hardstop Mechanism",
    "door-close-v3": "Hardstop Mechanism",
    "peg-insert-side-v3": "Hardstop Mechanism",
    "reach-v3": "Free-Space (0 N)",
}

print("| Task Name | Interaction Regime | 12T Kin MLP ($n=20$) | 12T Kin GRU ($n=20$) | 12T Force GRU ($n=20$) | Force Delta vs Kin GRU [90% Paired CI] |")
print("| :--- | :--- | :---: | :---: | :---: | :---: |")

for t in TASK_NAMES:
    regime = TASK_REGIMES[t]
    
    # Kin MLP
    k_mlp_det = data["12-Task Kinematic MLP"][t]["episode_details"]
    k_mlp_s = sum(1 for ep in k_mlp_det if ep["succ"])
    l_kmlp, u_kmlp = wilson_exact(k_mlp_s, 20)
    
    # Kin GRU
    k_gru_det = data["12-Task Kinematic GRU"][t]["episode_details"]
    k_gru_s = sum(1 for ep in k_gru_det if ep["succ"])
    l_kgru, u_kgru = wilson_exact(k_gru_s, 20)
    
    # Force GRU
    f_gru_det = data["12-Task Force GRU"][t]["episode_details"]
    f_gru_s = sum(1 for ep in f_gru_det if ep["succ"])
    l_fgru, u_fgru = wilson_exact(f_gru_s, 20)
    
    # Paired delta
    v_f = [int(ep["succ"]) for ep in f_gru_det]
    v_k = [int(ep["succ"]) for ep in k_gru_det]
    delta, d_low, d_high = paired_diff_ci(v_f, v_k)
    
    bold_f = f"**{f_gru_s/20*100:.1f}%**" if f_gru_s >= k_gru_s else f"{f_gru_s/20*100:.1f}%"
    bold_d = f"**{delta:+.1f}%**" if abs(delta) > 0 else f"{delta:+.1f}%"
    
    print(f"| `{t}` | {regime} | {k_mlp_s/20*100:.1f}% [{l_kmlp:.1f}, {u_kmlp:.1f}] | {k_gru_s/20*100:.1f}% [{l_kgru:.1f}, {u_kgru:.1f}] | {bold_f} [{l_fgru:.1f}, {u_fgru:.1f}] | {bold_d} [{d_low:+.1f}%, {d_high:+.1f}%] |")
