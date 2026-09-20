"""
Analyze Pre-Success vs Post-Success Force Profiles Across All 12 Tasks:
1. Identify the first success timestep per episode.
2. Calculate the fraction of pre-success vs post-success timesteps per task.
3. Compare force metrics (mean force, max force, % in contact) in pre-success vs post-success regimes.
4. Flag tasks dominated by post-success hardstop holding.
"""
import pandas as pd
import numpy as np
from tabulate import tabulate

df = pd.read_parquet("dataset_12tasks/metaworld_12tasks_dataset.parquet")

# Calculate 3D force norm per timestep
df["force_norm"] = np.sqrt(df["wrench_0"]**2 + df["wrench_1"]**2 + df["wrench_2"]**2)

results = []

for task_name, group in df.groupby("task_name", sort=False):
    # Find first success timestep for each episode
    ep_stats = []
    
    for ep_id, ep_df in group.groupby("global_episode_id"):
        succ_steps = ep_df[ep_df["success"] > 0.5]["timestep"]
        if len(succ_steps) > 0:
            first_succ = succ_steps.min()
        else:
            first_succ = len(ep_df)  # never succeeded (if any)
            
        pre = ep_df[ep_df["timestep"] < first_succ]
        post = ep_df[ep_df["timestep"] >= first_succ]
        
        ep_stats.append({
            "first_succ": first_succ,
            "total_len": len(ep_df),
            "pre_len": len(pre),
            "post_len": len(post),
            "pre_f_mean": pre["force_norm"].mean() if len(pre) > 0 else 0.0,
            "pre_f_max": pre["force_norm"].max() if len(pre) > 0 else 0.0,
            "pre_contact_pct": (pre["force_norm"] > 1e-4).mean() * 100.0 if len(pre) > 0 else 0.0,
            "post_f_mean": post["force_norm"].mean() if len(post) > 0 else 0.0,
            "post_f_max": post["force_norm"].max() if len(post) > 0 else 0.0,
            "post_contact_pct": (post["force_norm"] > 1e-4).mean() * 100.0 if len(post) > 0 else 0.0,
        })
        
    ep_df_stats = pd.DataFrame(ep_stats)
    
    mean_succ_step = ep_df_stats["first_succ"].mean()
    pct_pre = 100.0 * ep_df_stats["pre_len"].sum() / (len(group))
    pct_post = 100.0 * ep_df_stats["post_len"].sum() / (len(group))
    
    pre_f_mean = ep_df_stats["pre_f_mean"].mean()
    pre_f_max = ep_df_stats["pre_f_max"].max()
    pre_contact_pct = ep_df_stats["pre_contact_pct"].mean()
    
    post_f_mean = ep_df_stats["post_f_mean"].mean()
    post_f_max = ep_df_stats["post_f_max"].max()
    post_contact_pct = ep_df_stats["post_contact_pct"].mean()
    
    # Check if post-success force ratio is high (> 3x pre-success force)
    ratio = post_f_mean / (pre_f_mean + 1e-6)
    if pct_post > 50.0 and post_f_mean > 50.0 and ratio > 3.0:
        flag = "HARDSTOP HOLD (Post >> Pre)"
    elif pct_post > 50.0 and post_f_mean > 50.0:
        flag = "SUSTAINED CONTACT (High Both)"
    elif post_f_mean < 1.0 and pre_f_mean < 1.0:
        flag = "ZERO / MINIMAL CONTACT"
    else:
        flag = "BALANCED CONTACT DYNAMIC"
        
    results.append({
        "Task": task_name,
        "Mean First Succ": f"{mean_succ_step:.1f} / 500",
        "Pre-Succ Steps": f"{pct_pre:.1f}%",
        "Post-Succ Steps": f"{pct_post:.1f}%",
        "Pre-Succ Mean F": f"{pre_f_mean:.2f} N",
        "Post-Succ Mean F": f"{post_f_mean:.2f} N",
        "Pre-Succ Contact %": f"{pre_contact_pct:.1f}%",
        "Post-Succ Contact %": f"{post_contact_pct:.1f}%",
        "Regime Classification": flag
    })

res_df = pd.DataFrame(results)
print(tabulate(res_df, headers="keys", tablefmt="github", showindex=False))
