#!/usr/bin/env python3
"""
regenerate_report_tables.py
Generates Markdown tables from models/benchmark_window_master.json
and prints per-seed vectors and summary metrics.
"""

import json
import sys
from tabulate import tabulate
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TASK_NAMES = [
    "pick-place-v3",
    "pick-place-wall-v3",
    "peg-insert-side-v3",
    "assembly-v3",
    "hammer-v3",
    "sweep-into-v3",
    "reach-v3",
    "door-open-v3",
    "drawer-open-v3",
    "button-press-topdown-v3",
    "drawer-close-v3",
    "door-close-v3",
]

TASK_REGIMES = {
    "pick-place-v3": "True Contact",
    "pick-place-wall-v3": "True Contact",
    "peg-insert-side-v3": "Hardstop Hold",
    "assembly-v3": "True Contact",
    "hammer-v3": "Hardstop Hold",
    "sweep-into-v3": "True Contact",
    "reach-v3": "Free-Space (0N)",
    "door-open-v3": "Hardstop Hold",
    "drawer-open-v3": "Hardstop Hold",
    "button-press-topdown-v3": "Hardstop Hold",
    "drawer-close-v3": "Hardstop Hold",
    "door-close-v3": "Hardstop Hold",
}

POSE_INDICES = [0, 4, 7, 10, 15, 20, 24, 31, 35, 40, 44, 47]

def main():
    with open("models/benchmark_window_master.json", "r") as f:
        master = json.load(f)
        
    t_data = master["benchmark_12tasks_window"]
    p_data = master["benchmark_12poses_window"]
    
    # 1. 12-Pose Benchmark Table across seeds
    print("=" * 80)
    print("12-POSE BENCHMARK (pick-place-v3) UNDER WINDOW (ALL SEEDS)")
    print("=" * 80)
    
    pose_headers = ["Pose ID", "Kin s42", "Force s42", "Kin s43", "Force s43", "Kin s44", "Force s44"]
    pose_rows = []
    
    for idx in POSE_INDICES:
        row = [f"#{idx:02d}"]
        for m in ["Kin GRU s42", "Force GRU s42", "Kin GRU s43", "Force GRU s43", "Kin GRU s44", "Force GRU s44"]:
            succ = p_data[m]["pose_results"][str(idx)]["successes"]
            row.append(f"{succ}/5")
        pose_rows.append(row)
        
    summary_row = ["Poses (5/5)"]
    for m in ["Kin GRU s42", "Force GRU s42", "Kin GRU s43", "Force GRU s43", "Kin GRU s44", "Force GRU s44"]:
        summary_row.append(f"{p_data[m]['poses_solved_5of5']}/12")
    pose_rows.append(summary_row)
    
    rate_row = ["Rollouts (%)"]
    for m in ["Kin GRU s42", "Force GRU s42", "Kin GRU s43", "Force GRU s43", "Kin GRU s44", "Force GRU s44"]:
        rate_row.append(f"{p_data[m]['total_succ_rollouts']}/60 ({p_data[m]['pct_rollouts']:.1f}%)")
    pose_rows.append(rate_row)
    
    print(tabulate(pose_rows, headers=pose_headers, tablefmt="pipe"))
    print()
    
    # 2. 12-Task Benchmark Table across seeds
    print("=" * 80)
    print("12-TASK BENCHMARK (MT50 seed=42) UNDER WINDOW (ALL SEEDS)")
    print("=" * 80)
    
    task_headers = ["Task", "Regime", "Kin s42", "Force s42", "Kin s43", "Force s43", "Kin s44", "Force s44", "Mean Kin", "Mean Force", "Mean Δ"]
    task_rows = []
    
    kin_totals = []
    force_totals = []
    
    for t in TASK_NAMES:
        k42 = t_data["Kin GRU s42"]["task_results"][t]["success_count"]
        f42 = t_data["Force GRU s42"]["task_results"][t]["success_count"]
        k43 = t_data["Kin GRU s43"]["task_results"][t]["success_count"]
        f43 = t_data["Force GRU s43"]["task_results"][t]["success_count"]
        k44 = t_data["Kin GRU s44"]["task_results"][t]["success_count"]
        f44 = t_data["Force GRU s44"]["task_results"][t]["success_count"]
        
        m_kin = (k42 + k43 + k44) / 3.0
        m_force = (f42 + f43 + f44) / 3.0
        delta = m_force - m_kin
        
        task_rows.append([
            t, TASK_REGIMES[t],
            f"{k42}/20", f"{f42}/20",
            f"{k43}/20", f"{f43}/20",
            f"{k44}/20", f"{f44}/20",
            f"{m_kin:.1f}/20", f"{m_force:.1f}/20",
            f"{delta:+.1f}"
        ])
        
    tot_k42 = t_data["Kin GRU s42"]["total_succ"]
    tot_f42 = t_data["Force GRU s42"]["total_succ"]
    tot_k43 = t_data["Kin GRU s43"]["total_succ"]
    tot_f43 = t_data["Force GRU s43"]["total_succ"]
    tot_k44 = t_data["Kin GRU s44"]["total_succ"]
    tot_f44 = t_data["Force GRU s44"]["total_succ"]
    
    tot_m_kin = (tot_k42 + tot_k43 + tot_k44) / 3.0
    tot_m_force = (tot_f42 + tot_f43 + tot_f44) / 3.0
    tot_delta = tot_m_force - tot_m_kin
    
    task_rows.append([
        "TOTAL", "--",
        f"{tot_k42}/240", f"{tot_f42}/240",
        f"{tot_k43}/240", f"{tot_f43}/240",
        f"{tot_k44}/240", f"{tot_f44}/240",
        f"{tot_m_kin:.1f}/240", f"{tot_m_force:.1f}/240",
        f"{tot_delta:+.1f}"
    ])
    
    task_rows.append([
        "RATE (%)", "--",
        f"{tot_k42/2.4:.1f}%", f"{tot_f42/2.4:.1f}%",
        f"{tot_k43/2.4:.1f}%", f"{tot_f43/2.4:.1f}%",
        f"{tot_k44/2.4:.1f}%", f"{tot_f44/2.4:.1f}%",
        f"{tot_m_kin/2.4:.1f}%", f"{tot_m_force/2.4:.1f}%",
        f"{tot_delta/2.4:+.1f}%"
    ])
    
    print(tabulate(task_rows, headers=task_headers, tablefmt="pipe"))
    print()
    
    # 3. Print per-seed episode vectors
    print("=" * 80)
    print("PER-SEED 12-TASK EPISODE SUCCESS VECTORS")
    print("=" * 80)
    for m in ["Kin GRU s42", "Force GRU s42", "Kin GRU s43", "Force GRU s43", "Kin GRU s44", "Force GRU s44"]:
        vecs = {t: t_data[m]["task_results"][t]["ep_vector"] for t in TASK_NAMES}
        print(f"\nModel: {m}")
        for t in TASK_NAMES:
            print(f"  {t:<24}: {vecs[t]}")

if __name__ == "__main__":
    main()
