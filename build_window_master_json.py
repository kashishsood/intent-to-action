#!/usr/bin/env python3
"""
build_window_master_json.py
Parses all 12-pose and 12-task logs and existing json outputs into a single consolidated master JSON file:
models/benchmark_window_master.json
"""

import os
import re
import json
from collections import OrderedDict

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

def parse_12tasks_log(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # regex for episode lines
    # [TASK_EP] Ep 001/240 | Task: pick-place-v3            | EpSeed    42 | Success: False | Steps: 500
    ep_pattern = re.compile(r"\[TASK_EP\] Ep (\d+)/240 \| Task: ([\w\-]+)\s+\| EpSeed\s+(\d+) \| Success: (True|False)")
    
    tasks_data = {t: {"success_count": 0, "episodes": 0, "ep_vector": [], "regime": TASK_REGIMES[t]} for t in TASK_NAMES}
    
    for m in ep_pattern.finditer(text):
        ep_num = int(m.group(1))
        t_name = m.group(2)
        seed = int(m.group(3))
        succ = (m.group(4) == "True")
        
        if t_name in tasks_data:
            tasks_data[t_name]["episodes"] += 1
            if succ:
                tasks_data[t_name]["success_count"] += 1
            tasks_data[t_name]["ep_vector"].append(1 if succ else 0)
            
    for t in TASK_NAMES:
        eps = tasks_data[t]["episodes"]
        sc = tasks_data[t]["success_count"]
        tasks_data[t]["success_rate"] = (sc / eps * 100.0) if eps > 0 else 0.0
        
    total_succ = sum(tasks_data[t]["success_count"] for t in TASK_NAMES)
    total_eps = sum(tasks_data[t]["episodes"] for t in TASK_NAMES)
    total_rate = (total_succ / total_eps * 100.0) if total_eps > 0 else 0.0
    
    return {
        "task_results": tasks_data,
        "total_succ": total_succ,
        "total_episodes": total_eps,
        "total_rate": total_rate,
    }

def parse_12poses_log(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    ep_pattern = re.compile(r"\[(?:POSE_EP|EP_LOG)\] .*?Pose #(\d+).*?Seed\s+(\d+).*?Success: (True|False)")
    
    pose_results = {p: {"successes": 0, "repeats": 0, "succ_vector": []} for p in POSE_INDICES}
    
    for m in ep_pattern.finditer(text):
        p_idx = int(m.group(1))
        seed = int(m.group(2))
        succ = (m.group(3) == "True")
        
        if p_idx in pose_results:
            pose_results[p_idx]["repeats"] += 1
            if succ:
                pose_results[p_idx]["successes"] += 1
            pose_results[p_idx]["succ_vector"].append(1 if succ else 0)
            
    counts_vector = [pose_results[p]["successes"] for p in POSE_INDICES]
    binary_vector = [1 if pose_results[p]["successes"] == 5 else 0 for p in POSE_INDICES]
    total_succ = sum(counts_vector)
    
    return {
        "pose_results": pose_results,
        "counts_vector": counts_vector,
        "binary_vector": binary_vector,
        "total_succ_rollouts": total_succ,
        "total_rollouts": 60,
        "pct_rollouts": (total_succ / 60.0) * 100.0,
        "poses_solved_5of5": sum(binary_vector),
    }

def main():
    master = {
        "benchmark_12tasks_window": {},
        "benchmark_12poses_window": {},
    }
    
    # 1. 12-task window evaluations
    task_models = [
        ("Kin GRU s42", "logs/eval_12tasks_gru_kinematic_s42_window.log"),
        ("Force GRU s42", "logs/eval_12tasks_gru_force_s42_window.log"),
        ("Kin GRU s43", "logs/eval_12tasks_gru_kinematic_s43_window.log"),
        ("Force GRU s43", "logs/eval_12tasks_gru_force_s43_window.log"),
        ("Kin GRU s44", "logs/eval_12tasks_gru_kinematic_s44_window.log"),
        ("Force GRU s44", "logs/eval_12tasks_gru_force_s44_window.log"),
    ]
    
    for name, log_p in task_models:
        if os.path.exists(log_p):
            master["benchmark_12tasks_window"][name] = parse_12tasks_log(log_p)
            print(f"Parsed 12-tasks: {name} -> {master['benchmark_12tasks_window'][name]['total_succ']}/240")
        else:
            print(f"Missing log: {log_p}")
            
    # 2. 12-pose window evaluations
    pose_models = [
        ("Kin GRU s42", "logs/eval_12poses_gru_kinematic_s42_window.log"),
        ("Force GRU s42", "logs/eval_12poses_gru_force_s42_window.log"),
        ("Kin GRU s43", "logs/eval_12poses_gru_kinematic_s43_window.log"),
        ("Force GRU s43", "logs/eval_12poses_gru_force_s43_window.log"),
        ("Kin GRU s44", "logs/eval_12poses_gru_kinematic_s44_window.log"),
        ("Force GRU s44", "logs/eval_12poses_gru_force_s44_window.log"),
    ]
    
    for name, log_p in pose_models:
        if os.path.exists(log_p):
            master["benchmark_12poses_window"][name] = parse_12poses_log(log_p)
            print(f"Parsed 12-poses: {name} -> {master['benchmark_12poses_window'][name]['poses_solved_5of5']}/12 poses ({master['benchmark_12poses_window'][name]['counts_vector']})")
        else:
            print(f"Missing log: {log_p}")
            
    out_path = "models/benchmark_window_master.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)
    print(f"\nSaved consolidated master JSON to {out_path}")

if __name__ == "__main__":
    main()
