#!/usr/bin/env python3
"""
run_batch_evals.py
Runs 12-task MT50(seed=42) benchmark under method=window for Kin GRU and Force GRU seeds 43 and 44.
Executes seeds 43 in parallel, then seeds 44 in parallel.
Pipes raw stdout directly to logs/.
"""

import subprocess
import sys
import time

tasks = [
    # (name, ckpt, log_file, json_file)
    ("Kin GRU s43", "models/bc_12tasks_gru_kinematic_s43.pt", "logs/eval_12tasks_gru_kinematic_s43_window.log", "models/res_kin_s43.json"),
    ("Force GRU s43", "models/bc_12tasks_gru_force_s43.pt", "logs/eval_12tasks_gru_force_s43_window.log", "models/res_force_s43.json"),
    ("Kin GRU s44", "models/bc_12tasks_gru_kinematic_s44.pt", "logs/eval_12tasks_gru_kinematic_s44_window.log", "models/res_kin_s44.json"),
    ("Force GRU s44", "models/bc_12tasks_gru_force_s44.pt", "logs/eval_12tasks_gru_force_s44_window.log", "models/res_force_s44.json"),
]

def run_pair(task1, task2):
    print(f"Starting pair: {task1[0]} and {task2[0]}...", flush=True)
    t0 = time.time()
    
    cmd1 = [sys.executable, "-u", "eval_study_window.py", "--ckpt", task1[1], "--benchmark", "12tasks", "--method", "window", "--out-json", task1[3]]
    cmd2 = [sys.executable, "-u", "eval_study_window.py", "--ckpt", task2[1], "--benchmark", "12tasks", "--method", "window", "--out-json", task2[3]]
    
    with open(task1[2], "w", encoding="utf-8") as f1, open(task2[2], "w", encoding="utf-8") as f2:
        p1 = subprocess.Popen(cmd1, stdout=f1, stderr=subprocess.STDOUT)
        p2 = subprocess.Popen(cmd2, stdout=f2, stderr=subprocess.STDOUT)
        
        while p1.poll() is None or p2.poll() is None:
            time.sleep(5)
            elapsed = time.time() - t0
            print(f"  [RUNNING] Elapsed: {elapsed:.0f}s | {task1[0]}: {'DONE' if p1.poll() is not None else 'ACTIVE'} | {task2[0]}: {'DONE' if p2.poll() is not None else 'ACTIVE'}", flush=True)
            
    print(f"Pair complete in {time.time() - t0:.1f}s!\n", flush=True)

if __name__ == "__main__":
    # Seed 43 pair
    run_pair(tasks[0], tasks[1])
    # Seed 44 pair
    run_pair(tasks[2], tasks[3])
    print("ALL EVALUATIONS COMPLETE!", flush=True)
