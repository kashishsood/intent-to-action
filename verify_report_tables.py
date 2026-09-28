#!/usr/bin/env python3
"""
verify_report_tables.py
-----------------------
Parses every results table in force_augmentation_study_report.md,
recomputes each cell from models/benchmark_window_master.json,
and exits nonzero on any mismatch.
"""

import json
import re
import sys

REPORT_PATH = "force_augmentation_study_report.md"
MASTER_JSON_PATH = "models/benchmark_window_master.json"

TASK_NAME_MAP = {
    "`pick-place-v3`": "pick-place-v3",
    "`pick-place-wall-v3`": "pick-place-wall-v3",
    "`peg-insert-side-v3`": "peg-insert-side-v3",
    "`assembly-v3`": "assembly-v3",
    "`hammer-v3`": "hammer-v3",
    "`sweep-into-v3`": "sweep-into-v3",
    "`reach-v3`": "reach-v3",
    "`door-open-v3`": "door-open-v3",
    "`drawer-open-v3`": "drawer-open-v3",
    "`button-press-topdown-v3`": "button-press-topdown-v3",
    "`drawer-close-v3`": "drawer-close-v3",
    "`door-close-v3`": "door-close-v3",
}

MODEL_TASK_COLS = [
    ("Kin s42", "Kin GRU s42"),
    ("Force s42", "Force GRU s42"),
    ("Kin s43", "Kin GRU s43"),
    ("Force s43", "Force GRU s43"),
    ("Kin s44", "Kin GRU s44"),
    ("Force s44", "Force GRU s44"),
]

def verify_12poses_table(report_text: str, master_data: dict) -> int:
    print("=" * 80)
    print("VERIFYING TABLE 3.1: 12-POSE PAIR BENCHMARK (pick-place-v3)")
    print("=" * 80)
    mismatches = 0
    p_data = master_data["benchmark_12poses_window"]

    # Locate section 3.1 table
    lines = report_text.splitlines()
    in_table = False
    table_rows = []
    for line in lines:
        if "### 3.1 Pose-by-Pose Results" in line:
            in_table = True
            continue
        if in_table:
            if line.strip().startswith("|") and "Pose ID" in line:
                continue
            if line.strip().startswith("| :---"):
                continue
            if line.strip().startswith("|") and ("**#" in line or "#" in line):
                table_rows.append(line.strip())
            elif line.strip().startswith("###") or (table_rows and not line.strip().startswith("|")):
                break

    print(f"Found {len(table_rows)} pose rows in Table 3.1.")
    if len(table_rows) != 12:
        print(f"[ERROR] Expected 12 pose rows, found {len(table_rows)}")
        return 1

    for row in table_rows:
        parts = [c.strip() for c in row.split("|")[1:-1]]
        pose_str = parts[0].replace("*", "").replace("#", "").strip()
        pose_id = int(pose_str)
        # Columns: Pose ID, Obj Pos, Goal Pos, Kin s42, Force s42, Kin s43, Force s43, Kin s44, Force s44
        col_vals = parts[3:9]

        for (col_name, model_key), val_str in zip(MODEL_TASK_COLS, col_vals):
            m = re.match(r"(\d+)/5", val_str)
            if not m:
                print(f"[ERROR] Cannot parse cell '{val_str}' in row #{pose_id:02d}, col {col_name}")
                mismatches += 1
                continue
            found_val = int(m.group(1))
            expected_val = p_data[model_key]["pose_results"][str(pose_id)]["successes"]
            if found_val != expected_val:
                print(f"[MISMATCH] Pose #{pose_id:02d} | Col {col_name:<9} | Expected: {expected_val}/5 | Found in report: {found_val}/5")
                mismatches += 1
            else:
                print(f"[OK] Pose #{pose_id:02d} | Col {col_name:<9} | {found_val}/5 == {expected_val}/5")

    return mismatches


def verify_12tasks_table(report_text: str, master_data: dict) -> int:
    print("\n" + "=" * 80)
    print("VERIFYING TABLE 4.1: 12-TASK MULTI-TASK BENCHMARK (MT50 seed=42)")
    print("=" * 80)
    mismatches = 0
    t_data = master_data["benchmark_12tasks_window"]

    lines = report_text.splitlines()
    in_table = False
    task_rows = {}
    for line in lines:
        if "### 4.1 Multi-Seed Results Table" in line:
            in_table = True
            continue
        if in_table:
            if line.strip().startswith("|") and ("Task Name" in line or line.strip().startswith("| :---")):
                continue
            if line.strip().startswith("|") and "`" in line:
                parts = [c.strip() for c in line.split("|")[1:-1]]
                raw_task = parts[0]
                if raw_task in TASK_NAME_MAP:
                    t_name = TASK_NAME_MAP[raw_task]
                    task_rows[t_name] = parts
            elif line.strip().startswith("###") or (len(task_rows) == 12 and not line.strip().startswith("|")):
                break

    print(f"Found {len(task_rows)} task rows in Table 4.1.")
    if len(task_rows) != 12:
        print(f"[ERROR] Expected 12 task rows, found {len(task_rows)}")
        return 1

    for t_name, parts in task_rows.items():
        # Columns: Task Name (0), Regime (1), Kin s42 (2), Force s42 (3), Kin s43 (4), Force s43 (5), Kin s44 (6), Force s44 (7), Mean Kin (8), Mean Force (9), Mean Delta (10)
        col_vals = parts[2:8]
        for (col_name, model_key), val_str in zip(MODEL_TASK_COLS, col_vals):
            m = re.match(r"(\d+)/20", val_str)
            if not m:
                print(f"[ERROR] Cannot parse cell '{val_str}' for task {t_name}, col {col_name}")
                mismatches += 1
                continue
            found_val = int(m.group(1))
            expected_val = t_data[model_key]["task_results"][t_name]["success_count"]
            if found_val != expected_val:
                print(f"[MISMATCH] Task {t_name:<24} | Col {col_name:<9} | Expected: {expected_val:2d}/20 | Found in report: {found_val:2d}/20")
                mismatches += 1
            else:
                print(f"[OK] Task {t_name:<24} | Col {col_name:<9} | {found_val:2d}/20 == {expected_val:2d}/20")

    return mismatches


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print(f"Reading report: {REPORT_PATH}")
    print(f"Reading master JSON: {MASTER_JSON_PATH}\n")

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report_text = f.read()

    with open(MASTER_JSON_PATH, "r", encoding="utf-8") as f:
        master_data = json.load(f)

    total_mismatches = 0
    total_mismatches += verify_12poses_table(report_text, master_data)
    total_mismatches += verify_12tasks_table(report_text, master_data)

    print("\n" + "=" * 80)
    if total_mismatches == 0:
        print("ALL REPORT CELLS VERIFIED: 100% BITWISE MATCH WITH MASTER JSON (0 MISMATCHES)")
        print("=" * 80)
        sys.exit(0)
    else:
        print(f"VERIFICATION FAILED: {total_mismatches} MISMATCHES FOUND")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
