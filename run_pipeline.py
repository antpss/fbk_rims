#!/usr/bin/env python3
"""
================================================================================
RIMS++ MASTER PIPELINE ORCHESTRATOR
File: run_pipeline.py

Executes the entire RIMS++ process mining, training, and simulation pipeline
end-to-end, with configuration driven entirely by config.yaml.

Pipeline Stages:
  [Step 00]  00_filter_log.py            (Standardize columns, filter prefixes, map lifecycles)
  [Step 01]  01_process_discovery.py     (Mine Petri Net via Split Miner)
  [Step 01b] 01b_discover_roles.py       (Mine Organizational Roles & Staffing Capacities)
  [Step 02]  02_align_and_repair_log.py  (Align traces to net, repair log lifecycle)
  [Step 02b] 02b_discover_delays.py      (Mine Extraneous Delays & Refractory Latencies)
  [Step 03]  03_dataset_builder.py       (Feature Engineering & Context Extraction)
  [Step 04]  04_train.py                 (Train Duration & Waiting Time Models)
  [Step 04b] 04b_benchmark_models.py     (Benchmark Candidates & Crown Hybrid Champions)
  [Step 05]  05_train_routing.py         (Train White-Box XOR Decision Tree Classifiers)
  [Step 06]  06_simulate.py              (Execute SimPy Discrete-Event Simulator & Benchmark)

Usage:
  python run_pipeline.py                         # Run full pipeline from config.yaml
  python run_pipeline.py --config custom.yaml    # Run with custom config file
  python run_pipeline.py --steps 3,4,4b,5,6      # Run only ML and Simulation stages
  python run_pipeline.py --skip-discovery        # Skip Split Miner and Alignment stages
  python run_pipeline.py --skip-train            # Skip training, run simulation with existing models
  python run_pipeline.py --cases 100             # Override simulation case count
================================================================================
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from config_loader import load_config


def run_stage(title, cmd, cwd=PROJECT_ROOT):
    """Executes a pipeline stage subprocess and streams output to terminal."""
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    start_t = time.time()
    res = subprocess.run(cmd, cwd=cwd)
    dur = time.time() - start_t

    if res.returncode != 0:
        print(f"\n[!] ERROR: Stage '{title}' failed with exit code {res.returncode}.")
        print("    Aborting pipeline execution.")
        sys.exit(res.returncode)

    print(f"\n[✓] {title} completed successfully in {dur:.2f} seconds.")
    return dur


def main():
    parser = argparse.ArgumentParser(description="RIMS++ Master Pipeline Runner")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--steps", type=str, default="all", help="Comma-separated steps to run (e.g. '0,1,1b,2,2b,3,4,4b,5,6' or 'all')")
    parser.add_argument("--start-from", type=str, default=None, help="Start pipeline from specific step (e.g. '03' or '04')")
    parser.add_argument("--skip-discovery", action="store_true", help="Skip stages 01, 01b, 02, 02b (use existing Petri net and roles)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training stages 04, 04b, 05 (use existing models)")
    parser.add_argument("--cases", type=int, default=None, help="Override simulation case count in Step 06")
    parser.add_argument("--dry-run", action="store_true", help="Print planned stages without executing")
    args = parser.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(PROJECT_ROOT, args.config)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    cfg = load_config(cfg_path)
    py = sys.executable

    print("\n" + "#" * 80)
    print("       RIMS++ MODERN BUSINESS PROCESS MINING & SIMULATION PIPELINE")
    print(f"       Project: {cfg.get('project', {}).get('name', 'Process_Simulation')}")
    print(f"       Config:  {cfg_path}")
    print("#" * 80)

    # Define stage sequence
    stages = [
        ("00", "Step 00: Preprocessing & Log Standardization", [py, "src/00_filter_log.py", "--config", cfg_path]),
        ("01", "Step 01: Process Discovery (Split Miner)", [py, "src/01_process_discovery.py", "--config", cfg_path]),
        ("01b", "Step 01b: Organizational Role Discovery", [py, "src/01b_discover_roles.py", "--config", cfg_path]),
        ("02", "Step 02: Conformance Alignment & Log Repair", [py, "src/02_align_and_repair_log.py", "--config", cfg_path]),
        ("02b", "Step 02b: Causal Delays & Worker Latency Mining", [py, "src/02b_discover_delays.py", "--config", cfg_path]),
        ("03", "Step 03: Feature Engineering & Dataset Builder (Duration)", [py, "src/03_dataset_builder.py", "--task", "duration", "--config", cfg_path]),
        ("03_wait", "Step 03: Feature Engineering & Dataset Builder (Waiting Time)", [py, "src/03_dataset_builder.py", "--task", "waiting_time", "--config", cfg_path]),
        ("04b", "Step 04b: Automated Benchmarking & Champion Synthesis (Duration)", [py, "src/04b_benchmark_models.py", "--task", "duration", "--config", cfg_path, "--run_all"]),
        ("04b_wait", "Step 04b: Automated Benchmarking & Champion Synthesis (Waiting Time)", [py, "src/04b_benchmark_models.py", "--task", "waiting_time", "--config", cfg_path, "--run_all"]),
        ("05", "Step 05: White-Box XOR Decision Mining", [py, "src/05_train_routing.py", "--config", cfg_path]),
        ("06", "Step 06: Discrete-Event Simulator & Benchmark", [py, "src/06_simulate.py", "--config", cfg_path] + (["--cases", str(args.cases)] if args.cases else [])),
    ]

    # Filter stages according to arguments
    selected_steps = None
    if args.steps.lower() != "all":
        selected_steps = set(s.strip().lower().lstrip("0") or "0" for s in args.steps.split(","))
        if "4" in selected_steps:
            selected_steps.add("4b")

    start_idx = 0
    if args.start_from:
        target = args.start_from.strip().lower().lstrip("0") or "0"
        if target == "4":
            target = "4b"
        for i, (sid, _, _) in enumerate(stages):
            clean_sid = sid.split("_")[0].lstrip("0") or "0"
            if clean_sid == target:
                start_idx = i
                break

    planned = []
    for i, (sid, title, cmd) in enumerate(stages):
        if i < start_idx:
            continue
        base_step = sid.split("_")[0]
        clean_base = base_step.lstrip("0") or "0"
        clean_sid = sid.lstrip("0") or "0"

        if selected_steps and clean_base not in selected_steps and clean_sid not in selected_steps:
            continue
        if args.skip_discovery and base_step in ["01", "01b", "02", "02b"]:
            continue
        if args.skip_train and base_step in ["04", "04b", "05"]:
            continue

        planned.append((sid, title, cmd))

    print(f"\nPlanned Pipeline Stages ({len(planned)} steps):")
    for sid, title, _ in planned:
        print(f"  [{sid:>6}] {title}")

    if args.dry_run:
        print("\nDry-run mode active. No commands were executed.")
        return

    total_start = time.time()
    stage_durations = []

    for sid, title, cmd in planned:
        d = run_stage(title, cmd)
        stage_durations.append((title, d))

    total_time = time.time() - total_start

    print("\n" + "=" * 80)
    print("               RIMS++ PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    for title, dur in stage_durations:
        print(f"  {title:<60} | {dur:>6.2f}s")
    print("-" * 80)
    print(f"  {'TOTAL PIPELINE RUNTIME':<60} | {total_time:>6.2f}s ({(total_time/60.0):.1f} min)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
