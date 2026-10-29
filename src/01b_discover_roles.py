#!/usr/bin/env python3
"""
src/01b_discover_roles.py

Organizational Role Discovery using Professor Alessandro Sperduti's Algorithm
(Burattin, Sperduti, & Veluscek, 2013: 'Business models enhancement through discovery of roles').

Discovers:
  1. Activity-to-Role partition (each activity belongs to exactly one role).
  2. Role-to-Workers rosters (originator multisets per role).
  3. Realistic Department Staffing Capacities (observed concurrent capacity).
  4. Worker Concurrency Quotas (C_w >= 1 to allow natural task interleaving).

Outputs:
  - models/role_mapping.json
  - models/role_discovery_report.md
"""

import os
import sys
import json
import argparse
from collections import defaultdict, Counter
from datetime import datetime

import pandas as pd
import numpy as np
import pm4py
from pm4py.algo.organizational_mining.roles import algorithm as roles_miner

try:
    from config_loader import load_config, PROJECT_ROOT
except ImportError:
    from src.config_loader import load_config, PROJECT_ROOT


def _generate_descriptive_role_name(idx: int, activities: list) -> str:
    """Generates an intuitive business department name based on activity semantics."""
    acts_lower = " ".join(activities).lower()
    if "fraud" in acts_lower:
        base = "Fraud_Squad"
    elif "validate" in acts_lower:
        base = "Validation_Underwriting"
    elif any(k in acts_lower for k in ["complete", "offer", "lead", "file"]):
        base = "FrontOffice_Processing"
    elif "contract" in acts_lower:
        base = "Contract_Management"
    else:
        # Fallback to dominant activity name
        clean_act = activities[0].replace("W_", "").replace(" ", "_")
        base = clean_act[:20]
    return f"Role_{idx}_{base}"


def discover_roles(cfg=None, threshold=None, force=False):
    cfg = cfg or load_config()
    res_cfg = cfg.get("resources", {})
    disc_cfg = res_cfg.get("discovery", {})

    th = threshold if threshold is not None else float(disc_cfg.get("threshold", 0.35))
    models_dir = os.path.join(PROJECT_ROOT, cfg.get("paths", {}).get("models_dir", "models"))
    output_file = os.path.join(PROJECT_ROOT, disc_cfg.get("output_file", os.path.join(models_dir, "role_mapping.json")))
    report_file = os.path.join(PROJECT_ROOT, disc_cfg.get("report_file", os.path.join(models_dir, "role_discovery_report.md")))

    aligned_log_rel = cfg.get("paths", {}).get("aligned_log", "data/processed/aligned_BPI_2012.xes")
    log_path = os.path.join(PROJECT_ROOT, aligned_log_rel)
    if not os.path.exists(log_path):
        log_path = os.path.join(PROJECT_ROOT, cfg.get("paths", {}).get("raw_log", "data/processed/BPI_2012_filtered.xes"))

    print(f"Loading event log from: {log_path}...")
    df = pm4py.read_xes(log_path)
    if not isinstance(df, pd.DataFrame):
        df = pm4py.convert_to_dataframe(df)

    # Filter out synthetic/system resources
    df = df[df["org:resource"].notna()].copy()
    df["org:resource"] = df["org:resource"].astype(str)
    df = df[~df["org:resource"].str.upper().isin(["SYNTHETIC", "SYSTEM", "NONE", "NAN"])].copy()

    # Target activities
    target_activities = cfg.get("target_activities", [])
    if target_activities:
        df_target = df[df["concept:name"].isin(target_activities)].copy()
    else:
        proc_prefixes = tuple(cfg.get("preprocessing", {}).get("process_activity_prefixes", []))
        if proc_prefixes:
            df_target = df[df["concept:name"].str.startswith(proc_prefixes)].copy()
        else:
            ms_prefixes = tuple(cfg.get("preprocessing", {}).get("milestone_prefixes", []))
            if ms_prefixes:
                df_target = df[~df["concept:name"].str.startswith(ms_prefixes)].copy()
            else:
                df_target = df.copy()
        target_activities = sorted(df_target["concept:name"].unique().tolist())

    print(f"Running Sperduti Role Discovery (Burattin et al. 2013) on {len(target_activities)} activities with threshold θ={th}...")
    roles_discovered = roles_miner.apply(df_target, parameters={"roles_threshold_parameter": th})

    # Sort roles by descending frequency of activities / originators
    roles_discovered = sorted(roles_discovered, key=lambda r: len(r.originator_importance), reverse=True)

    activity_to_role = {}
    role_to_activities = {}
    role_to_workers = {}
    role_originator_importance = {}
    all_workers_set = set()

    for idx, r in enumerate(roles_discovered):
        role_name = _generate_descriptive_role_name(idx, r.activities)
        role_to_activities[role_name] = sorted(r.activities)
        for act in r.activities:
            activity_to_role[act] = role_name

        workers_in_role = sorted(list(r.originator_importance.keys()))
        role_to_workers[role_name] = workers_in_role
        role_originator_importance[role_name] = {k: int(v) for k, v in r.originator_importance.items()}
        all_workers_set.update(workers_in_role)

    all_resources = sorted(list(all_workers_set))

    # Compute observed concurrency & capacities from the log
    print("Computing empirical department staffing capacities and worker concurrency quotas...")
    df_sorted = df_target.sort_values("time:timestamp").copy()

    # Worker concurrency: 95th percentile concurrent open tasks
    worker_open_tasks = defaultdict(int)
    worker_max_concurrency = defaultdict(int)
    worker_concurrency_history = defaultdict(list)

    # Role concurrency: active cases in each department over time
    role_open_tasks = defaultdict(int)
    role_max_concurrency = defaultdict(int)
    role_concurrency_history = defaultdict(list)

    events_list = []
    needed_cols = [c for c in ["concept:name", "org:resource", "lifecycle:transition", "time:timestamp"] if c in df_sorted.columns]
    for row in df_sorted[needed_cols].to_dict("records"):
        act = row["concept:name"]
        w = row["org:resource"]
        trans = str(row.get("lifecycle:transition", "COMPLETE")).upper()
        ts = row["time:timestamp"]
        role = activity_to_role.get(act)
        events_list.append((ts, trans, w, act, role))

    # Process events chronologically
    events_list.sort(key=lambda x: (x[0], 0 if x[1] == "COMPLETE" else 1))

    for ts, trans, w, act, role in events_list:
        if trans == "START":
            worker_open_tasks[w] += 1
            worker_concurrency_history[w].append(worker_open_tasks[w])
            if worker_open_tasks[w] > worker_max_concurrency[w]:
                worker_max_concurrency[w] = worker_open_tasks[w]

            if role:
                role_open_tasks[role] += 1
                role_concurrency_history[role].append(role_open_tasks[role])
                if role_open_tasks[role] > role_max_concurrency[role]:
                    role_max_concurrency[role] = role_open_tasks[role]

        elif trans == "COMPLETE":
            worker_open_tasks[w] = max(0, worker_open_tasks[w] - 1)
            if role:
                role_open_tasks[role] = max(0, role_open_tasks[role] - 1)

    # 95th percentile worker concurrency (minimum 1, default 2 for active desk staff)
    worker_concurrencies = {}
    for w in all_resources:
        hist = worker_concurrency_history.get(w, [1])
        p95 = int(np.percentile(hist, 95)) if hist else 1
        worker_concurrencies[w] = max(1, p95)

    # Role staffing capacities (95th percentile concurrent workload in department)
    role_capacities = {}
    for r_name in role_to_activities:
        hist = role_concurrency_history.get(r_name, [1])
        p95 = int(np.percentile(hist, 95)) if hist else 1
        # Capacity is bounded between p95 concurrency and total eligible workers in role
        pool_size = len(role_to_workers[r_name])
        cap = max(1, min(p95, pool_size))
        role_capacities[r_name] = cap

    # Construct final manifest dictionary
    role_mapping_data = {
        "__metadata__": {
            "algorithm": "Burattin-Sperduti-Veluscek (2013)",
            "threshold": th,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_distinct_resources": len(all_resources),
            "total_roles": len(role_to_activities)
        },
        "__role_names__": list(role_to_activities.keys()),
        "roles": role_to_activities,
        "activity_to_role": activity_to_role,
        "role_to_workers": role_to_workers,
        "role_capacities": role_capacities,
        "worker_concurrencies": worker_concurrencies,
        "__all_resources__": all_resources,
        "__role_originator_importance__": role_originator_importance
    }

    # Also preserve backward-compatibility keys: map each activity to its worker pool
    for act, r_name in activity_to_role.items():
        role_mapping_data[act] = role_to_workers[r_name]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(role_mapping_data, f, indent=2)
    print(f"Successfully generated role manifest: {output_file}")

    # Generate Markdown Report
    _write_markdown_report(report_file, th, role_to_activities, role_to_workers, role_capacities, worker_concurrencies, all_resources)
    print(f"Successfully generated discovery report: {report_file}")

    return role_mapping_data


def _write_markdown_report(report_path, threshold, role_to_activities, role_to_workers, role_capacities, worker_concurrencies, all_resources):
    lines = [
        "# Organizational Role Discovery Report",
        "",
        f"**Algorithm**: Burattin, Sperduti, & Veluscek (2013) — *Business models enhancement through discovery of roles*",
        f"**Aggregation Threshold (θ)**: `{threshold}`",
        f"**Total Discovered Roles**: `{len(role_to_activities)}`",
        f"**Total Distinct Workers**: `{len(all_resources)}`",
        "",
        "---",
        "",
        "## Discovered Organizational Departments",
        "",
        "| Role / Department | Activities Assigned | Qualified Staff Count | Effective Concurrency Capacity |",
        "| :--- | :--- | :---: | :---: |"
    ]

    for r_name, acts in role_to_activities.items():
        acts_str = "<br/>".join([f"`{a}`" for a in acts])
        staff_cnt = len(role_to_workers[r_name])
        cap = role_capacities.get(r_name, 1)
        lines.append(f"| **{r_name}** | {acts_str} | {staff_cnt} workers | **{cap}** |")

    lines.extend([
        "",
        "---",
        "",
        "## Multi-Skilled Human Staffing Analysis",
        "",
        "In this hybrid model, each enterprise worker is tracked as an individual biological entity with an empirical concurrency quota ($C_w$), eliminating worker ubiquity across departments while allowing natural task interleaving.",
        "",
        f"- **Median Worker Concurrency Quota ($C_w$)**: `{int(np.median(list(worker_concurrencies.values())))}` simultaneous open cases",
        f"- **Max Worker Concurrency Quota**: `{max(worker_concurrencies.values())}` simultaneous open cases",
        "",
        "### Top Staff Roster & Concurrency Quotas",
        "",
        "| Worker ID | Concurrency Quota ($C_w$) | Department Memberships |",
        "| :--- | :---: | :--- |"
    ])

    top_workers = sorted(all_resources, key=lambda w: worker_concurrencies.get(w, 1), reverse=True)[:15]
    for w in top_workers:
        memberships = [r for r, wlist in role_to_workers.items() if w in wlist]
        m_str = ", ".join(memberships)
        lines.append(f"| `{w}` | **{worker_concurrencies.get(w, 1)}** | {m_str} |")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover organizational roles using Sperduti algorithm.")
    parser.add_argument("--threshold", "-t", type=float, default=None, help="Sperduti similarity threshold (default: 0.35)")
    args = parser.parse_args()

    discover_roles(threshold=args.threshold)

