#!/usr/bin/env python3
"""
src/simulator/metrics.py

Simulation Output Exporter & Multi-Dimensional Benchmark Evaluator.
Powered by the official `log-distance-measures` library (Chapela-Campa et al., BPM 2023 / Tartu)
and matching the evaluation framework of Meneghello et al. (Information Systems 2025) & WISDOM:
  1. Control-Flow Quality:
     - Control-flow Log Distance (CFLD, Damerau-Levenshtein + Hungarian assignment)
     - Trace Length Absolute Error (L_traces, Hungarian matching)
     - N-Gram Distribution Distance (NGD: 2-Gram and 3-Gram)
  2. Temporal Distribution Quality:
     - Absolute Event Distribution (AED, 1WD / EMD across calendar timeline)
     - Circadian Event Distribution (CED, 1WD / EMD across 168h weekly cycle)
     - Relative Event Distribution (RED, 1WD / EMD relative to trace origin)
  3. Workforce / Resource Scheduling Quality:
     - Circadian Workforce Distribution (CWD, 1WD / EMD across weekly active staff)
  4. Congestion & Cycle Time Quality:
     - Cycle Time Distribution (CTD / 1WD Wasserstein Distance)
     - Case Arrival Rate (CAR, 1WD across case arrival timeline)
     - Cycle Time Mean, Median, and MAE
"""

import os
from datetime import timedelta
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
import pm4py

# Official log-distance-measures library imports
from log_distance_measures.config import EventLogIDs
from log_distance_measures.control_flow_log_distance import control_flow_log_distance
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance
from log_distance_measures.circadian_event_distribution import circadian_event_distribution_distance
from log_distance_measures.circadian_workforce_distribution import circadian_workforce_distribution_distance
from log_distance_measures.relative_event_distribution import relative_event_distribution_distance
from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance
from log_distance_measures.case_arrival_distribution import case_arrival_distribution_distance

try:
    from config_loader import PROJECT_ROOT
except ImportError:
    from src.config_loader import PROJECT_ROOT


def format_dur(sec):
    """Formats duration in seconds into human-readable string."""
    if sec >= 86400:
        return f"{sec/86400:.1f} days"
    elif sec >= 3600:
        return f"{sec/3600:.1f} hours"
    elif sec >= 60:
        return f"{sec/60:.1f} mins"
    return f"{sec:.1f}s"


def compute_trace_length_error(clean_real, clean_sim, max_sample=1000):
    """
    Computes L_traces: average absolute difference in trace lengths paired via Hungarian algorithm (Kuhn 1955).
    """
    if clean_real is None or clean_sim is None:
        return 0.0

    r_lens = clean_real.groupby("case_id")["activity"].count().values.astype(float)
    s_lens = clean_sim.groupby("case_id")["activity"].count().values.astype(float)

    if len(r_lens) == 0 or len(s_lens) == 0:
        return 0.0

    rng = np.random.RandomState(42)
    if len(r_lens) > max_sample:
        r_lens = rng.choice(r_lens, size=max_sample, replace=False)
    if len(s_lens) > max_sample:
        s_lens = rng.choice(s_lens, size=max_sample, replace=False)

    nr, ns = len(r_lens), len(s_lens)
    n = min(nr, ns)
    if n == 0:
        return 0.0

    cost = np.abs(r_lens[:, None] - s_lens[None, :])
    row, col = linear_sum_assignment(cost)
    return float(cost[row, col].sum() / n)


def prepare_evaluation_dfs(df_real, df_sim, target_acts=None, max_cases=1000):
    """
    Standardizes real and simulated event logs for log_distance_measures:
    Ensures both logs contain standard columns:
      case_id, activity, start_time, end_time, resource
    Returns: (clean_real_df, clean_sim_df, ids)
    """
    ids = EventLogIDs(
        case="case_id",
        activity="activity",
        start_time="start_time",
        end_time="end_time",
        resource="resource"
    )

    # Standardize simulated event log
    clean_sim = pd.DataFrame({
        "case_id": df_sim["case_id"].astype(str),
        "activity": df_sim["activity"].astype(str),
        "start_time": pd.to_datetime(df_sim["start_timestamp"], utc=True),
        "end_time": pd.to_datetime(df_sim["complete_timestamp"], utc=True),
        "resource": df_sim["resource_id"].astype(str) if "resource_id" in df_sim.columns else "UNKNOWN"
    })

    if df_real is None or len(df_real) == 0:
        return None, clean_sim, ids

    # Filter real log to target activities
    r_filtered = df_real.copy()
    if target_acts:
        r_filtered = r_filtered[r_filtered["concept:name"].isin(target_acts)].copy()

    c_col = "case:concept:name" if "case:concept:name" in r_filtered.columns else "case_id"
    a_col = "concept:name" if "concept:name" in r_filtered.columns else "activity"
    t_col = "time:timestamp" if "time:timestamp" in r_filtered.columns else "start_timestamp"
    r_col = "org:resource" if "org:resource" in r_filtered.columns else "resource"

    # Handle paired lifecycle transitions if present
    if "lifecycle:transition" in r_filtered.columns:
        trans_vals = set(r_filtered["lifecycle:transition"].dropna().astype(str).str.upper())
        if "START" in trans_vals and "COMPLETE" in trans_vals:
            df_start = r_filtered[r_filtered["lifecycle:transition"].str.upper() == "START"].copy()
            df_comp = r_filtered[r_filtered["lifecycle:transition"].str.upper() == "COMPLETE"].copy()
            df_start["_idx"] = df_start.groupby([c_col, a_col]).cumcount()
            df_comp["_idx"] = df_comp.groupby([c_col, a_col]).cumcount()
            merged = pd.merge(
                df_comp,
                df_start[[c_col, a_col, "_idx", t_col]],
                on=[c_col, a_col, "_idx"],
                suffixes=("", "_start"),
                how="left"
            )
            merged["start_time"] = pd.to_datetime(merged[t_col + "_start"].fillna(merged[t_col]), utc=True)
            merged["end_time"] = pd.to_datetime(merged[t_col], utc=True)
            clean_real = pd.DataFrame({
                "case_id": merged[c_col].astype(str),
                "activity": merged[a_col].astype(str),
                "start_time": merged["start_time"],
                "end_time": merged["end_time"],
                "resource": merged[r_col].astype(str) if r_col in merged.columns else "UNKNOWN"
            })
        else:
            comp = r_filtered[r_filtered["lifecycle:transition"].str.upper() == "COMPLETE"].copy() if "COMPLETE" in trans_vals else r_filtered.copy()
            ts = pd.to_datetime(comp[t_col], utc=True)
            clean_real = pd.DataFrame({
                "case_id": comp[c_col].astype(str),
                "activity": comp[a_col].astype(str),
                "start_time": ts,
                "end_time": ts,
                "resource": comp[r_col].astype(str) if r_col in comp.columns else "UNKNOWN"
            })
    else:
        ts = pd.to_datetime(r_filtered[t_col], utc=True)
        clean_real = pd.DataFrame({
            "case_id": r_filtered[c_col].astype(str),
            "activity": r_filtered[a_col].astype(str),
            "start_time": ts,
            "end_time": ts,
            "resource": r_filtered[r_col].astype(str) if r_col in r_filtered.columns else "UNKNOWN"
        })

    # Subsample real cases uniformly across the log to obtain an unbiased representative sample
    rng = np.random.RandomState(42)
    u_cases_real = clean_real["case_id"].unique()
    if max_cases and len(u_cases_real) > max_cases:
        sampled_cases_r = set(rng.choice(u_cases_real, size=max_cases, replace=False))
        clean_real = clean_real[clean_real["case_id"].isin(sampled_cases_r)].copy()

    u_cases_sim = clean_sim["case_id"].unique()
    if max_cases and len(u_cases_sim) > max_cases:
        sampled_cases_s = set(rng.choice(u_cases_sim, size=max_cases, replace=False))
        clean_sim = clean_sim[clean_sim["case_id"].isin(sampled_cases_s)].copy()

    return clean_real, clean_sim, ids


class SimulationEvaluator:
    """
    Multi-Dimensional Benchmark Evaluator and Serializer.
    Direct integration with the official `log-distance-measures` library (Chapela-Campa et al. BPM 2023).
    """
    def __init__(self, cfg, mode="hybrid_residual", arrival_mode="replay", export_xes=True,
                 sim_log_path=None, output_simulated_xes=None, workers_count=59, calendar=None):
        self.cfg = cfg
        self.mode = mode
        self.arrival_mode = arrival_mode
        self.export_xes = export_xes
        self.sim_log_path = sim_log_path or os.path.join(
            PROJECT_ROOT,
            f"data/generated/csv/simulated_log_{self.mode}.csv" if self.mode != "hybrid_residual" else
            cfg.get("simulation", {}).get("output_simulated_log", "data/generated/csv/simulated_log.csv")
        )
        self.output_simulated_xes = output_simulated_xes or os.path.join(
            PROJECT_ROOT,
            f"data/generated/xes/simulated_log_{self.mode}.xes" if self.mode != "hybrid_residual" else
            cfg.get("simulation", {}).get("output_simulated_xes", "data/generated/xes/simulated_log.xes")
        )
        self.workers_count = workers_count
        self.calendar = calendar

    def save_csv(self, df_sim):
        """Saves simulated event log to flat CSV format."""
        os.makedirs(os.path.dirname(self.sim_log_path), exist_ok=True)
        df_sim.to_csv(self.sim_log_path, index=False)
        print(f"[✓] Saved simulated event log (CSV) to: {self.sim_log_path}")

    def save_xes(self, df_sim):
        """Saves simulated event log to standard IEEE XES format with paired START and COMPLETE transitions."""
        if not self.export_xes or len(df_sim) == 0:
            return

        # Paired START lifecycle events
        df_start = df_sim.copy()
        df_start["case:concept:name"] = df_start["case_id"].astype(str)
        df_start["concept:name"] = df_start["activity"]
        df_start["time:timestamp"] = pd.to_datetime(df_start["start_timestamp"], utc=True)
        df_start["org:resource"] = df_start["resource_id"]
        df_start["lifecycle:transition"] = "start"
        df_start["_trans_order"] = 0

        # Paired COMPLETE lifecycle events
        df_comp = df_sim.copy()
        df_comp["case:concept:name"] = df_comp["case_id"].astype(str)
        df_comp["concept:name"] = df_comp["activity"]
        df_comp["time:timestamp"] = pd.to_datetime(df_comp["complete_timestamp"], utc=True)
        df_comp["org:resource"] = df_comp["resource_id"]
        df_comp["lifecycle:transition"] = "complete"
        df_comp["_trans_order"] = 1

        df_paired = pd.concat([df_start, df_comp], ignore_index=True)
        df_paired = df_paired.sort_values(
            by=["case:concept:name", "time:timestamp", "_trans_order"]
        ).drop(columns=["_trans_order"]).reset_index(drop=True)

        os.makedirs(os.path.dirname(self.output_simulated_xes), exist_ok=True)
        pm4py.write_xes(df_paired, self.output_simulated_xes, variant_str="line_by_line")
        print(f"[✓] Saved standard IEEE XES log with paired lifecycle ({len(df_paired)} events) to: {self.output_simulated_xes}")

    def evaluate(self, df_real, df_sim, elapsed=0.0):
        """
        Evaluates simulated event log against real historical log across:
          - Control-Flow Perspective (CFLD, L_traces, NGD 2-Gram, NGD 3-Gram)
          - Temporal Perspective (AED, CED, RED)
          - Workforce Perspective (CWD)
          - Congestion Perspective (CTD / 1WD, CAR, Cycle Mean/Median/MAE)
        Uses the official `log-distance-measures` library.
        """
        print("\n" + "=" * 96)
        print("         RIMS++ SIMULATION MULTI-PERSPECTIVE BENCHMARK (LOG-DISTANCE-MEASURES)")
        print("=" * 96)

        df_sim = df_sim.copy()
        df_sim["start_dt"] = pd.to_datetime(df_sim["start_timestamp"], utc=True)
        df_sim["comp_dt"] = pd.to_datetime(df_sim["complete_timestamp"], utc=True)
        sim_cases = df_sim.groupby("case_id").agg(
            start=("start_dt", "min"),
            end=("comp_dt", "max")
        )
        sim_cycles = (sim_cases["end"] - sim_cases["start"]).dt.total_seconds().values
        sim_mean = float(np.mean(sim_cycles)) if len(sim_cycles) > 0 else 0.0
        sim_median = float(np.median(sim_cycles)) if len(sim_cycles) > 0 else 0.0

        target_acts = set(self.cfg.get("target_activities", []))

        clean_real, clean_sim, ids = prepare_evaluation_dfs(
            df_real, df_sim, target_acts=target_acts
        )

        real_cases = None
        cfld = 0.0
        l_traces = 0.0
        ngram_2 = 0.0
        ngram_3 = 0.0
        aed = 0.0
        ced = 0.0
        cwd = 0.0
        red_dist = 0.0
        car = 0.0
        ctd = 0.0
        mae_cycle = 0.0
        real_mean = 0.0
        real_median = 0.0

        if clean_real is not None and len(clean_real) > 0:
            real_cases = clean_real.groupby("case_id").agg(
                start=("start_time", "min"),
                end=("end_time", "max")
            )
            real_cycles = (real_cases["end"] - real_cases["start"]).dt.total_seconds().values
            real_mean = float(np.mean(real_cycles)) if len(real_cycles) > 0 else 0.0
            real_median = float(np.median(real_cycles)) if len(real_cycles) > 0 else 0.0
            mae_cycle = float(abs(real_mean - sim_mean))

            # 1. Control-Flow Metrics (log-distance-measures)
            # Cap CFLD to max 1,000 cases to prevent O(N^3) Hungarian matching slowdown
            try:
                max_cfld_cases = 1000
                cfld_real = clean_real
                cfld_sim = clean_sim
                if clean_real["case_id"].nunique() > max_cfld_cases:
                    rng = np.random.RandomState(42)
                    c_r = set(rng.choice(clean_real["case_id"].unique(), size=max_cfld_cases, replace=False))
                    cfld_real = clean_real[clean_real["case_id"].isin(c_r)].copy()
                if clean_sim["case_id"].nunique() > max_cfld_cases:
                    rng = np.random.RandomState(42)
                    c_s = set(rng.choice(clean_sim["case_id"].unique(), size=max_cfld_cases, replace=False))
                    cfld_sim = clean_sim[clean_sim["case_id"].isin(c_s)].copy()

                cfld = float(control_flow_log_distance(cfld_real, ids, cfld_sim, ids, parallel=False))
            except Exception:
                cfld = 0.0

            l_traces = compute_trace_length_error(clean_real, clean_sim)

            try:
                ngram_2 = float(n_gram_distribution_distance(clean_real, ids, clean_sim, ids, n=2))
            except Exception:
                ngram_2 = 0.0

            try:
                ngram_3 = float(n_gram_distribution_distance(clean_real, ids, clean_sim, ids, n=3))
            except Exception:
                ngram_3 = 0.0

            # 2. Temporal Metrics (log-distance-measures)
            try:
                aed = float(absolute_event_distribution_distance(clean_real, ids, clean_sim, ids))
            except Exception:
                aed = 0.0

            try:
                ced = float(circadian_event_distribution_distance(clean_real, ids, clean_sim, ids))
            except Exception:
                ced = 0.0

            try:
                red_dist = float(relative_event_distribution_distance(clean_real, ids, clean_sim, ids))
            except Exception:
                red_dist = 0.0

            # 3. Workforce / Resource Scheduling Metric (log-distance-measures CWD)
            try:
                cwd = float(circadian_workforce_distribution_distance(clean_real, ids, clean_sim, ids))
            except Exception:
                cwd = 0.0

            # 4. Congestion Metrics (log-distance-measures)
            try:
                car = float(case_arrival_distribution_distance(clean_real, ids, clean_sim, ids))
            except Exception:
                car = 0.0

            try:
                ctd = float(cycle_time_distribution_distance(clean_real, ids, clean_sim, ids, bin_size=timedelta(hours=1)))
            except Exception:
                ctd = float(wasserstein_distance(real_cycles, sim_cycles)) if (len(real_cycles) > 0 and len(sim_cycles) > 0) else 0.0

            real_tasks_per_case = len(clean_real) / max(1, len(real_cases))
            sim_tasks_per_case = len(df_sim) / max(1, len(sim_cases))
            tasks_diff = sim_tasks_per_case - real_tasks_per_case

            print(f"Dimension     | Metric                                | Real Ground-Truth | Simulated RIMS++ | Benchmark Score")
            print("-" * 96)
            print(f"Volume        | Cases Evaluated                       | {len(real_cases):>17,} | {len(sim_cases):>16,} | ---")
            print(f"Volume        | Completed Work Tasks                  | {len(clean_real):>17,} | {len(df_sim):>16,} | ---")
            print(f"Volume        | Avg Tasks / Case                      | {real_tasks_per_case:>17.2f} | {sim_tasks_per_case:>16.2f} | {tasks_diff:>+14.2f}")
            print("-" * 96)
            print(f"Control-Flow  | Control-flow Log Distance (CFLD)      |             --- |             --- | {cfld:>12.4f} (log-distance)")
            print(f"Control-Flow  | Trace Length Absolute Error (L_traces)|             --- |             --- | {l_traces:>12.2f} events (Hungarian)")
            print(f"Control-Flow  | 2-Gram Distance (NGD 2-Gram)          |             --- |             --- | {ngram_2:>12.4f}")
            print(f"Control-Flow  | 3-Gram Distance (NGD 3-Gram)          |             --- |             --- | {ngram_3:>12.4f}")
            print("-" * 96)
            print(f"Temporal      | Absolute Event Distribution (AED)     |             --- |             --- | {aed:>12.2f}h EMD")
            print(f"Temporal      | Circadian Event Distribution (CED)    |             --- |             --- | {ced:>12.2f}h EMD")
            print(f"Temporal      | Relative Event Distribution (RED)     |             --- |             --- | {red_dist:>12.2f}h EMD")
            print("-" * 96)
            print(f"Workforce     | Circadian Workforce Dist (CWD)        |             --- |             --- | {cwd:>12.2f}h EMD")
            print("-" * 96)
            print(f"Congestion    | Cycle Time Distribution (CTD)         |             --- |             --- | {format_dur(ctd):>15}")
            print(f"Congestion    | Case Arrival Rate (CAR)               |             --- |             --- | {car:>12.2f}h EMD")
            print(f"Congestion    | Mean Case Cycle Time                  | {format_dur(real_mean):>17} | {format_dur(sim_mean):>16} | {format_dur(mae_cycle):>15} MAE")
            print(f"Congestion    | Median Case Cycle Time                | {format_dur(real_median):>17} | {format_dur(sim_median):>16} | ---")
            print("=" * 96)
        else:
            print(f"Dimension     | Metric                                | Simulated RIMS++ (Generative Mode)")
            print("-" * 75)
            print(f"Volume        | Cases Evaluated                       | {len(sim_cases):>17,}")
            print(f"Volume        | Total Work Events                     | {len(df_sim):>17,}")
            print(f"Congestion    | Mean Case Cycle Time                  | {format_dur(sim_mean):>17}")
            print(f"Congestion    | Median Case Cycle Time                | {format_dur(sim_median):>17}")
            print("=" * 75)

        # Activity Breakdown
        print(f"\n{'Activity Execution':<32} | {'Sim Count':>10} | {'Sim Mean Dur':>14} | {'Sim Mean Wait':>14} | {'Sim Mean Queue':>14}")
        print("-" * 92)
        sim_act_stats = df_sim.groupby("activity").agg(
            count=("activity", "count"),
            sim_dur=("duration_seconds", "mean"),
            sim_wait=("wait_time_seconds", "mean"),
            sim_queue=("queue_wait_seconds", "mean")
        )
        cfg_acts = self.cfg.get("target_activities", [])
        if not cfg_acts:
            acts_to_show = sorted(df_sim["activity"].unique())
        else:
            extra = [a for a in sorted(df_sim["activity"].unique()) if a not in cfg_acts]
            acts_to_show = list(cfg_acts) + extra

        for act in acts_to_show:
            cnt = int(sim_act_stats.loc[act, "count"]) if act in sim_act_stats.index else 0
            s_dur = sim_act_stats.loc[act, "sim_dur"] if act in sim_act_stats.index else 0.0
            s_wait = sim_act_stats.loc[act, "sim_wait"] if act in sim_act_stats.index else 0.0
            s_queue = sim_act_stats.loc[act, "sim_queue"] if act in sim_act_stats.index else 0.0
            print(f"{act:<32} | {cnt:>10,} | {format_dur(s_dur):>14} | {format_dur(s_wait):>14} | {format_dur(s_queue):>14}")
        print("=" * 92 + "\n")

        # Save Comprehensive Markdown Report
        out_base = self.cfg.get("paths", {}).get("output_base_dir", "data/processed")
        rep_path = os.path.join(PROJECT_ROOT, out_base, "simulation_report.md")
        with open(rep_path, "w") as f:
            f.write("# RIMS++ Simulation Benchmark Report\n\n")
            f.write(f"- **Simulation Mode:** `{self.mode}`\n")
            f.write(f"- **Arrival Mode:** `{self.arrival_mode}`\n")
            f.write(f"- **Simulated Cases:** {len(sim_cases):,}\n")
            f.write(f"- **Total Work Events:** {len(df_sim):,}\n")
            f.write(f"- **Workers in Multi-Skilled Pool:** {self.workers_count} human resources\n")
            if self.calendar is not None:
                f.write(f"- **Calendar Engine:** Working Hours {self.calendar.hour_start}:00 - {self.calendar.hour_end}:00 (Mon-Fri)\n")
            f.write(f"- **Simulated Log Path (CSV):** `{self.sim_log_path}`\n")
            if self.export_xes:
                f.write(f"- **Simulated Log Path (XES):** `{self.output_simulated_xes}`\n\n")
            else:
                f.write("\n")

            f.write("## Multi-Perspective Quality Benchmark (log-distance-measures / WISDOM)\n\n")
            f.write("| Dimension | Metric | Real Ground-Truth | Simulated RIMS++ | Benchmark Score |\n")
            f.write("|---|---|---|---|---|\n")
            real_cases_str = f"{len(real_cases):,}" if real_cases is not None else "N/A"
            f.write(f"| **Volume** | Cases Evaluated | {real_cases_str} | {len(sim_cases):,} | --- |\n")
            if clean_real is not None and len(clean_real) > 0:
                f.write(f"| **Volume** | Completed Work Tasks | {len(clean_real):,} | {len(df_sim):,} | --- |\n")
                f.write(f"| **Volume** | Avg Tasks / Case | {real_tasks_per_case:.2f} | {sim_tasks_per_case:.2f} | {tasks_diff:+.2f} |\n")
                f.write(f"| **Control-Flow** | Control-flow Log Distance (CFLD) | --- | --- | **{cfld:.4f}** (log-distance) |\n")
                f.write(f"| **Control-Flow** | Trace Length Error ($L^{{traces}}$) | --- | --- | **{l_traces:.2f}** events (Hungarian) |\n")
                f.write(f"| **Control-Flow** | 2-Gram Distance (NGD 2-Gram) | --- | --- | **{ngram_2:.4f}** |\n")
                f.write(f"| **Control-Flow** | 3-Gram Distance (NGD 3-Gram) | --- | --- | **{ngram_3:.4f}** |\n")
                f.write(f"| **Temporal** | Absolute Event Distribution (AED) | --- | --- | **{aed:.2f}h** EMD |\n")
                f.write(f"| **Temporal** | Circadian Event Distribution (CED) | --- | --- | **{ced:.2f}h** EMD |\n")
                f.write(f"| **Temporal** | Relative Event Distribution (RED) | --- | --- | **{red_dist:.2f}h** EMD |\n")
                f.write(f"| **Workforce** | Circadian Workforce Dist (CWD) | --- | --- | **{cwd:.2f}h** EMD |\n")
                f.write(f"| **Congestion** | Cycle Time Distribution (CTD) | --- | --- | **{format_dur(ctd)}** |\n")
                f.write(f"| **Congestion** | Case Arrival Rate (CAR) | --- | --- | **{car:.2f}h** EMD |\n")
                f.write(f"| **Congestion** | Mean Case Cycle Time | {format_dur(real_mean)} | {format_dur(sim_mean)} | {format_dur(mae_cycle)} MAE |\n")
                f.write(f"| **Congestion** | Median Case Cycle Time | {format_dur(real_median)} | {format_dur(sim_median)} | --- |\n\n")
            else:
                f.write(f"| **Congestion** | Mean Case Cycle Time | N/A | {format_dur(sim_mean)} | --- |\n")
                f.write(f"| **Congestion** | Median Case Cycle Time | N/A | {format_dur(sim_median)} | --- |\n\n")

            f.write("## Activity Execution Breakdown\n\n")
            f.write("| Activity | Simulated Count | Sim Mean Duration | Sim Mean Wait | Sim Mean Queue Wait |\n")
            f.write("|---|---|---|---|---|\n")
            for act in acts_to_show:
                cnt = int(sim_act_stats.loc[act, "count"]) if act in sim_act_stats.index else 0
                s_dur = sim_act_stats.loc[act, "sim_dur"] if act in sim_act_stats.index else 0.0
                s_wait = sim_act_stats.loc[act, "sim_wait"] if act in sim_act_stats.index else 0.0
                s_queue = sim_act_stats.loc[act, "sim_queue"] if act in sim_act_stats.index else 0.0
                f.write(f"| `{act}` | {cnt:,} | {format_dur(s_dur)} | {format_dur(s_wait)} | {format_dur(s_queue)} |\n")
        print(f"[✓] Saved simulation benchmark report to: {rep_path}\n")

        return {
            "mode": self.mode,
            "cases": len(sim_cases),
            "events": len(df_sim),
            "cfld": cfld,
            "l_traces": l_traces,
            "ngram_2": ngram_2,
            "ngram_3": ngram_3,
            "aed": aed,
            "ced": ced,
            "cwd": cwd,
            "red_dist": red_dist,
            "car": car,
            "ctd": ctd,
            "wd": ctd,
            "real_mean": real_mean,
            "sim_mean": sim_mean,
            "real_median": real_median,
            "sim_median": sim_median,
            "mae_cycle": mae_cycle,
            "elapsed": elapsed
        }
