#!/usr/bin/env python3
"""
src/simulator/metrics.py

Simulation Output Exporter & Benchmark Evaluator.
Handles:
  1. CSV Event Log Serialization (simulated_log.csv)
  2. Standard IEEE XES Event Log Export (simulated_log.xes)
  3. Cycle Time Validation (Mean, Median, MAE, 1D Wasserstein Distance)
  4. Activity Breakdown Statistics & Markdown Report Generation
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import pm4py

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


class SimulationEvaluator:
    """
    Serializer and Benchmark Evaluator (Legacy Parity with result_analysis.py / evaluate.py).
    """
    def __init__(self, cfg, mode="hybrid_residual", arrival_mode="replay", export_xes=True,
                 sim_log_path=None, output_simulated_xes=None, workers_count=59, calendar=None):
        self.cfg = cfg
        self.mode = mode
        self.arrival_mode = arrival_mode
        self.export_xes = export_xes
        self.sim_log_path = sim_log_path or os.path.join(
            PROJECT_ROOT,
            f"data/processed/simulated_log_{self.mode}.csv" if self.mode != "hybrid_residual" else
            cfg.get("simulation", {}).get("output_simulated_log", "data/processed/simulated_log.csv")
        )
        self.output_simulated_xes = output_simulated_xes or os.path.join(
            PROJECT_ROOT,
            cfg.get("simulation", {}).get("output_simulated_xes", "data/processed/simulated_log.xes")
        )
        self.workers_count = workers_count
        self.calendar = calendar

    def save_csv(self, df_sim):
        """Saves simulated event log to flat CSV format."""
        os.makedirs(os.path.dirname(self.sim_log_path), exist_ok=True)
        df_sim.to_csv(self.sim_log_path, index=False)
        print(f"[✓] Saved simulated event log (CSV) to: {self.sim_log_path}")

    def save_xes(self, df_sim):
        """Saves simulated event log to standard IEEE XES format using PM4Py."""
        if not self.export_xes or len(df_sim) == 0:
            return
        df_xes = df_sim.copy()
        df_xes["case:concept:name"] = df_xes["case_id"].astype(str)
        df_xes["concept:name"] = df_xes["activity"]
        df_xes["time:timestamp"] = pd.to_datetime(df_xes["complete_timestamp"])
        df_xes["start_timestamp"] = pd.to_datetime(df_xes["start_timestamp"])
        df_xes["org:resource"] = df_xes["resource_id"]
        df_xes["lifecycle:transition"] = "complete"
        os.makedirs(os.path.dirname(self.output_simulated_xes), exist_ok=True)
        pm4py.write_xes(df_xes, self.output_simulated_xes)
        print(f"[✓] Saved standard IEEE XES log to: {self.output_simulated_xes}")

    def compute_red_distance(self, df_real, df_sim, bin_size_hours=1.0):
        """
        Computes Relative Event Distribution (RED) distance (Earth Mover's Distance)
        between real and simulated logs (Chapela-Campa).
        Measures how closely simulated event offsets replicate historical event timing
        relative to trace start.
        """
        try:
            bin_sec = bin_size_hours * 3600.0
            r_df = df_real.copy()
            c_col = "case:concept:name" if "case:concept:name" in r_df.columns else "case_id"
            t_col = "time:timestamp" if "time:timestamp" in r_df.columns else "start_timestamp"
            r_ts = pd.to_datetime(r_df[t_col])
            r_starts = r_df.groupby(c_col)[t_col].transform("min")
            r_offsets = (r_ts - pd.to_datetime(r_starts)).dt.total_seconds().values
            r_bins = np.floor(r_offsets / bin_sec)

            s_df = df_sim.copy()
            s_ts = pd.to_datetime(s_df["start_timestamp"])
            s_starts = s_df.groupby("case_id")["start_timestamp"].transform("min")
            s_offsets = (s_ts - pd.to_datetime(s_starts)).dt.total_seconds().values
            s_bins = np.floor(s_offsets / bin_sec)

            if len(r_bins) > 0 and len(s_bins) > 0:
                return float(wasserstein_distance(r_bins, s_bins))
            return 0.0
        except Exception:
            return 0.0

    def evaluate(self, df_real, df_sim, elapsed=0.0):
        """
        Compares simulated event log against real historical log (or summarizes simulated stats),
        prints formatted comparison tables, and generates a markdown report.
        """
        print("\n" + "=" * 80)
        print("          RIMS+ SIMULATION VS. REAL LOG BENCHMARK")
        print("=" * 80)

        # Simulated cycle times
        df_sim = df_sim.copy()
        df_sim["start_dt"] = pd.to_datetime(df_sim["start_timestamp"])
        df_sim["comp_dt"] = pd.to_datetime(df_sim["complete_timestamp"])
        sim_cases = df_sim.groupby("case_id").agg(
            start=("start_dt", "min"),
            end=("comp_dt", "max")
        )
        sim_cycles = (sim_cases["end"] - sim_cases["start"]).dt.total_seconds().values
        sim_mean = float(np.mean(sim_cycles)) if len(sim_cycles) > 0 else 0.0
        sim_median = float(np.median(sim_cycles)) if len(sim_cycles) > 0 else 0.0

        real_w = None
        real_cases = None
        if df_real is not None:
            # Real cycle times
            proc_prefixes = tuple(self.cfg.get("preprocessing", {}).get("process_activity_prefixes", []))
            target_acts = set(self.cfg.get("target_activities", []))
            if target_acts:
                real_w = df_real[df_real["concept:name"].isin(target_acts)].copy()
            elif proc_prefixes:
                real_w = df_real[df_real["concept:name"].str.startswith(proc_prefixes)].copy()
            else:
                real_w = df_real.copy()

            real_cases = real_w.groupby("case:concept:name").agg(
                start=("time:timestamp", "min"),
                end=("time:timestamp", "max")
            )
            real_cycles = (real_cases["end"] - real_cases["start"]).dt.total_seconds().values
            real_mean = float(np.mean(real_cycles)) if len(real_cycles) > 0 else 0.0
            real_median = float(np.median(real_cycles)) if len(real_cycles) > 0 else 0.0
            wd = float(wasserstein_distance(real_cycles, sim_cycles)) if (len(real_cycles) > 0 and len(sim_cycles) > 0) else 0.0
            mae_cycle = float(abs(real_mean - sim_mean))
            red_dist = self.compute_red_distance(real_w, df_sim)

            # Filter to completed tasks (excluding START lifecycle rows)
            if "lifecycle:transition" in real_w.columns:
                real_completed = real_w[real_w["lifecycle:transition"].str.upper() == "COMPLETE"]
            else:
                real_completed = real_w

            real_tasks_per_case = len(real_completed) / max(1, len(real_cases))
            sim_tasks_per_case = len(df_sim) / max(1, len(sim_cases))
            tasks_diff = sim_tasks_per_case - real_tasks_per_case

            print(f"Metric                      | Real Historical | Simulated RIMS+ | Difference")
            print("-" * 75)
            print(f"Cases Evaluated             | {len(real_cases):>15,} | {len(sim_cases):>15,} | ---")
            print(f"Completed Work Tasks        | {len(real_completed):>15,} | {len(df_sim):>15,} | ---")
            print(f"Avg Tasks / Case            | {real_tasks_per_case:>15.2f} | {sim_tasks_per_case:>15.2f} | {tasks_diff:>+14.2f}")
            print(f"Mean Case Cycle Time        | {format_dur(real_mean):>15} | {format_dur(sim_mean):>15} | {format_dur(mae_cycle)}")
            print(f"Median Case Cycle Time      | {format_dur(real_median):>15} | {format_dur(sim_median):>15} | ---")
            print(f"Cycle Time Wasserstein Dist |             --- |             --- | {format_dur(wd)}")
            print(f"Relative Event Dist (RED)   |             --- |             --- | {red_dist:>12.2f}h EMD")
            print("=" * 75)
        else:
            real_mean = real_median = mae_cycle = wd = 0.0
            print(f"Metric                      | Simulated RIMS+ (Generative Mode)")
            print("-" * 55)
            print(f"Cases Evaluated             | {len(sim_cases):>15,}")
            print(f"Total Work Events           | {len(df_sim):>15,}")
            print(f"Mean Case Cycle Time        | {format_dur(sim_mean):>15}")
            print(f"Median Case Cycle Time      | {format_dur(sim_median):>15}")
            print("=" * 55)

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
            f.write("# RIMS+ Simulation Benchmark Report\n\n")
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
            f.write("## Overall Cycle Time Benchmark\n\n")
            f.write("| Metric | Real Historical | Simulated RIMS+ | Difference |\n")
            f.write("|---|---|---|---|\n")
            f.write(f"| **Cases Evaluated** | {len(real_cases) if real_cases is not None else 'N/A'} | {len(sim_cases):,} | --- |\n")
            if real_cases is not None:
                f.write(f"| **Completed Work Tasks** | {len(real_completed):,} | {len(df_sim):,} | --- |\n")
                f.write(f"| **Avg Tasks / Case** | {real_tasks_per_case:.2f} | {sim_tasks_per_case:.2f} | {tasks_diff:+.2f} |\n")
            else:
                f.write(f"| **Total Work Events** | N/A | {len(df_sim):,} | --- |\n")
            f.write(f"| **Mean Case Cycle Time** | {format_dur(real_mean)} | {format_dur(sim_mean)} | {format_dur(mae_cycle)} |\n")
            f.write(f"| **Median Case Cycle Time** | {format_dur(real_median)} | {format_dur(sim_median)} | --- |\n")
            f.write(f"| **Cycle Time Wasserstein Dist** | --- | --- | {format_dur(wd)} |\n")
            f.write(f"| **Relative Event Dist (RED)** | --- | --- | **{red_dist:.2f}h EMD** |\n\n")
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
            "real_mean": real_mean,
            "sim_mean": sim_mean,
            "real_median": real_median,
            "sim_median": sim_median,
            "mae_cycle": mae_cycle,
            "wd": wd,
            "red_dist": red_dist,
            "elapsed": elapsed
        }
