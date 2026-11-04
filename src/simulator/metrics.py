#!/usr/bin/env python3
"""
src/simulator/metrics.py

  1. Control-Flow Quality:
     - Control-flow Log Distance (CFLD, Hungarian + Levenshtein)
     - Trace Length Absolute Error (L_traces, Hungarian matching)
     - N-Gram Distribution Distance (NGD: 2-Gram and 3-Gram)
  2. Temporal Distribution Quality:
     - Absolute Event Distribution (AED, 1WD / EMD across calendar timeline)
     - Circadian Event Distribution (CED, 1WD / EMD across 168h weekly cycle)
     - Relative Event Distribution (RED, 1WD / EMD relative to trace origin)
  3. Congestion & Cycle Time Quality:
     - Cycle Time Distribution (CTD / 1WD Wasserstein Distance)
     - Case Arrival Rate (CAR, 1WD across case arrival timeline)
     - Cycle Time Mean, Median, and MAE
"""

import os
from collections import Counter
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
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


def levenshtein_distance(seq1, seq2):
    """Computes Levenshtein edit distance between two sequences of activity labels."""
    n1, n2 = len(seq1), len(seq2)
    dp = list(range(n2 + 1))
    for i, c1 in enumerate(seq1):
        new_dp = [i + 1] * (n2 + 1)
        for j, c2 in enumerate(seq2):
            cost = 0 if c1 == c2 else 1
            new_dp[j + 1] = min(dp[j + 1] + 1, new_dp[j] + 1, dp[j] + cost)
        dp = new_dp
    return dp[n2]


def compute_cfld_and_trace_length(traces_real, traces_sim, max_sample=1000):
    """
    Computes:
      - Control-flow Log Distance (CFLD): Hungarian minimum assignment over normalized Levenshtein distance.
      - L_traces: Average absolute difference in trace lengths paired via Hungarian algorithm.
    """
    if not traces_real or not traces_sim:
        return {"cfld": 0.0, "l_traces": 0.0}

    # Sample for speed if traces exceed max_sample
    rng = np.random.RandomState(42)
    if len(traces_real) > max_sample:
        idx_r = rng.choice(len(traces_real), size=max_sample, replace=False)
        r_sub = [traces_real[i] for i in idx_r]
    else:
        r_sub = traces_real

    if len(traces_sim) > max_sample:
        idx_s = rng.choice(len(traces_sim), size=max_sample, replace=False)
        s_sub = [traces_sim[i] for i in idx_s]
    else:
        s_sub = traces_sim

    nr, ns = len(r_sub), len(s_sub)
    n = min(nr, ns)
    if n == 0:
        return {"cfld": 0.0, "l_traces": 0.0}

    # Cache edit distances between unique trace variants
    u_real = list(set(r_sub))
    u_sim = list(set(s_sub))
    dist_cache = {}
    for r_var in u_real:
        len_r = len(r_var)
        for s_var in u_sim:
            len_s = len(s_var)
            max_len = max(len_r, len_s)
            dist_cache[(r_var, s_var)] = levenshtein_distance(r_var, s_var) / max_len if max_len > 0 else 0.0

    cost_cfld = np.zeros((nr, ns), dtype=float)
    cost_len = np.zeros((nr, ns), dtype=float)
    r_lens = [len(t) for t in r_sub]
    s_lens = [len(t) for t in s_sub]

    for i in range(nr):
        t_r = r_sub[i]
        len_r = r_lens[i]
        for j in range(ns):
            t_s = s_sub[j]
            cost_cfld[i, j] = dist_cache[(t_r, t_s)]
            cost_len[i, j] = abs(len_r - s_lens[j])

    row_c, col_c = linear_sum_assignment(cost_cfld)
    cfld = float(cost_cfld[row_c, col_c].sum() / n)

    row_l, col_l = linear_sum_assignment(cost_len)
    l_traces = float(cost_len[row_l, col_l].sum() / n)

    return {"cfld": cfld, "l_traces": l_traces}


def compute_ngram_distance(traces_real, traces_sim, n=2):
    """
    Computes N-Gram Distance (NGD) comparing sub-sequence frequency distributions (N=2, N=3).
    Bounded in [0.0, 1.0].
    """
    if not traces_real or not traces_sim:
        return 0.0

    def extract_ngrams(traces, ngram_len):
        counts = Counter()
        for tr in traces:
            tokens = ["▶"] + list(tr) + ["■"]
            for i in range(len(tokens) - ngram_len + 1):
                counts[tuple(tokens[i:i + ngram_len])] += 1
        return counts

    r_counts = extract_ngrams(traces_real, n)
    s_counts = extract_ngrams(traces_sim, n)

    all_keys = sorted(list(set(r_counts.keys()) | set(s_counts.keys())))
    r_total = max(1, sum(r_counts.values()))
    s_total = max(1, sum(s_counts.values()))

    p_r = np.array([r_counts.get(k, 0) / r_total for k in all_keys], dtype=float)
    p_s = np.array([s_counts.get(k, 0) / s_total for k in all_keys], dtype=float)

    return float(0.5 * np.sum(np.abs(p_r - p_s)))


class SimulationEvaluator:
    """
    Multi-Dimensional Benchmark Evaluator and Serializer.
    Full parity with the Meneghello et al. (Information Systems 2025) paper evaluation suite.
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
        pm4py.write_xes(df_paired, self.output_simulated_xes)
        print(f"[✓] Saved standard IEEE XES log with paired lifecycle ({len(df_paired)} events) to: {self.output_simulated_xes}")

    def compute_aed(self, df_real, df_sim, bin_size_hours=1.0):
        """
        Absolute Event Distribution (AED) Distance (1WD / EMD in hours).
        Measures difference in the trend of all events across the absolute timeline.
        """
        try:
            bin_sec = bin_size_hours * 3600.0
            t_col = "time:timestamp" if "time:timestamp" in df_real.columns else "start_timestamp"
            r_ts = pd.to_datetime(df_real[t_col], utc=True).astype("int64") // 10**9
            s_ts = pd.to_datetime(df_sim["start_timestamp"], utc=True).astype("int64") // 10**9

            if len(r_ts) == 0 or len(s_ts) == 0:
                return 0.0

            min_t = min(r_ts.min(), s_ts.min())
            r_bins = np.floor((r_ts - min_t) / bin_sec)
            s_bins = np.floor((s_ts - min_t) / bin_sec)

            return float(wasserstein_distance(r_bins, s_bins))
        except Exception:
            return 0.0

    def compute_ced(self, df_real, df_sim):
        """
        Circadian Event Distribution (CED) Distance (1WD / EMD in hours).
        Measures seasonality and diurnal patterns across 168 hours of the week (7 days x 24 hours).
        """
        try:
            t_col = "time:timestamp" if "time:timestamp" in df_real.columns else "start_timestamp"
            r_dt = pd.to_datetime(df_real[t_col], utc=True)
            s_dt = pd.to_datetime(df_sim["start_timestamp"], utc=True)

            r_circ = (r_dt.dt.weekday * 24 + r_dt.dt.hour).values
            s_circ = (s_dt.dt.weekday * 24 + s_dt.dt.hour).values

            if len(r_circ) == 0 or len(s_circ) == 0:
                return 0.0

            return float(wasserstein_distance(r_circ, s_circ))
        except Exception:
            return 0.0

    def compute_car(self, df_real, df_sim, bin_size_hours=1.0):
        """
        Case Arrival Rate (CAR) Distance (1WD / EMD in hours).
        Measures difference in case arrival patterns and volumes over time.
        """
        try:
            bin_sec = bin_size_hours * 3600.0
            c_col = "case:concept:name" if "case:concept:name" in df_real.columns else "case_id"
            t_col = "time:timestamp" if "time:timestamp" in df_real.columns else "start_timestamp"

            r_arrivals = df_real.groupby(c_col)[t_col].min()
            s_arrivals = df_sim.groupby("case_id")["start_timestamp"].min()

            r_ts = pd.to_datetime(r_arrivals, utc=True).astype("int64") // 10**9
            s_ts = pd.to_datetime(s_arrivals, utc=True).astype("int64") // 10**9

            if len(r_ts) == 0 or len(s_ts) == 0:
                return 0.0

            min_t = min(r_ts.min(), s_ts.min())
            r_bins = np.floor((r_ts - min_t) / bin_sec)
            s_bins = np.floor((s_ts - min_t) / bin_sec)

            return float(wasserstein_distance(r_bins, s_bins))
        except Exception:
            return 0.0

    def compute_red_distance(self, df_real, df_sim, bin_size_hours=1.0):
        """
        Relative Event Distribution (RED) Distance (1WD / EMD in hours).
        Measures how closely simulated event offsets replicate historical event timing
        relative to trace start.
        """
        try:
            bin_sec = bin_size_hours * 3600.0
            r_df = df_real.copy()
            c_col = "case:concept:name" if "case:concept:name" in r_df.columns else "case_id"
            t_col = "time:timestamp" if "time:timestamp" in r_df.columns else "start_timestamp"
            r_ts = pd.to_datetime(r_df[t_col], utc=True)
            r_starts = r_df.groupby(c_col)[t_col].transform("min")
            r_offsets = (r_ts - pd.to_datetime(r_starts, utc=True)).dt.total_seconds().values
            r_bins = np.floor(r_offsets / bin_sec)

            s_df = df_sim.copy()
            s_ts = pd.to_datetime(s_df["start_timestamp"], utc=True)
            s_starts = s_df.groupby("case_id")["start_timestamp"].transform("min")
            s_offsets = (s_ts - pd.to_datetime(s_starts, utc=True)).dt.total_seconds().values
            s_bins = np.floor(s_offsets / bin_sec)

            if len(r_bins) > 0 and len(s_bins) > 0:
                return float(wasserstein_distance(r_bins, s_bins))
            return 0.0
        except Exception:
            return 0.0

    def evaluate(self, df_real, df_sim, elapsed=0.0):
        """
        Compares simulated event log against real historical log across:
          - Control-Flow Perspective (CFLD, L_traces, NGD 2-Gram, NGD 3-Gram)
          - Temporal Perspective (AED, CED, RED)
          - Congestion Perspective (CTD / 1WD, CAR, Cycle Mean/Median/MAE)
        Prints formatted benchmark tables and writes comprehensive Markdown report.
        """
        print("\n" + "=" * 96)
        print("         RIMS++ SIMULATION MULTI-PERSPECTIVE BENCHMARK (MENEGHELLO ET AL. 2025)")
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

        # Simulated Traces for Control-Flow Evaluation
        sim_sorted = df_sim.sort_values(by=["case_id", "start_timestamp"])
        traces_sim = sim_sorted.groupby("case_id")["activity"].apply(tuple).tolist()

        real_w = None
        real_cases = None
        cfld = 0.0
        l_traces = 0.0
        ngram_2 = 0.0
        ngram_3 = 0.0
        aed = 0.0
        ced = 0.0
        red_dist = 0.0
        car = 0.0
        wd = 0.0
        mae_cycle = 0.0
        real_mean = 0.0
        real_median = 0.0

        if df_real is not None:
            proc_prefixes = tuple(self.cfg.get("preprocessing", {}).get("process_activity_prefixes", []))
            target_acts = set(self.cfg.get("target_activities", []))
            if target_acts:
                real_w = df_real[df_real["concept:name"].isin(target_acts)].copy()
            elif proc_prefixes:
                real_w = df_real[df_real["concept:name"].str.startswith(proc_prefixes)].copy()
            else:
                real_w = df_real.copy()

            # Filter to completed tasks if lifecycle transition is present
            if "lifecycle:transition" in real_w.columns:
                real_completed = real_w[real_w["lifecycle:transition"].str.upper() == "COMPLETE"].copy()
            else:
                real_completed = real_w.copy()

            c_col = "case:concept:name" if "case:concept:name" in real_completed.columns else "case_id"
            t_col = "time:timestamp" if "time:timestamp" in real_completed.columns else "start_timestamp"
            act_col = "concept:name" if "concept:name" in real_completed.columns else "activity"

            real_cases = real_w.groupby(c_col).agg(
                start=(t_col, "min"),
                end=(t_col, "max")
            )
            real_cycles = (pd.to_datetime(real_cases["end"], utc=True) - pd.to_datetime(real_cases["start"], utc=True)).dt.total_seconds().values
            real_mean = float(np.mean(real_cycles)) if len(real_cycles) > 0 else 0.0
            real_median = float(np.median(real_cycles)) if len(real_cycles) > 0 else 0.0
            wd = float(wasserstein_distance(real_cycles, sim_cycles)) if (len(real_cycles) > 0 and len(sim_cycles) > 0) else 0.0
            mae_cycle = float(abs(real_mean - sim_mean))

            # 1. Control-Flow Metrics
            real_sorted = real_completed.sort_values(by=[c_col, t_col])
            traces_real = real_sorted.groupby(c_col)[act_col].apply(tuple).tolist()

            cf_res = compute_cfld_and_trace_length(traces_real, traces_sim)
            cfld = cf_res["cfld"]
            l_traces = cf_res["l_traces"]
            ngram_2 = compute_ngram_distance(traces_real, traces_sim, n=2)
            ngram_3 = compute_ngram_distance(traces_real, traces_sim, n=3)

            # 2. Temporal Metrics
            aed = self.compute_aed(real_completed, df_sim)
            ced = self.compute_ced(real_completed, df_sim)
            red_dist = self.compute_red_distance(real_completed, df_sim)

            # 3. Congestion Metrics
            car = self.compute_car(real_completed, df_sim)

            real_tasks_per_case = len(real_completed) / max(1, len(real_cases))
            sim_tasks_per_case = len(df_sim) / max(1, len(sim_cases))
            tasks_diff = sim_tasks_per_case - real_tasks_per_case

            print(f"Dimension     | Metric                                | Real Ground-Truth | Simulated RIMS++ | Benchmark Score")
            print("-" * 96)
            print(f"Volume        | Cases Evaluated                       | {len(real_cases):>17,} | {len(sim_cases):>16,} | ---")
            print(f"Volume        | Completed Work Tasks                  | {len(real_completed):>17,} | {len(df_sim):>16,} | ---")
            print(f"Volume        | Avg Tasks / Case                      | {real_tasks_per_case:>17.2f} | {sim_tasks_per_case:>16.2f} | {tasks_diff:>+14.2f}")
            print("-" * 96)
            print(f"Control-Flow  | Control-flow Log Distance (CFLD)      |             --- |             --- | {cfld:>12.4f} (Hungarian)")
            print(f"Control-Flow  | Trace Length Absolute Error (L_traces)|             --- |             --- | {l_traces:>12.2f} events")
            print(f"Control-Flow  | 2-Gram Distance (NGD 2-Gram)          |             --- |             --- | {ngram_2:>12.4f}")
            print(f"Control-Flow  | 3-Gram Distance (NGD 3-Gram)          |             --- |             --- | {ngram_3:>12.4f}")
            print("-" * 96)
            print(f"Temporal      | Absolute Event Distribution (AED)     |             --- |             --- | {aed:>12.2f}h EMD")
            print(f"Temporal      | Circadian Event Distribution (CED)    |             --- |             --- | {ced:>12.2f}h EMD")
            print(f"Temporal      | Relative Event Distribution (RED)     |             --- |             --- | {red_dist:>12.2f}h EMD")
            print("-" * 96)
            print(f"Congestion    | Cycle Time 1WD (CTD)                  |             --- |             --- | {format_dur(wd):>15}")
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

            f.write("## Multi-Perspective Quality Benchmark\n\n")
            f.write("| Dimension | Metric | Real Ground-Truth | Simulated RIMS++ | Benchmark Score |\n")
            f.write("|---|---|---|---|---|\n")
            real_cases_str = f"{len(real_cases):,}" if real_cases is not None else "N/A"
            f.write(f"| **Volume** | Cases Evaluated | {real_cases_str} | {len(sim_cases):,} | --- |\n")
            if real_cases is not None:
                f.write(f"| **Volume** | Completed Work Tasks | {len(real_completed):,} | {len(df_sim):,} | --- |\n")
                f.write(f"| **Volume** | Avg Tasks / Case | {real_tasks_per_case:.2f} | {sim_tasks_per_case:.2f} | {tasks_diff:+.2f} |\n")
                f.write(f"| **Control-Flow** | Control-flow Log Distance (CFLD) | --- | --- | **{cfld:.4f}** (Hungarian) |\n")
                f.write(f"| **Control-Flow** | Trace Length Error ($L^{{traces}}$) | --- | --- | **{l_traces:.2f}** events |\n")
                f.write(f"| **Control-Flow** | 2-Gram Distance (NGD 2-Gram) | --- | --- | **{ngram_2:.4f}** |\n")
                f.write(f"| **Control-Flow** | 3-Gram Distance (NGD 3-Gram) | --- | --- | **{ngram_3:.4f}** |\n")
                f.write(f"| **Temporal** | Absolute Event Distribution (AED) | --- | --- | **{aed:.2f}h** EMD |\n")
                f.write(f"| **Temporal** | Circadian Event Distribution (CED) | --- | --- | **{ced:.2f}h** EMD |\n")
                f.write(f"| **Temporal** | Relative Event Distribution (RED) | --- | --- | **{red_dist:.2f}h** EMD |\n")
                f.write(f"| **Congestion** | Cycle Time 1WD (CTD) | --- | --- | **{format_dur(wd)}** |\n")
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
            "red_dist": red_dist,
            "car": car,
            "ctd": wd,
            "wd": wd,
            "real_mean": real_mean,
            "sim_mean": sim_mean,
            "real_median": real_median,
            "sim_median": sim_median,
            "mae_cycle": mae_cycle,
            "elapsed": elapsed
        }

