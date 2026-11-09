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
import json
import warnings
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
    if sec is None or not np.isfinite(sec):
        return "N/A"
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
        df_start = df_sim[df_sim["start_timestamp"].notna()].copy()
        df_start["case:concept:name"] = df_start["case_id"].astype(str)
        df_start["concept:name"] = df_start["activity"]
        df_start["time:timestamp"] = pd.to_datetime(df_start["start_timestamp"], utc=True)
        df_start["org:resource"] = df_start["resource_id"]
        df_start["lifecycle:transition"] = "start"
        df_start["_trans_order"] = 0

        # Paired COMPLETE lifecycle events
        df_comp = df_sim[df_sim["complete_timestamp"].notna()].copy()
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
        """Report full-case distances separately from clipped observation-window data."""
        cols = ["case_id", "activity", "start_timestamp", "complete_timestamp", "resource_id",
                "duration_seconds", "wait_time_seconds", "queue_wait_seconds"]
        df_sim = df_sim.reindex(columns=list(dict.fromkeys(list(df_sim.columns) + cols)))
        df_sim["case_id"] = df_sim.case_id.astype(str)
        clean_real, clean_sim, ids = prepare_evaluation_dfs(
            df_real, df_sim, target_acts=set(self.cfg.get("target_activities", [])), max_cases=None)
        statuses = getattr(self, "case_status", None)
        window_start = getattr(self, "window_start", None)
        window_end = getattr(self, "window_end", None)
        populations = {"simulated_scheduled": len(statuses) if statuses is not None else clean_sim.case_id.nunique(),
                       "simulated_statuses": statuses.status.value_counts().to_dict() if statuses is not None else {},
                       "reference_cases": df_real["case:concept:name"].nunique() if df_real is not None and not df_real.empty else 0}
        def clipped(frame):
            if frame is None:
                return None
            out = frame.copy()
            if window_end is not None:
                out = out[out.start_time < window_end].copy()
                out["end_time"] = out.end_time.fillna(window_end).clip(upper=window_end)
            if window_start is not None:
                out = out[out.end_time >= window_start].copy()
                out["start_time"] = out.start_time.clip(lower=window_start)
            return out.dropna(subset=["start_time", "end_time"])
        window_real, window_sim = clipped(clean_real), clipped(clean_sim)
        real_ids = set(clean_real.case_id) if clean_real is not None else set()
        sim_ids = set(statuses.loc[statuses.status == "completed", "case_id"].astype(str)) if statuses is not None else set(clean_sim.case_id)
        real_bounds = None
        if df_real is not None and not df_real.empty:
            raw = df_real.assign(_timestamp=pd.to_datetime(df_real["time:timestamp"], utc=True))
            real_bounds = raw.groupby("case:concept:name")._timestamp.agg(["min", "max"])
            real_bounds.index = real_bounds.index.astype(str)
            if window_end is not None:
                real_ids &= set(real_bounds.index[real_bounds["max"] < window_end])
        paired = self.arrival_mode == "replay" and clean_real is not None
        if paired:
            real_ids = sim_ids = real_ids & sim_ids
        # Sample one paired ID set for replay, separately for generated populations.
        rng = np.random.RandomState(42)
        def sample(case_ids):
            ordered = sorted(case_ids)
            return set(rng.choice(ordered, size=1000, replace=False)) if len(ordered) > 1000 else set(ordered)
        real_ids = sample(real_ids)
        sim_ids = real_ids if paired else sample(sim_ids)
        real = clean_real[clean_real.case_id.isin(real_ids)].copy() if clean_real is not None else None
        sim = clean_sim[clean_sim.case_id.isin(sim_ids)].dropna(subset=["end_time"]).copy()
        populations.update(completed_reference_evaluated=len(real_ids), completed_simulated_evaluated=len(sim_ids),
                           window_reference_intervals=len(window_real) if window_real is not None else 0,
                           window_simulated_intervals=len(window_sim))
        metric_status = {}
        def calculate(name, fn, available=True):
            if not available:
                metric_status[name] = {"status": "unavailable", "reason": "empty or missing comparison population"}
                return None
            try:
                value = float(fn())
                if not np.isfinite(value):
                    raise ValueError("non-finite metric result")
                metric_status[name] = {"status": "ok"}
                return value
            except Exception as exc:
                metric_status[name] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                warnings.warn(f"Metric {name} unavailable: {exc}")
                return None
        available = real is not None and not real.empty and not sim.empty
        functions = {
            "cfld": lambda: control_flow_log_distance(real, ids, sim, ids, parallel=False),
            "l_traces": lambda: compute_trace_length_error(real, sim),
            "ngram_2": lambda: n_gram_distribution_distance(real, ids, sim, ids, n=2),
            "ngram_3": lambda: n_gram_distribution_distance(real, ids, sim, ids, n=3),
            "aed": lambda: absolute_event_distribution_distance(real, ids, sim, ids),
            "ced": lambda: circadian_event_distribution_distance(real, ids, sim, ids),
            "red_dist": lambda: relative_event_distribution_distance(real, ids, sim, ids),
            "cwd": lambda: circadian_workforce_distribution_distance(real, ids, sim, ids),
            "car": lambda: case_arrival_distribution_distance(real, ids, sim, ids),
        }
        result = {name: calculate(name, fn, available) for name, fn in functions.items()}
        # Case boundaries include initial waiting, not just the first completed task.
        def cycle_frame(frame, case_ids, historical=False):
            if historical and real_bounds is not None:
                out = real_bounds.loc[real_bounds.index.isin(case_ids)].rename(columns={"min": "start_time", "max": "end_time"}).reset_index().rename(columns={"case:concept:name": "case_id"})
            elif not historical and statuses is not None:
                out = statuses[statuses.case_id.isin(case_ids)][["case_id", "arrival", "end"]].rename(columns={"arrival": "start_time", "end": "end_time"}).copy()
            elif frame is not None:
                out = frame.groupby("case_id").agg(start_time=("start_time", "min"), end_time=("end_time", "max")).reset_index()
            else:
                out = pd.DataFrame(columns=["case_id", "start_time", "end_time"])
            out["start_time"] = pd.to_datetime(out.start_time, utc=True)
            out["end_time"] = pd.to_datetime(out.end_time, utc=True)
            return out
        rc, sc = cycle_frame(real, real_ids, True), cycle_frame(sim, sim_ids)
        rcycles = (rc.end_time - rc.start_time).dt.total_seconds()
        scycles = (sc.end_time - sc.start_time).dt.total_seconds()
        for prefix, cycles in [("real", rcycles), ("sim", scycles)]:
            result[prefix + "_mean"] = float(cycles.mean()) if len(cycles) else None
            result[prefix + "_median"] = float(cycles.median()) if len(cycles) else None
        cycle_available = len(rc) > 0 and len(sc) > 0
        result["mae_cycle"] = calculate("mae_cycle", lambda: abs(result["real_mean"] - result["sim_mean"]), cycle_available)
        bin_size = timedelta(hours=1)
        result["ctd"] = calculate("ctd", lambda: cycle_time_distribution_distance(rc, ids, sc, ids, bin_size=bin_size) * bin_size.total_seconds(), cycle_available)
        result["wd"] = result["ctd"]
        window_available = window_real is not None and not window_real.empty and not window_sim.empty
        window_metrics = {
            "aed": calculate("window_aed", lambda: absolute_event_distribution_distance(window_real, ids, window_sim, ids), window_available),
            "ced": calculate("window_ced", lambda: circadian_event_distribution_distance(window_real, ids, window_sim, ids), window_available),
            "cwd": calculate("window_cwd", lambda: circadian_workforce_distribution_distance(window_real, ids, window_sim, ids), window_available),
        }
        result.update(mode=self.mode, cases=populations["simulated_scheduled"], events=len(df_sim), elapsed=elapsed,
                      populations=populations, metric_status=metric_status, window_metrics=window_metrics,
                      metric_units={"ctd": "seconds", "wd": "seconds", "mae_cycle": "seconds"},
                      completed_case_metrics_conditional=True,
                      generation_status=getattr(self, "generation_status", {"status": "unknown"}))
        lines = ["# Simulation benchmark", "", "Full-case metrics below are conditional on completion; censored and failed cases are excluded.",
                 "Replay uses paired completed IDs. Window metrics use observed intervals clipped at the cutoff; clipped endpoints are not recorded completions.", "",
                 "## Populations", "", "```json", json.dumps(populations, indent=2), "```", "",
                 "## Completed-case metrics", "", "| Metric | Value | Status |", "|---|---:|---|"]
        for name in list(functions) + ["ctd", "mae_cycle"]:
            val = result[name]
            display = format_dur(val) if name in ("ctd", "mae_cycle") else "N/A" if val is None else f"{val:.4f}"
            lines.append(f"| {name} | {display} | {metric_status[name]} |")
        lines.extend(["", "## Observation-window metrics", "", "```json", json.dumps(window_metrics, indent=2), "```",
                      "", "## Completed-case cycle times", "",
                      f"Mean: reference {format_dur(result['real_mean'])}; simulated {format_dur(result['sim_mean'])}.",
                      f"Median: reference {format_dur(result['real_median'])}; simulated {format_dur(result['sim_median'])}.",
                      "", "## Activity execution", "",
                      "| Activity | Completed tasks | Mean duration | Mean waiting | Mean worker queue |",
                      "|---|---:|---:|---:|---:|"])
        completed_events = df_sim[df_sim.complete_timestamp.notna()]
        for act, group in completed_events.groupby("activity"):
            lines.append(f"| {act} | {len(group)} | {format_dur(group.duration_seconds.mean())} | "
                         f"{format_dur(group.wait_time_seconds.mean())} | {format_dur(group.queue_wait_seconds.mean())} |")
        lines.extend(["", "Activity statistics above describe completed tasks in the observation window, including tasks in unfinished cases.",
                      "", "## Arrival generation", "", "```json", json.dumps(result["generation_status"], indent=2), "```"])
        out = os.path.join(PROJECT_ROOT, self.cfg.get("paths", {}).get("output_base_dir", "data/processed"))
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "simulation_report.md"), "w") as f:
            f.write("\n".join(lines) + "\n")
        with open(os.path.join(out, "simulation_metrics.json"), "w") as f:
            json.dump(result, f, indent=2, allow_nan=False)
        print("\n".join(lines))
        return result
