#!/usr/bin/env python3
"""
src/activity_profiler.py

Universal Data-Driven Activity Profiler for RIMS++.
Automatically profiles and classifies process activities based on historical execution statistics:
  1. HUMAN_WORK_ITEM: Positive execution duration, requires human worker capacity, subject to calendar hours.
  2. AUTOMATED_TASK: Instantaneous execution (duration == 0s), runs 24/7 without human worker capacity,
                     retains external / batch waiting time delay.
  3. INSTANT_GATEWAY: Instantaneous execution and negligible wait time (duration == 0s, wait <= 60s),
                      advances Petri net marking immediately.

Classification Rule:
  If p_zero_duration >= zero_duration_threshold (default 0.95):
      If median_wait_time > wait_threshold (default 60.0s):
          -> AUTOMATED_TASK
      Else:
          -> INSTANT_GATEWAY
  Else:
      -> HUMAN_WORK_ITEM
"""

import os
import json
import argparse
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
import pm4py

try:
    from config_loader import load_config, PROJECT_ROOT
except ImportError:
    from src.config_loader import load_config, PROJECT_ROOT


NON_HUMAN_RESOURCES = {"SYSTEM", "SYNTHETIC", "NONE", "NAN", "UNKNOWN", "", "NONE"}


class ActivityProfiler:
    def __init__(self, cfg=None, zero_duration_threshold=0.95, min_automated_wait_seconds=60.0, overrides=None):
        self.cfg = cfg or {}
        prof_cfg = self.cfg.get("activity_profiling", {})
        self.zero_duration_threshold = float(prof_cfg.get("zero_duration_threshold", zero_duration_threshold))
        self.min_automated_wait_seconds = float(prof_cfg.get("min_automated_wait_seconds", min_automated_wait_seconds))
        raw_overrides = prof_cfg.get("overrides", overrides or {})
        # Normalize override keys and values
        self.overrides = {}
        if isinstance(raw_overrides, dict):
            for k, v in raw_overrides.items():
                self.overrides[str(k)] = str(v).upper()

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg=cfg)

    def profile_dataframe(self, df: pd.DataFrame) -> dict:
        """
        Profiles a pandas DataFrame of process events containing:
        'case:concept:name', 'concept:name', 'time:timestamp', and optional 'lifecycle:transition', 'org:resource'.
        """
        c_col = "case:concept:name" if "case:concept:name" in df.columns else "case_id"
        a_col = "concept:name" if "concept:name" in df.columns else "activity"
        t_col = "time:timestamp" if "time:timestamp" in df.columns else "start_timestamp"
        r_col = "org:resource" if "org:resource" in df.columns else "resource"
        lc_col = "lifecycle:transition" if "lifecycle:transition" in df.columns else None

        df_sorted = df.copy()
        df_sorted[t_col] = pd.to_datetime(df_sorted[t_col], utc=True)
        if lc_col:
            trans_order = {"START": 0, "COMPLETE": 1}
            df_sorted["_tr"] = df_sorted[lc_col].astype(str).str.upper().map(trans_order).fillna(2)
            df_sorted = df_sorted.sort_values(by=[t_col, "_tr"]).drop(columns=["_tr"]).reset_index(drop=True)
        else:
            df_sorted = df_sorted.sort_values(by=t_col).reset_index(drop=True)

        # Track durations and waiting times per activity
        act_durations = defaultdict(list)
        act_wait_times = defaultdict(list)
        act_resources = defaultdict(list)
        act_starts_seen = defaultdict(int)
        act_completes_seen = defaultdict(int)

        pending_starts = defaultdict(list)  # (case, act) -> [(timestamp, res)]
        case_last_complete = {}

        cols_to_use = [c for c in [c_col, a_col, t_col, r_col, lc_col] if c and c in df_sorted.columns]
        for row in df_sorted[cols_to_use].to_dict("records"):
            cid = str(row[c_col])
            act = str(row[a_col])
            ts = row[t_col]
            res = str(row.get(r_col, "UNKNOWN")) if r_col in row and pd.notna(row.get(r_col)) else "UNKNOWN"
            trans = str(row.get(lc_col, "COMPLETE")).upper() if lc_col else "COMPLETE"

            if trans == "START":
                act_starts_seen[act] += 1
                # Waiting time: from previous complete in trace
                if cid in case_last_complete:
                    w = (ts - case_last_complete[cid]).total_seconds()
                    if w >= 0:
                        act_wait_times[act].append(w)
                else:
                    act_wait_times[act].append(0.0)

                pending_starts[(cid, act)].append((ts, res))

            elif trans == "COMPLETE":
                act_completes_seen[act] += 1
                act_resources[act].append(res)

                if pending_starts[(cid, act)]:
                    s_ts, _ = pending_starts[(cid, act)].pop(0)
                    dur = max(0.0, (ts - s_ts).total_seconds())
                    act_durations[act].append(dur)
                else:
                    # Single-timestamp event (no START) -> duration is 0.0s
                    act_durations[act].append(0.0)
                    # Waiting time for single-timestamp event
                    if cid in case_last_complete:
                        w = (ts - case_last_complete[cid]).total_seconds()
                        if w >= 0:
                            act_wait_times[act].append(w)
                    else:
                        act_wait_times[act].append(0.0)

                case_last_complete[cid] = ts

        # Also capture any unmatched starts (duration = 0 fallback)
        for (cid, act), start_list in pending_starts.items():
            for _ in start_list:
                act_durations[act].append(0.0)

        all_activities = sorted(list(set(list(act_completes_seen.keys()) + list(act_starts_seen.keys()))))
        profiles = {}
        activity_types = {}

        for act in all_activities:
            durs = act_durations.get(act, [0.0])
            waits = act_wait_times.get(act, [0.0])
            res_list = act_resources.get(act, [])

            total_events = len(durs)
            zero_dur_events = sum(1 for d in durs if d <= 0.0)
            p_zero = zero_dur_events / max(1, total_events)

            med_dur = float(np.median(durs)) if durs else 0.0
            med_wait = float(np.median(waits)) if waits else 0.0

            human_res_count = sum(1 for r in res_list if r.upper() not in NON_HUMAN_RESOURCES)
            human_res_ratio = human_res_count / max(1, len(res_list))

            # Classification
            if act in self.overrides:
                cls_type = self.overrides[act]
                if cls_type not in ("HUMAN_WORK_ITEM", "AUTOMATED_TASK", "INSTANT_GATEWAY"):
                    cls_type = "HUMAN_WORK_ITEM"
            elif p_zero >= self.zero_duration_threshold:
                if med_wait > self.min_automated_wait_seconds:
                    cls_type = "AUTOMATED_TASK"
                else:
                    cls_type = "INSTANT_GATEWAY"
            else:
                cls_type = "HUMAN_WORK_ITEM"

            activity_types[act] = cls_type
            profiles[act] = {
                "type": cls_type,
                "total_events": total_events,
                "p_zero_duration": round(float(p_zero), 4),
                "median_duration_seconds": round(float(med_dur), 2),
                "median_wait_seconds": round(float(med_wait), 2),
                "human_resource_ratio": round(float(human_res_ratio), 4),
                "has_start_lifecycle": bool(act_starts_seen.get(act, 0) > 0)
            }

        result = {
            "__metadata__": {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "zero_duration_threshold": self.zero_duration_threshold,
                "min_automated_wait_seconds": self.min_automated_wait_seconds,
                "total_activities": len(all_activities),
                "summary": {
                    "human_work_items": sum(1 for t in activity_types.values() if t == "HUMAN_WORK_ITEM"),
                    "automated_tasks": sum(1 for t in activity_types.values() if t == "AUTOMATED_TASK"),
                    "instant_gateways": sum(1 for t in activity_types.values() if t == "INSTANT_GATEWAY")
                }
            },
            "activity_types": activity_types,
            "profiles": profiles
        }
        return result

    def profile_log_file(self, log_path: str) -> dict:
        """Loads an XES or CSV log and computes activity profiles."""
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Log file not found: {log_path}")

        print(f"Profiling event log: {log_path}...")
        if log_path.endswith(".xes") or log_path.endswith(".xes.gz"):
            df = pm4py.read_xes(log_path)
        else:
            df = pd.read_csv(log_path)

        return self.profile_dataframe(df)

    def save_profiles(self, profiles_data: dict, output_file: str, report_file: str = None):
        """Saves JSON profiles and markdown documentation report."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(profiles_data, f, indent=2)
        print(f"[✓] Saved Activity Profiles to: {output_file}")

        if report_file:
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            with open(report_file, "w") as f:
                f.write("# RIMS++ Activity Behavioral Profiling Report\n\n")
                meta = profiles_data.get("__metadata__", {})
                f.write(f"- **Generated At:** {meta.get('generated_at')}\n")
                f.write(f"- **Zero-Duration Threshold:** `{meta.get('zero_duration_threshold') * 100:.1f}%`\n")
                f.write(f"- **Automated Delay Threshold:** `{meta.get('min_automated_wait_seconds')}s`\n")
                summary = meta.get("summary", {})
                f.write(f"- **Total Activities:** {meta.get('total_activities')}\n")
                f.write(f"  - **Human Work Items:** {summary.get('human_work_items')}\n")
                f.write(f"  - **Automated Tasks with Delay:** {summary.get('automated_tasks')}\n")
                f.write(f"  - **Instant Gateways / Milestones:** {summary.get('instant_gateways')}\n\n")

                f.write("## Activity Breakdown\n\n")
                f.write("| Activity | Profile Type | Total Events | % Zero Duration | Med Duration | Med Wait | % Human Res |\n")
                f.write("|---|---|---|---|---|---|---|\n")
                for act, p in profiles_data.get("profiles", {}).items():
                    f.write(f"| `{act}` | **{p['type']}** | {p['total_events']:,} | {p['p_zero_duration']*100:.1f}% | {p['median_duration_seconds']:.1f}s | {p['median_wait_seconds']:.1f}s | {p['human_resource_ratio']*100:.1f}% |\n")
            print(f"[✓] Saved Profiling Report to: {report_file}")


def load_or_discover_activity_profiles(cfg: dict, force_retrain: bool = False) -> dict:
    """
    Helper function used by DatasetBuilder, ChampionPredictor, and Process.
    Loads models/activity_profiles.json if present; otherwise runs discovery automatically.
    """
    models_dir = os.path.join(PROJECT_ROOT, cfg.get("paths", {}).get("models_dir", "models"))
    default_out = os.path.join(models_dir, "activity_profiles.json")
    default_rep = os.path.join(models_dir, "activity_profiles_report.md")

    prof_cfg = cfg.get("activity_profiling", {})
    output_path = os.path.join(PROJECT_ROOT, prof_cfg.get("output_file", default_out))
    report_path = os.path.join(PROJECT_ROOT, prof_cfg.get("report_file", default_rep))

    if os.path.exists(output_path) and not force_retrain:
        with open(output_path, "r") as f:
            return json.load(f)

    # Automatically discover from log
    log_path = os.path.join(PROJECT_ROOT, cfg.get("paths", {}).get("aligned_log", cfg.get("paths", {}).get("raw_log")))
    if not os.path.exists(log_path):
        raw_log = os.path.join(PROJECT_ROOT, cfg.get("paths", {}).get("raw_log"))
        log_path = raw_log if os.path.exists(raw_log) else log_path

    profiler = ActivityProfiler.from_config(cfg)
    data = profiler.profile_log_file(log_path)
    profiler.save_profiles(data, output_path, report_path)
    return data


def main():
    parser = argparse.ArgumentParser(description="RIMS++ Activity Behavioral Profiling Engine")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", type=str, default=None, help="Explicit log path to profile")
    parser.add_argument("--force", action="store_true", help="Force recalculation of profiles")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_path = args.log or os.path.join(PROJECT_ROOT, cfg.get("paths", {}).get("aligned_log", cfg.get("paths", {}).get("raw_log")))

    profiler = ActivityProfiler.from_config(cfg)
    data = profiler.profile_log_file(log_path)

    models_dir = os.path.join(PROJECT_ROOT, cfg.get("paths", {}).get("models_dir", "models"))
    out_file = os.path.join(models_dir, "activity_profiles.json")
    rep_file = os.path.join(models_dir, "activity_profiles_report.md")
    profiler.save_profiles(data, out_file, rep_file)


if __name__ == "__main__":
    main()

