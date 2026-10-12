#!/usr/bin/env python3
"""
src/simulator/arrivals.py

Case Arrival Generator & Inter-Trigger Timing Engine.
Supports:
  1. Replay Mode: Replays exact historical arrival timelines, initial case states,
     and requested loan amounts from the aligned event log.
  2. Generative Mode: Generates synthetic arrivals using statistical distributions
     (exponential, uniform, constant), bounded by calendar shifts and empirical attributes.
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pm4py


class ArrivalGenerator:
    """
    Case Spawning Engine (Parity with Legacy InterTriggerTimer).
    Responsible for generating case entry schedules and payloads for the simulation.
    """
    def __init__(self, cfg, num_cases=None, arrival_mode="replay", calendar=None, duration_days=None):
        self.cfg = cfg
        self.num_cases = num_cases
        self.duration_days = duration_days if duration_days is not None else cfg.get("simulation", {}).get("duration_simulation")
        arr_cfg = cfg.get("simulation", {}).get("arrival", {})
        self.arrival_mode = arrival_mode or arr_cfg.get("mode", "replay")
        self.arr_cfg = arr_cfg
        self.calendar = calendar

    def prepare_arrivals(self):
        """
        Prepares case arrival schedule:
          - Replay Mode: Replays historical timestamps & case payloads from aligned XES log.
          - Generative Mode: Generates synthetic arrivals from distributions bounded by working calendar.
        Returns:
          tuple: (arrivals: list[dict], df_real: pd.DataFrame or None)
        """
        aligned_log_path = self.cfg["paths"]["aligned_log"]
        df_real = None
        empirical_amounts = []
        case_attrs = self.cfg.get("features", {}).get("case_attributes", [])
        amt_col = case_attrs[0]["column"] if (case_attrs and "column" in case_attrs[0]) else None

        if os.path.exists(aligned_log_path):
            try:
                df_real = pm4py.read_xes(aligned_log_path)
                if amt_col and amt_col in df_real.columns:
                    empirical_amounts = df_real[amt_col].dropna().astype(float).tolist()
            except Exception as e:
                print(f"[!] Warning: Could not read aligned log: {e}")

        arrivals = []

        if self.arrival_mode == "replay" and df_real is not None:
            print(f"Loading arrival timeline from historical log: {aligned_log_path}...")
            target_acts = set(self.cfg.get("target_activities", []))
            proc_prefixes = tuple(self.cfg.get("preprocessing", {}).get("process_activity_prefixes", []))
            ms_prefixes = tuple(self.cfg.get("preprocessing", {}).get("milestone_prefixes", []))

            if target_acts:
                w_mask = df_real["concept:name"].isin(target_acts)
            elif proc_prefixes:
                w_mask = df_real["concept:name"].str.startswith(proc_prefixes)
            else:
                w_mask = pd.Series(True, index=df_real.index)

            cases_with_w = set(df_real.loc[w_mask, "case:concept:name"].unique())
            first_times = df_real.groupby("case:concept:name")["time:timestamp"].min()
            valid_case_series = first_times[first_times.index.isin(cases_with_w)].sort_values()

            if self.duration_days and float(self.duration_days) > 0 and len(valid_case_series) > 0:
                t_start = valid_case_series.iloc[0]
                t_cutoff = t_start + pd.Timedelta(days=float(self.duration_days))
                valid_case_series = valid_case_series[valid_case_series <= t_cutoff]

            target_case_ids = list(valid_case_series.index)
            if self.num_cases and self.num_cases > 0:
                target_case_ids = target_case_ids[:self.num_cases]

            df_target = df_real[df_real["case:concept:name"].isin(set(target_case_ids))]
            case_grps = df_target.groupby("case:concept:name", sort=False)

            for cid, grp in case_grps:
                grp_sorted = grp.sort_values(by="time:timestamp")
                t_first = grp_sorted["time:timestamp"].iloc[0]
                amt = grp_sorted[amt_col].iloc[0] if (amt_col and amt_col in grp_sorted.columns) else 0.0

                ms_states = []
                cur_ms = set()
                if ms_prefixes:
                    for _, r in grp_sorted.iterrows():
                        act = str(r["concept:name"])
                        trans = str(r.get("lifecycle:transition", "COMPLETE")).upper()
                        if act.startswith(ms_prefixes):
                            cur_ms.add(act)
                        elif w_mask.get(r.name, False) and trans == "COMPLETE":
                            ms_states.append(set(cur_ms))

                init_ms = set()
                if ms_prefixes:
                    for _, r in grp_sorted.iterrows():
                        act = str(r["concept:name"])
                        if w_mask.get(r.name, False):
                            break
                        if act.startswith(ms_prefixes):
                            init_ms.add(act)

                arrivals.append({
                    "case_id": str(cid),
                    "arrival_time": t_first,
                    "requested_amount": float(amt) if pd.notna(amt) else 0.0,
                    "init_ms": init_ms,
                    "ms_states": ms_states
                })

            arrivals.sort(key=lambda x: x["arrival_time"])

        else:
            # Generative Arrival Mode (RIMS InterTriggerTimer Standard)
            n_cases = self.num_cases
            dur_seconds = float(self.duration_days * 86400.0) if (self.duration_days and float(self.duration_days) > 0) else None

            if n_cases:
                desc_str = f"{n_cases:,} cases"
            elif self.duration_days:
                desc_str = f"{self.duration_days} days ({dur_seconds:,.0f}s horizon)"
            else:
                desc_str = "1,000 cases"
            print(f"Generating synthetic case arrivals ({desc_str}) [Generative Arrival Mode]...")

            start_str = self.arr_cfg.get("start_timestamp", "2012-01-01 08:00:00")
            start_time = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
            cur_time = start_time
            if self.calendar is not None:
                self.calendar.set_ref_dt(cur_time)

            dist_name = self.arr_cfg.get("distribution", "exponential")
            scale = float(self.arr_cfg.get("scale", 300.0))
            calendar_bounded = self.arr_cfg.get("calendar_bounded", True)
            ms_prefixes = tuple(self.cfg.get("preprocessing", {}).get("milestone_prefixes", []))

            i = 0
            while True:
                if n_cases is not None and i >= n_cases:
                    break
                if dur_seconds is not None and (cur_time - start_time).total_seconds() >= dur_seconds:
                    break
                if n_cases is None and dur_seconds is None and i >= 1000:
                    break

                cid = f"synth_case_{i+1:05d}"
                if dist_name == "exponential":
                    interval = float(np.random.exponential(scale))
                elif dist_name == "uniform":
                    interval = float(np.random.uniform(scale * 0.5, scale * 1.5))
                else:
                    interval = scale

                cur_time += timedelta(seconds=interval)
                if calendar_bounded and self.calendar is not None:
                    gap = self.calendar.get_gap_to_next_work_window(cur_time)
                    if gap > 0:
                        cur_time += timedelta(seconds=gap)

                amt = float(random.choice(empirical_amounts)) if empirical_amounts else (float(random.randint(5000, 35000)) if amt_col else 0.0)
                cfg_init_ms = set(self.cfg.get("preprocessing", {}).get("initial_milestones", []))
                init_ms = cfg_init_ms if cfg_init_ms else ({"A_SUBMITTED", "A_PARTLYSUBMITTED"} if any(m.startswith("A_") for m in ms_prefixes) else set())
                arrivals.append({
                    "case_id": cid,
                    "arrival_time": cur_time,
                    "requested_amount": amt,
                    "init_ms": init_ms,
                    "ms_states": []
                })

        return arrivals, df_real
