#!/usr/bin/env python3
"""
================================================================================
RIMS+ AGNOSTIC CAUSAL DELAY & HUMAN LATENCY DISCOVERY ENGINE
File: src/02b_discover_delays.py

Combines Chapela-Campa & Dumas algorithms with RIMS+:
  1. Activity Instance Log Builder: Pairs START & COMPLETE lifecycle transitions.
  2. Concurrency Oracle: Detects true concurrency (overlap ratio >= ζ) to avoid
     false delays between parallel branches.
  3. Causal Predecessor Identification: Finds ε_cp as the non-overlapping,
     non-concurrent predecessor with maximum completion timestamp.
  4. Worker Busy & Calendar Timelines: Reconstructs worker execution schedules
     and non-working calendar intervals NW(ρ).
  5. Eclipse-Aware Estimators (Naive, EclAw, EclAw+): Computes extraneous delays
     un-eclipsed by worker busy tasks or off-duty shifts using midpoint extrapolation.
  6. RIMS+ Causal Orthogonality Tests:
     - WIP Independence Test: r(delay, WIP) < 0.30 confirms external uncoupled nature.
     - Calendar Invariance Test: Off-hours ratio >= 40% confirms 24/7 clock.
  7. Human Inter-Ticket Latency: Mines worker refractory cooldowns (5s - 2h).
  8. Parametric Fitting (KS-Test): Fits Lognormal, Gamma, and Exponential distributions.

Outputs:
  - models/delays/delay_manifest.json
  - models/delays/delay_discovery_report.md
================================================================================
"""

import os
import sys
import json
import time
import argparse
import warnings
import bisect
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats
import pm4py

# Project Path Setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
from config_loader import load_config
from simulator.calendar import CalendarManager

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =====================================================================
# 1. Distribution Fitting Helper
# =====================================================================
def fit_best_distribution(data_seconds):
    """
    Fits Lognormal, Gamma, and Exponential distributions to empirical delay data.
    Selects the champion distribution based on Kolmogorov-Smirnov (KS) test statistic.
    """
    data = np.array(data_seconds, dtype=np.float64)
    data = data[data > 0]
    if len(data) < 10:
        mean_val = float(np.mean(data)) if len(data) > 0 else 300.0
        med_val = float(np.median(data)) if len(data) > 0 else 300.0
        return {
            "distribution": "exponential",
            "mean_seconds": mean_val,
            "median_seconds": med_val,
            "p75_seconds": med_val,
            "p90_seconds": med_val,
            "params": {"loc": 0.0, "scale": mean_val}
        }

    candidates = {}

    # 1. Lognormal
    try:
        shape, loc, scale = stats.lognorm.fit(data, floc=0)
        ks_stat, _ = stats.kstest(data, "lognorm", args=(shape, loc, scale))
        candidates["lognormal"] = {
            "ks_stat": ks_stat,
            "params": {"shape": float(shape), "loc": float(loc), "scale": float(scale)}
        }
    except Exception:
        pass

    # 2. Gamma
    try:
        shape, loc, scale = stats.gamma.fit(data, floc=0)
        ks_stat, _ = stats.kstest(data, "gamma", args=(shape, loc, scale))
        candidates["gamma"] = {
            "ks_stat": ks_stat,
            "params": {"shape": float(shape), "loc": float(loc), "scale": float(scale)}
        }
    except Exception:
        pass

    # 3. Exponential
    try:
        loc, scale = stats.expon.fit(data, floc=0)
        ks_stat, _ = stats.kstest(data, "expon", args=(loc, scale))
        candidates["exponential"] = {
            "ks_stat": ks_stat,
            "params": {"loc": float(loc), "scale": float(scale)}
        }
    except Exception:
        pass

    if candidates:
        best_name = min(candidates.keys(), key=lambda k: candidates[k]["ks_stat"])
        best_params = candidates[best_name]["params"]
    else:
        best_name = "lognormal"
        best_params = {"shape": 1.0, "loc": 0.0, "scale": float(np.median(data))}

    return {
        "distribution": best_name,
        "mean_seconds": float(np.mean(data)),
        "median_seconds": float(np.median(data)),
        "p75_seconds": float(np.percentile(data, 75)),
        "p90_seconds": float(np.percentile(data, 90)),
        "params": best_params
    }


# =====================================================================
# 2. Concurrency Oracle & Causal Identification
# =====================================================================
class ConcurrencyOracle:
    """
    Computes pairwise activity concurrency.
    Two activities A and B are concurrent (A || B) if the proportion of times their
    execution intervals overlap relative to their trace co-occurrence meets or exceeds zeta.
    """
    def __init__(self, zeta=0.20):
        self.zeta = zeta
        self.concurrent_pairs = set()

    def fit(self, instances_by_case):
        co_occur = defaultdict(int)
        overlap = defaultdict(int)

        for case_id, insts in instances_by_case.items():
            unique_acts = list({x["act"] for x in insts})
            for i in range(len(unique_acts)):
                for j in range(i + 1, len(unique_acts)):
                    pair = tuple(sorted([unique_acts[i], unique_acts[j]]))
                    co_occur[pair] += 1

            # Sweep-line interval overlap check for activities with duration > 0
            timed_insts = [x for x in insts if x["start"] < x["end"]]
            if len(timed_insts) > 1:
                timed_insts.sort(key=lambda x: x["start"])
                trace_overlaps = set()
                for i in range(len(timed_insts)):
                    r1 = timed_insts[i]
                    for j in range(i + 1, len(timed_insts)):
                        r2 = timed_insts[j]
                        if r2["start"] >= r1["end"]:
                            break
                        if r1["act"] != r2["act"]:
                            pair = tuple(sorted([r1["act"], r2["act"]]))
                            trace_overlaps.add(pair)

                for pair in trace_overlaps:
                    overlap[pair] += 1

        self.concurrent_pairs = set()
        for pair, co in co_occur.items():
            ov = overlap.get(pair, 0)
            if co >= 20 and (ov / co) >= self.zeta:
                self.concurrent_pairs.add(pair)

        return self

    def are_concurrent(self, act_a, act_b):
        if act_a == act_b:
            return False
        return tuple(sorted([act_a, act_b])) in self.concurrent_pairs


# =====================================================================
# 3. Core Causal Delay Discovery Engine
# =====================================================================
class ChapelaDelayDiscoveryEngine:
    """
    RIMS+ Agnostic Causal Delay & Human Latency Discovery Engine.
    Fuses Chapela-Campa Concurrency Oracle & EclAw+ Estimators with
    RIMS+ WIP Orthogonality and Worker Refractory Cooldown Mining.
    """
    def __init__(self, cfg, estimator="eclaw_plus", placement="ex_ante",
                 min_delay_hours=1.0, min_samples=30, zeta=0.20, delta=0.05,
                 min_break_seconds=5.0):
        self.cfg = cfg
        self.estimator = estimator.lower()
        self.placement = placement.lower()
        self.min_delay_seconds = min_delay_hours * 3600.0
        self.min_samples = min_samples
        self.zeta = zeta
        self.delta = delta
        self.min_break_seconds = min_break_seconds

        self.log_path = cfg["paths"]["aligned_log"]
        self.output_dir = os.path.join(PROJECT_ROOT, "models/delays")
        os.makedirs(self.output_dir, exist_ok=True)
        self.calendar_mgr = CalendarManager.from_config(cfg)

    def run(self):
        print("\n" + "=" * 85)
        print("     RIMS+ CAUSAL DELAY & HUMAN LATENCY DISCOVERY ENGINE")
        print(f"     Algorithm: [{self.estimator.upper()}] | Placement: [{self.placement.upper()}] | ζ={self.zeta} | δ={self.delta}")
        print("=" * 85)

        start_time = time.time()
        print(f"Loading aligned event log: {self.log_path}...")
        df = pm4py.read_xes(self.log_path)
        print(f"Loaded {len(df):,} events across {df['case:concept:name'].nunique():,} cases.")

        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
        df = df.sort_values(by="time:timestamp").reset_index(drop=True)

        # -------------------------------------------------------------
        # STEP 1: Compute System-Wide WIP Time Series
        # -------------------------------------------------------------
        print("\n[Step 1/4] Building system-wide Work-In-Progress (WIP) time series...")
        case_starts = df.groupby("case:concept:name")["time:timestamp"].min()
        case_ends = df.groupby("case:concept:name")["time:timestamp"].max()

        start_events = pd.DataFrame({"time": case_starts.values, "change": 1})
        end_events = pd.DataFrame({"time": case_ends.values, "change": -1})
        wip_events = pd.concat([start_events, end_events]).sort_values(by="time").reset_index(drop=True)
        wip_events["wip"] = wip_events["change"].cumsum()

        wip_times = wip_events["time"].values
        wip_values = wip_events["wip"].values

        def lookup_wip(t):
            idx = np.searchsorted(wip_times, np.datetime64(t), side="right") - 1
            return int(wip_values[idx]) if idx >= 0 else 0

        # -------------------------------------------------------------
        # STEP 2: Extract Activity Instances & Fit Concurrency Oracle
        # -------------------------------------------------------------
        print("\n[Step 2/4] Extracting activity instances & building Concurrency Oracle...")
        start_events_map = defaultdict(list)
        instances = []

        for _, row in df.iterrows():
            c = str(row["case:concept:name"])
            a = str(row["concept:name"])
            trans = str(row.get("lifecycle:transition", "COMPLETE")).upper()
            t = row["time:timestamp"]
            r = str(row.get("org:resource", "UNKNOWN"))

            if trans == "START":
                start_events_map[(c, a)].append((t, r))
            elif trans == "COMPLETE":
                if start_events_map[(c, a)]:
                    t_s, r_s = start_events_map[(c, a)].pop(0)
                    res = r_s if r_s not in ("nan", "UNKNOWN", "None", "") else r
                    instances.append({"case": c, "act": a, "start": t_s, "end": t, "res": res})
                else:
                    instances.append({"case": c, "act": a, "start": t, "end": t, "res": r})

        print(f"Extracted {len(instances):,} activity instances.")
        instances_by_case = defaultdict(list)
        for inst in instances:
            instances_by_case[inst["case"]].append(inst)

        oracle = ConcurrencyOracle(zeta=self.zeta).fit(instances_by_case)
        print(f"Discovered {len(oracle.concurrent_pairs)} concurrent activity relations.")

        # Reconstruct Worker Timelines & Slicing Arrays
        resource_busy = defaultdict(list)
        for inst in instances:
            res = inst["res"]
            if res not in ("UNKNOWN", "nan", "None", "SYSTEM", "", "None") and inst["start"] < inst["end"]:
                resource_busy[res].append((inst["start"], inst["end"]))

        resource_busy_starts = {}
        for res in resource_busy:
            resource_busy[res].sort(key=lambda x: x[0])
            resource_busy_starts[res] = [b[0] for b in resource_busy[res]]

        print(f"Reconstructed active execution timelines for {len(resource_busy)} workers.")

        # -------------------------------------------------------------
        # STEP 3: Causal Predecessor & Extraneous Delay Estimation
        # -------------------------------------------------------------
        print(f"\n[Step 3/4] Estimating extraneous delays via Chapela-Campa [{self.estimator.upper()}]...")
        target_activities = set(self.cfg.get("target_activities", []))
        proc_prefixes = tuple(self.cfg.get("preprocessing", {}).get("process_activity_prefixes", []))

        cal = self.calendar_mgr.get_calendar(role_or_act=None)
        cal_enabled = cal.enabled
        work_days = cal.work_days
        h_start, h_end = cal.hour_start, cal.hour_end

        delay_samples_ex_ante = defaultdict(list)
        delay_samples_ex_post = defaultdict(list)
        delay_samples_arcs = defaultdict(list)

        total_causal_pairs = 0

        for case_id, insts in instances_by_case.items():
            insts_sorted = sorted(insts, key=lambda x: (x["start"], x["end"]))

            for idx, inst_t in enumerate(insts_sorted):
                act_t = inst_t["act"]
                is_target = True
                if target_activities and act_t not in target_activities:
                    is_target = False
                elif not target_activities and proc_prefixes and not act_t.startswith(proc_prefixes):
                    is_target = False

                if not is_target:
                    continue

                # Find valid causal predecessors
                preds = [
                    p for p in insts_sorted
                    if p["end"] <= inst_t["start"]
                    and not oracle.are_concurrent(p["act"], act_t)
                    and p != inst_t
                ]
                if not preds:
                    continue

                # Causal predecessor is the one with max end time
                inst_s = max(preds, key=lambda p: p["end"])
                act_s = inst_s["act"]
                S = inst_s["end"]
                E = inst_t["start"]
                raw_wait = (E - S).total_seconds()
                if raw_wait < 0:
                    continue

                total_causal_pairs += 1

                res_t = inst_t["res"]
                unavail = []

                # 1. Busy intervals of target worker
                busy_list = resource_busy.get(res_t, [])
                starts = resource_busy_starts.get(res_t, [])
                if busy_list:
                    idx_r = bisect.bisect_right(starts, E)
                    for b_s, b_e in busy_list[:idx_r]:
                        if b_e > S:
                            unavail.append((max(S, b_s), min(E, b_e)))

                # 2. Calendar non-working periods NW(ρ)
                cal_nw_sec = 0.0
                if cal_enabled:
                    cur_day = S.replace(hour=0, minute=0, second=0, microsecond=0)
                    end_day = E.replace(hour=0, minute=0, second=0, microsecond=0)
                    delta_days = (end_day - cur_day).days
                    for d in range(delta_days + 1):
                        c_day = cur_day + timedelta(days=d)
                        wday = c_day.weekday()
                        if wday not in work_days:
                            win_s = max(S, c_day)
                            win_e = min(E, c_day + timedelta(days=1))
                            if win_s < win_e:
                                unavail.append((win_s, win_e))
                                cal_nw_sec += (win_e - win_s).total_seconds()
                        else:
                            m_s = max(S, c_day)
                            m_e = min(E, c_day.replace(hour=h_start))
                            if m_s < m_e:
                                unavail.append((m_s, m_e))
                                cal_nw_sec += (m_e - m_s).total_seconds()
                            e_s = max(S, c_day.replace(hour=h_end))
                            e_e = min(E, c_day + timedelta(days=1))
                            if e_s < e_e:
                                unavail.append((e_s, e_e))
                                cal_nw_sec += (e_e - e_s).total_seconds()

                # 3. Merge unavailable intervals
                valid_unavail = [u for u in unavail if u[0] < u[1]]
                if not valid_unavail:
                    avail = [(S, E)] if raw_wait >= self.min_break_seconds else []
                else:
                    valid_unavail.sort(key=lambda x: x[0])
                    merged = []
                    c_s, c_e = valid_unavail[0]
                    for u_s, u_e in valid_unavail[1:]:
                        if u_s <= c_e:
                            c_e = max(c_e, u_e)
                        else:
                            merged.append((c_s, c_e))
                            c_s, c_e = u_s, u_e
                    merged.append((c_s, c_e))

                    avail = []
                    cursor = S
                    for u_s, u_e in merged:
                        if (u_s - cursor).total_seconds() >= self.min_break_seconds:
                            avail.append((cursor, u_s))
                        cursor = max(cursor, u_e)
                    if (E - cursor).total_seconds() >= self.min_break_seconds:
                        avail.append((cursor, E))

                # 4. Compute Extraneous Delay Estimator
                if self.estimator == "eclaw_plus":
                    if not avail:
                        omega_ex = 0.0
                    else:
                        tau_fat = avail[0][0]
                        tau_lat = avail[-1][1]
                        tau_fat_p = tau_fat - (tau_fat - S) / 2
                        tau_lat_p = tau_lat + (E - tau_lat) / 2
                        omega_ex = max(0.0, (tau_lat_p - tau_fat_p).total_seconds())

                elif self.estimator == "eclaw":
                    if not avail:
                        omega_ex = 0.0
                    else:
                        tau_fat = avail[0][0]
                        tau_lat = avail[-1][1]
                        omega_ex = max(0.0, (tau_lat - tau_fat).total_seconds())

                else:  # Naive
                    prior_busy_ends = [b[1] for b in busy_list if b[1] <= E]
                    tau_rat = max(prior_busy_ends) if prior_busy_ends else S
                    omega_ex = max(0.0, (E - max(tau_rat, S)).total_seconds())

                off_ratio = cal_nw_sec / max(1.0, raw_wait)
                wip_at_s = lookup_wip(S)

                record = {
                    "delay_seconds": omega_ex,
                    "raw_wait_seconds": raw_wait,
                    "wip": wip_at_s,
                    "off_calendar_ratio": off_ratio,
                    "trigger_act": act_s,
                    "target_act": act_t
                }

                delay_samples_ex_ante[act_t].append(record)
                delay_samples_ex_post[act_s].append(record)
                delay_samples_arcs[(act_s, act_t)].append(record)

        print(f"Evaluated {total_causal_pairs:,} causally consecutive instance pairs.")

        # -------------------------------------------------------------
        # STEP 4: Attribution, Causal Validation & Manifest Generation
        # -------------------------------------------------------------
        if self.placement == "ex_ante":
            target_groups = {act: samples for act, samples in delay_samples_ex_ante.items()}
        elif self.placement == "ex_post":
            target_groups = {act: samples for act, samples in delay_samples_ex_post.items()}
        else:
            target_groups = {f"{a}->{b}": samples for (a, b), samples in delay_samples_arcs.items()}

        discovered_delays = []
        for key, samples in target_groups.items():
            if len(samples) < self.min_samples:
                continue

            delays = np.array([s["delay_seconds"] for s in samples])
            pos_delays = delays[delays > 0]
            pos_ratio = len(pos_delays) / len(delays)

            # Chapela Outlier Threshold Filter (δ)
            if pos_ratio < self.delta:
                continue

            median_d = float(np.median(pos_delays)) if len(pos_delays) > 0 else 0.0
            if median_d < self.min_delay_seconds:
                continue

            wips = np.array([s["wip"] for s in samples])
            off_ratios = np.array([s["off_calendar_ratio"] for s in samples])

            # RIMS+ WIP Orthogonality Test
            if np.std(wips) > 0 and np.std(delays) > 0:
                corr, _ = stats.pearsonr(wips, delays)
            else:
                corr = 0.0

            # RIMS+ Calendar Invariance Test (24/7 Clock Test)
            mean_off_ratio = float(np.mean(off_ratios))
            is_eud = (abs(corr) < 0.30) and (mean_off_ratio >= 0.40)
            clock_type = "24_7" if mean_off_ratio >= 0.40 else "business_shift"
            classification = "External Uncoupled Delay (EUD)" if is_eud else "Internal Worklist / Queue Delay"

            fit_result = fit_best_distribution(pos_delays)

            pred_counts = defaultdict(int)
            for s in samples:
                pred_counts[s["trigger_act"]] += 1
            top_preds = sorted(pred_counts.items(), key=lambda x: -x[1])

            entry = {
                "activity": key,
                "placement": self.placement,
                "trigger_activity": top_preds[0][0] if top_preds else key,
                "resume_activity": key,
                "sample_count": len(samples),
                "positive_delay_ratio": round(float(pos_ratio), 4),
                "classification": classification,
                "is_external": bool(is_eud),
                "clock_type": clock_type,
                "causal_evidence": {
                    "wip_correlation": round(float(corr), 4),
                    "mean_off_calendar_ratio": round(mean_off_ratio, 4),
                },
                "median_hours": round(float(median_d / 3600.0), 2),
                "median_days": round(float(median_d / 86400.0), 2),
                "mean_days": round(float(np.mean(pos_delays) / 86400.0), 2),
                "best_distribution": fit_result["distribution"],
                "distribution_params": fit_result["params"],
                "top_causal_predecessors": [p[0] for p in top_preds[:3]]
            }
            discovered_delays.append(entry)

        discovered_delays.sort(key=lambda x: -x["median_days"])

        # Also store causal transition arcs for granular inspection
        arc_manifest_entries = []
        for (act_a, act_b), samples in delay_samples_arcs.items():
            if len(samples) < self.min_samples:
                continue
            delays = np.array([s["delay_seconds"] for s in samples])
            pos = delays[delays > 0]
            if len(pos) / len(delays) < self.delta:
                continue
            med = float(np.median(pos)) if len(pos) > 0 else 0.0
            if med < self.min_delay_seconds:
                continue
            fit_r = fit_best_distribution(pos)
            arc_manifest_entries.append({
                "trigger_activity": act_a,
                "resume_activity": act_b,
                "sample_count": len(samples),
                "median_days": round(float(med / 86400.0), 2),
                "best_distribution": fit_r["distribution"],
                "distribution_params": fit_r["params"]
            })

        # -------------------------------------------------------------
        # STEP 5: Human Inter-Ticket Latency (Worker Refractory Mining)
        # -------------------------------------------------------------
        print("\n[Step 4/4] Mining Human Inter-Ticket Latency (Worker Refractory Gaps)...")
        df_shifts = df[df["lifecycle:transition"].isin(["START", "COMPLETE"])].sort_values(by="time:timestamp")

        res_events = defaultdict(list)
        for _, r in df_shifts.iterrows():
            res = str(r["org:resource"])
            if res in ("nan", "NONE", "", "None", "UNKNOWN", "SYSTEM"):
                continue
            res_events[res].append((r["lifecycle:transition"], r["time:timestamp"]))

        worker_gaps = defaultdict(list)
        all_human_gaps = []

        for res, evs in res_events.items():
            for i in range(len(evs) - 1):
                tr1, t1 = evs[i]
                tr2, t2 = evs[i + 1]
                if tr1 == "COMPLETE" and tr2 == "START":
                    gap = (t2 - t1).total_seconds()
                    if 5.0 <= gap <= 7200.0 and t1.date() == t2.date():
                        worker_gaps[res].append(gap)
                        all_human_gaps.append(gap)

        print(f"Extracted {len(all_human_gaps):,} intra-day human inter-ticket transitions across {len(worker_gaps)} workers.")
        global_cooldown_fit = fit_best_distribution(all_human_gaps) if all_human_gaps else {
            "distribution": "exponential",
            "median_seconds": 180.0,
            "mean_seconds": 300.0,
            "params": {"loc": 0.0, "scale": 300.0}
        }

        worker_profiles = {}
        for res, gaps in worker_gaps.items():
            if len(gaps) >= 50:
                worker_profiles[res] = {
                    "sample_count": len(gaps),
                    "median_seconds": round(float(np.median(gaps)), 1),
                    "mean_seconds": round(float(np.mean(gaps)), 1),
                    **fit_best_distribution(gaps)
                }

        # -------------------------------------------------------------
        # STEP 6: Save Delay Manifest & Report
        # -------------------------------------------------------------
        manifest = {
            "source_log": self.log_path,
            "discovery_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "engine": "RIMS+ & Chapela-Campa Causal Delay",
            "estimator": self.estimator,
            "placement": self.placement,
            "concurrency_zeta": self.zeta,
            "outlier_delta": self.delta,
            "external_uncoupled_delays": [d for d in discovered_delays if d["is_external"]],
            "internal_queue_delays": [d for d in discovered_delays if not d["is_external"]],
            "causal_transition_arcs": arc_manifest_entries,
            "human_inter_ticket_latency": {
                "enabled": True,
                "global_median_seconds": round(global_cooldown_fit["median_seconds"], 1),
                "global_mean_seconds": round(global_cooldown_fit["mean_seconds"], 1),
                "global_distribution": global_cooldown_fit["distribution"],
                "global_params": global_cooldown_fit["params"],
                "worker_count": len(worker_profiles),
                "per_worker_profiles": worker_profiles
            }
        }

        json_out_path = os.path.join(self.output_dir, "delay_manifest.json")
        with open(json_out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n[✓] Saved Delay Manifest to: {json_out_path}")

        report_out_path = os.path.join(self.output_dir, "delay_discovery_report.md")
        self._write_report(report_out_path, manifest, discovered_delays, all_human_gaps)
        print(f"[✓] Saved Discovery Report to: {report_out_path}")

        elapsed = time.time() - start_time
        print(f"[✓] Discovery completed in {elapsed:.2f} seconds.")
        return manifest

    def _write_report(self, path, manifest, delays, all_gaps):
        """Generates comprehensive Markdown report."""
        with open(path, "w") as f:
            f.write("# RIMS+ Chapela-Campa Extraneous Delay & Human Latency Discovery Report\n\n")
            f.write(f"- **Source Log:** `{manifest['source_log']}`\n")
            f.write(f"- **Algorithm:** `{manifest['estimator'].upper()}` (Eclipse-Aware Extrapolation)\n")
            f.write(f"- **Placement:** `{manifest['placement'].upper()}`\n")
            f.write(f"- **Concurrency Threshold (ζ):** `{manifest['concurrency_zeta']}`\n")
            f.write(f"- **Outlier Threshold (δ):** `{manifest['outlier_delta']}`\n")
            f.write(f"- **Generated:** `{manifest['discovery_timestamp']}`\n\n")

            f.write("## 1. Discovered External Uncoupled Delays (EUD)\n\n")
            f.write("These activity delays passed both the Chapela-Campa un-eclipsed availability test and RIMS+ WIP orthogonality ($|r| < 0.30$):\n\n")

            f.write("| Activity | Placement | Positive Ratio | Samples | Median Delay | WIP Corr ($r$) | Clock Type | Best Fit Distribution |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for d in manifest["external_uncoupled_delays"]:
                f.write(f"| `{d['activity']}` | `{d['placement']}` | {d['positive_delay_ratio']*100:.1f}% | {d['sample_count']:,} | **{d['median_days']} days** ({d['median_hours']}h) | {d['causal_evidence']['wip_correlation']:.3f} | `{d['clock_type']}` | `{d['best_distribution']}` |\n")

            f.write("\n## 2. Human Inter-Ticket Latency (Worker Refractory Cooldowns)\n\n")
            hl = manifest["human_inter_ticket_latency"]
            f.write(f"- **Total Intra-Day Work Gaps Mined:** {len(all_gaps):,} consecutive tasks\n")
            f.write(f"- **Global Median Cooldown:** **{hl['global_median_seconds']} seconds** ({hl['global_median_seconds']/60:.1f} mins)\n")
            f.write(f"- **Global Mean Cooldown:** **{hl['global_mean_seconds']} seconds** ({hl['global_mean_seconds']/60:.1f} mins)\n")
            f.write(f"- **Champion Fit Distribution:** `{hl['global_distribution']}`\n")
            f.write(f"- **Profiled Workers:** {hl['worker_count']} human agents\n")


# =====================================================================
# CLI Entry Point
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="RIMS+ Chapela-Campa Causal Delay & Human Latency Discovery Engine")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--estimator", type=str, default=None, choices=["eclaw_plus", "eclaw", "naive"],
                        help="Delay estimator: 'eclaw_plus' (extrapolated eclipse-aware), 'eclaw', or 'naive'")
    parser.add_argument("--placement", type=str, default=None, choices=["ex_ante", "ex_post", "causal_pair"],
                        help="Delay placement strategy: 'ex_ante' (before target), 'ex_post' (after source), or 'causal_pair'")
    parser.add_argument("--min_delay_hours", type=float, default=1.0, help="Minimum median delay threshold in hours")
    parser.add_argument("--min_samples", type=int, default=30, help="Minimum sample count")
    parser.add_argument("--zeta", type=float, default=0.20, help="Concurrency overlap threshold ζ")
    parser.add_argument("--delta", type=float, default=0.05, help="Outlier positive delay proportion threshold δ")
    args = parser.parse_args()

    cfg = load_config(args.config)
    delays_cfg = cfg.get("simulation", {}).get("extraneous_delays", {})
    estimator = args.estimator or delays_cfg.get("estimator", "eclaw_plus")
    placement = args.placement or delays_cfg.get("placement", "ex_ante")

    engine = ChapelaDelayDiscoveryEngine(
        cfg,
        estimator=estimator,
        placement=placement,
        min_delay_hours=args.min_delay_hours,
        min_samples=args.min_samples,
        zeta=args.zeta,
        delta=args.delta
    )
    engine.run()


if __name__ == "__main__":
    main()