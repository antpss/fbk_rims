#!/usr/bin/env python3
"""
src/simulator/arrivals.py

Case Arrival Generator & Inter-Trigger Timing Engine.
Supports:
  1. Replay Mode: Replays exact historical arrival timelines, initial case states,
     and requested loan amounts from the aligned event log.
  2. Generative Mode: Generates synthetic arrivals using statistical distributions
     (exponential, uniform, constant), bounded by calendar shifts and empirical attributes.
  3. Prophet Mode: Models case arrival dynamics as an hourly time-series prediction
     problem using Facebook Prophet (trend + seasonality) with intra-hour dispersion
     (Camargo et al. DeepSimulator parity).
"""

import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pm4py

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))


class ArrivalGenerator:
    """
    Case Spawning Engine (Parity with Legacy InterTriggerTimer).
    Responsible for generating case entry schedules and payloads for the simulation.
    """
    def __init__(self, cfg, num_cases=None, arrival_mode="replay", calendar=None, duration_days=None, force_retrain_prophet=False):
        self.cfg = cfg
        self.num_cases = num_cases
        self.duration_days = duration_days if duration_days is not None else cfg.get("simulation", {}).get("duration_simulation")
        arr_cfg = cfg.get("simulation", {}).get("arrival", {})
        self.arrival_mode = arrival_mode or arr_cfg.get("mode", "replay")
        self.arr_cfg = arr_cfg
        self.calendar = calendar
        self.force_retrain_prophet = force_retrain_prophet

    def prepare_arrivals(self):
        """
        Prepares case arrival schedule:
          - Replay Mode: Replays historical timestamps & case payloads from aligned XES log.
          - Generative Mode: Generates synthetic arrivals from distributions bounded by working calendar.
          - Prophet Mode: Generates synthetic arrivals forecasting hourly intensity via Facebook Prophet.
        Returns:
          tuple: (arrivals: list[dict], df_real: pd.DataFrame or None)
        """
        aligned_log_path = self.cfg["paths"]["aligned_log"]
        df_real = None
        empirical_amounts = []
        case_attrs = self.cfg.get("features", {}).get("case_attributes", [])
        amt_col = case_attrs[0]["column"] if (case_attrs and "column" in case_attrs[0]) else None

        attr_imputation = {}
        case_payload_records = []
        if os.path.exists(aligned_log_path):
            try:
                df_real = pm4py.read_xes(aligned_log_path)
                if amt_col and amt_col in df_real.columns:
                    empirical_amounts = df_real[amt_col].dropna().astype(float).tolist()

                for attr in case_attrs:
                    col = attr["column"]
                    clean_col = col.replace("case:", "")
                    series = df_real[col] if col in df_real.columns else (df_real[clean_col] if clean_col in df_real.columns else pd.Series(dtype=float))
                    is_cont = (attr.get("type") == "continuous")
                    treat_zero = attr.get("treat_zero_as_missing", False)
                    if is_cont:
                        clean_s = pd.to_numeric(series, errors="coerce")
                        if treat_zero:
                            clean_s = clean_s[clean_s != 0]
                        clean_s = clean_s.dropna()
                        val_stat = float(clean_s.median()) if not clean_s.empty else float(attr.get("fill_value", 0.0))
                    else:
                        clean_s = series.dropna()
                        val_stat = str(clean_s.mode().iloc[0]) if not clean_s.empty else str(attr.get("fill_value", "UNKNOWN"))
                    attr_imputation[attr["feature_name"]] = val_stat

                case_first_rows = df_real.groupby("case:concept:name", sort=False).first().reset_index()
                for _, r in case_first_rows.iterrows():
                    rec = {}
                    for attr in case_attrs:
                        col = attr["column"]
                        clean_col = col.replace("case:", "")
                        feat_name = attr["feature_name"]
                        default_val = attr_imputation.get(feat_name, attr.get("fill_value", 0.0))
                        val = r.get(col, r.get(clean_col, default_val))
                        if pd.isna(val):
                            val = default_val
                        is_cont = (attr.get("type") == "continuous")
                        treat_zero = attr.get("treat_zero_as_missing", False)
                        if is_cont:
                            try:
                                val = float(val)
                                if treat_zero and val == 0.0:
                                    val = float(default_val)
                            except Exception:
                                val = float(default_val)
                        else:
                            val = str(val) if val is not None else str(default_val)
                        rec[feat_name] = val
                    if rec:
                        case_payload_records.append(rec)
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

                extracted_case_attrs = {}
                for attr_def in case_attrs:
                    c_col = attr_def.get("column")
                    feat_name = attr_def.get("feature_name", c_col)
                    default_val = attr_imputation.get(feat_name, attr_def.get("fill_value", 0.0))
                    clean_col = c_col.replace("case:", "")
                    val = grp_sorted[c_col].iloc[0] if (c_col and c_col in grp_sorted.columns) else (grp_sorted[clean_col].iloc[0] if clean_col in grp_sorted.columns else default_val)
                    if pd.isna(val):
                        val = default_val

                    is_cont = (attr_def.get("type") == "continuous")
                    treat_zero = attr_def.get("treat_zero_as_missing", False)
                    if is_cont:
                        try:
                            val = float(val)
                            if treat_zero and val == 0.0:
                                val = float(default_val)
                        except Exception:
                            val = float(default_val)
                    else:
                        val = str(val) if val is not None else str(default_val)
                    extracted_case_attrs[feat_name] = val

                amt = float(extracted_case_attrs.get("RequestedAmount", amt))

                arrivals.append({
                    "case_id": str(cid),
                    "arrival_time": t_first,
                    "requested_amount": float(amt) if pd.notna(amt) else 0.0,
                    "case_attrs": extracted_case_attrs,
                    "init_ms": init_ms,
                    "ms_states": ms_states
                })

            arrivals.sort(key=lambda x: x["arrival_time"])

        elif self.arrival_mode == "prophet":
            arrivals = self._prepare_prophet_arrivals(df_real, empirical_amounts, amt_col, case_payload_records=case_payload_records)

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

                if case_payload_records:
                    sampled_payload = dict(random.choice(case_payload_records))
                    amt = float(sampled_payload.get("RequestedAmount", 0.0))
                else:
                    amt = float(random.choice(empirical_amounts)) if empirical_amounts else 0.0
                    sampled_payload = {"RequestedAmount": amt} if amt_col else {}

                init_dist = self.cfg.get("preprocessing", {}).get("initial_milestones_distribution", [])
                if init_dist:
                    cand_sets = [set(entry["milestones"]) for entry in init_dist]
                    probs = [float(entry["probability"]) for entry in init_dist]
                    init_ms = random.choices(cand_sets, weights=probs, k=1)[0]
                else:
                    cfg_init_ms = set(self.cfg.get("preprocessing", {}).get("initial_milestones", []))
                    init_ms = cfg_init_ms if cfg_init_ms else ({"A_SUBMITTED", "A_PARTLYSUBMITTED"} if any(m.startswith("A_") for m in ms_prefixes) else set())
                arrivals.append({
                    "case_id": cid,
                    "arrival_time": cur_time,
                    "requested_amount": amt,
                    "case_attrs": sampled_payload,
                    "init_ms": init_ms,
                    "ms_states": []
                })
                i += 1

        # Compute case inter-arrival time in seconds across sorted arrivals
        if arrivals:
            arrivals.sort(key=lambda x: x["arrival_time"])
            for idx in range(len(arrivals)):
                if idx == 0:
                    arrivals[idx]["inter_arrival_seconds"] = 0.0
                else:
                    delta = (arrivals[idx]["arrival_time"] - arrivals[idx - 1]["arrival_time"]).total_seconds()
                    arrivals[idx]["inter_arrival_seconds"] = max(0.0, float(delta))

        return arrivals, df_real

    def _prepare_prophet_arrivals(self, df_real, empirical_amounts, amt_col, case_payload_records=None):
        """
        Generates synthetic arrivals using Facebook Prophet time-series modeling (Camargo et al. DeepSimulator parity).
        Supports:
          - Bounded Logistic Growth (with dynamic capacity) or Linear Growth.
          - Uncertainty-aware stochastic sampling (Triangular, Poisson, or Round).
          - Smart Model Caching: Serializes fitted Prophet model to JSON for instant sub-second reuse.
        """
        import logging
        import warnings
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        warnings.filterwarnings("ignore", category=FutureWarning)

        try:
            from prophet import Prophet
            from prophet.serialize import model_to_json, model_from_json
        except ImportError:
            raise ImportError(
                "Facebook Prophet is required for --arrival_mode prophet. "
                "Install it using: pip install prophet"
            )

        if df_real is None or df_real.empty:
            raise ValueError(
                "Prophet arrival modeling requires historical case timestamps from an aligned log, "
                "but no valid event log was found."
            )

        # 1. Extract historical case start times and aggregate into hourly counts
        first_times = df_real.groupby("case:concept:name")["time:timestamp"].min().reset_index()
        first_times.columns = ["case_id", "timestamp"]
        first_times["timestamp"] = pd.to_datetime(first_times["timestamp"])
        if hasattr(first_times["timestamp"].dt, "tz") and first_times["timestamp"].dt.tz is not None:
            first_times["timestamp"] = first_times["timestamp"].dt.tz_localize(None)

        hourly_counts = (
            first_times.set_index("timestamp")
            .resample("1h")
            .size()
            .reset_index()
        )
        hourly_counts.columns = ["ds", "y"]

        n_cases = self.num_cases
        dur_days = float(self.duration_days) if (self.duration_days and float(self.duration_days) > 0) else None

        if n_cases is not None:
            effective_cases = n_cases
            desc_str = f"{n_cases:,} cases"
        elif dur_days is not None:
            effective_cases = None
            desc_str = f"{dur_days} days horizon"
        else:
            effective_cases = len(first_times)
            desc_str = f"{effective_cases:,} cases (matching historical log)"

        print(f"Generating synthetic case arrivals ({desc_str}) [Prophet Time-Series Arrival Mode]...")

        # 2. Configure & Fit / Load Cached Prophet Model
        prophet_cfg = self.arr_cfg.get("prophet", {})
        growth = prophet_cfg.get("growth", "logistic")
        cap_multiplier = float(prophet_cfg.get("cap_multiplier", 1.25))
        sampling_method = prophet_cfg.get("sampling_method", "triangular")
        cp_prior = float(prophet_cfg.get("changepoint_prior_scale", 0.05))
        sp_prior = float(prophet_cfg.get("seasonality_prior_scale", 10.0))
        daily_s = prophet_cfg.get("daily_seasonality", True)
        weekly_s = prophet_cfg.get("weekly_seasonality", True)
        intra_dist = prophet_cfg.get("intra_hour_distribution", "uniform")
        calendar_bounded = self.arr_cfg.get("calendar_bounded", True)
        cache_model = prophet_cfg.get("cache_model", True)
        model_cache_dir = prophet_cfg.get("model_cache_dir", "models/prophet")
        ms_prefixes = tuple(self.cfg.get("preprocessing", {}).get("milestone_prefixes", []))

        # Resolve cache paths
        cache_dir = os.path.join(PROJECT_ROOT, model_cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        proj_name = self.cfg.get("project", {}).get("name", "BPI_2012")
        model_cache_path = os.path.join(cache_dir, f"prophet_arrivals_{proj_name}.json")
        meta_cache_path = os.path.join(cache_dir, f"prophet_arrivals_{proj_name}_meta.json")

        aligned_log_path = self.cfg["paths"]["aligned_log"]
        if not os.path.isabs(aligned_log_path):
            aligned_log_path = os.path.join(PROJECT_ROOT, aligned_log_path)
        log_mtime = os.path.getmtime(aligned_log_path) if os.path.exists(aligned_log_path) else 0.0
        log_size = os.path.getsize(aligned_log_path) if os.path.exists(aligned_log_path) else 0

        hyperparams = {
            "growth": growth,
            "cap_multiplier": cap_multiplier,
            "changepoint_prior_scale": cp_prior,
            "seasonality_prior_scale": sp_prior,
            "daily_seasonality": daily_s,
            "weekly_seasonality": weekly_s,
        }

        model = None
        cap_val = None
        cache_valid = False

        if cache_model and not self.force_retrain_prophet and os.path.exists(model_cache_path) and os.path.exists(meta_cache_path):
            try:
                with open(meta_cache_path, "r") as mf:
                    meta = json.load(mf)
                if (meta.get("log_mtime") == log_mtime and
                    meta.get("log_size") == log_size and
                    meta.get("hyperparams") == hyperparams):
                    with open(model_cache_path, "r") as pf:
                        model = model_from_json(json.load(pf))
                    cap_val = meta.get("cap_val")
                    cache_valid = True
                    print(f"  [✓] Loaded cached Prophet model from: {model_cache_path} (sub-second startup)")
            except Exception as e:
                print(f"  [!] Note: Cached Prophet model read failed ({e}), retraining...")
                cache_valid = False

        if not cache_valid:
            if self.force_retrain_prophet:
                print(f"  [!] Force retraining requested: fitting Prophet model on {len(hourly_counts):,} historical hours...")
            else:
                print(f"  Fitting Prophet model on {len(hourly_counts):,} historical hours (Growth={growth.upper()})...")

            if growth == "logistic":
                max_observed = float(hourly_counts["y"].max()) if not hourly_counts.empty else 1.0
                cap_val = max(1.0, max_observed * cap_multiplier)
                hourly_counts["cap"] = cap_val
                hourly_counts["floor"] = 0.0
                model = Prophet(
                    growth="logistic",
                    changepoint_prior_scale=cp_prior,
                    seasonality_prior_scale=sp_prior,
                    daily_seasonality=daily_s,
                    weekly_seasonality=weekly_s,
                )
            else:
                cap_val = None
                model = Prophet(
                    growth="linear",
                    changepoint_prior_scale=cp_prior,
                    seasonality_prior_scale=sp_prior,
                    daily_seasonality=daily_s,
                    weekly_seasonality=weekly_s,
                )

            model.fit(hourly_counts)

            if cache_model:
                try:
                    with open(model_cache_path, "w") as pf:
                        json.dump(model_to_json(model), pf)
                    meta = {
                        "log_mtime": log_mtime,
                        "log_size": log_size,
                        "hyperparams": hyperparams,
                        "cap_val": cap_val,
                        "trained_at": datetime.now().isoformat()
                    }
                    with open(meta_cache_path, "w") as mf:
                        json.dump(meta, mf, indent=2)
                    print(f"  [✓] Cached fitted Prophet model to: {model_cache_path}")
                except Exception as e:
                    print(f"  [!] Warning: Failed to save Prophet cache: {e}")

        # 3. Forecast future periods
        start_str = self.arr_cfg.get("start_timestamp", "2012-01-01 08:00:00")
        start_time = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")

        # Determine forecasting horizon
        if dur_days is not None:
            hours_to_forecast = int(np.ceil(dur_days * 24))
        else:
            target_n = effective_cases if effective_cases is not None else 1000
            active_hours = hourly_counts[hourly_counts["y"] > 0]
            mean_per_hour = max(0.5, float(active_hours["y"].mean()) if not active_hours.empty else 2.0)
            hours_to_forecast = max(24, int((target_n / mean_per_hour) * 2.5))

        cur_start = start_time
        arrivals = []
        case_idx = 1
        cur_time = start_time
        if self.calendar is not None:
            self.calendar.set_ref_dt(cur_time)

        while True:
            future_dates = pd.date_range(start=cur_start, periods=hours_to_forecast, freq="1h")
            df_future = pd.DataFrame({"ds": future_dates})
            if growth == "logistic":
                df_future["cap"] = cap_val if cap_val is not None else float(hourly_counts["y"].max() * cap_multiplier)
                df_future["floor"] = 0.0

            forecast = model.predict(df_future)

            for _, row in forecast.iterrows():
                if effective_cases is not None and len(arrivals) >= effective_cases:
                    break
                if dur_days is not None and (cur_time - start_time).total_seconds() >= dur_days * 86400.0:
                    break

                hour_start = row["ds"].to_pydatetime()
                yhat = float(row.get("yhat", 0.0))

                if sampling_method == "triangular":
                    low = max(0.0, float(row.get("yhat_lower", 0.0)))
                    mode = max(0.0, yhat)
                    high = max(mode, float(row.get("yhat_upper", mode)))
                    if high > low:
                        raw_val = float(np.random.triangular(low, mode, high))
                    else:
                        raw_val = mode
                    hourly_expected = int(np.ceil(max(0.0, raw_val)))
                elif sampling_method == "poisson":
                    hourly_expected = int(np.random.poisson(lam=max(0.0, yhat)))
                else:  # "round" (deterministic baseline)
                    hourly_expected = max(0, int(round(yhat)))

                if hourly_expected == 0:
                    continue

                for _ in range(hourly_expected):
                    if effective_cases is not None and len(arrivals) >= effective_cases:
                        break

                    if intra_dist == "normal":
                        offset_sec = float(np.clip(np.random.normal(loc=1800.0, scale=600.0), 0.0, 3599.0))
                    else:
                        offset_sec = float(np.random.uniform(0.0, 3599.0))

                    t_arr = hour_start + timedelta(seconds=offset_sec)

                    if calendar_bounded and self.calendar is not None:
                        gap = self.calendar.get_gap_to_next_work_window(t_arr)
                        if gap > 0:
                            t_arr += timedelta(seconds=gap)

                    cur_time = max(cur_time, t_arr)
                    if dur_days is not None and (cur_time - start_time).total_seconds() >= dur_days * 86400.0:
                        break

                    cid = f"prophet_case_{case_idx:05d}"
                    if case_payload_records:
                        sampled_payload = dict(random.choice(case_payload_records))
                        amt = float(sampled_payload.get("RequestedAmount", 0.0))
                    else:
                        amt = float(random.choice(empirical_amounts)) if empirical_amounts else 0.0
                        sampled_payload = {"RequestedAmount": amt} if amt_col else {}

                    init_dist = self.cfg.get("preprocessing", {}).get("initial_milestones_distribution", [])
                    if init_dist:
                        cand_sets = [set(entry["milestones"]) for entry in init_dist]
                        probs = [float(entry["probability"]) for entry in init_dist]
                        init_ms = random.choices(cand_sets, weights=probs, k=1)[0]
                    else:
                        cfg_init_ms = set(self.cfg.get("preprocessing", {}).get("initial_milestones", []))
                        init_ms = cfg_init_ms if cfg_init_ms else ({"A_SUBMITTED", "A_PARTLYSUBMITTED"} if any(m.startswith("A_") for m in ms_prefixes) else set())

                    arrivals.append({
                        "case_id": cid,
                        "arrival_time": t_arr,
                        "requested_amount": amt,
                        "case_attrs": sampled_payload,
                        "init_ms": init_ms,
                        "ms_states": []
                    })
                    case_idx += 1

            if effective_cases is not None:
                if len(arrivals) >= effective_cases:
                    break
                # Extend window forward if more cases still needed
                cur_start = future_dates[-1] + timedelta(hours=1)
            elif dur_days is not None:
                if (cur_time - start_time).total_seconds() >= dur_days * 86400.0:
                    break
                cur_start = future_dates[-1] + timedelta(hours=1)
            else:
                break

        arrivals.sort(key=lambda x: x["arrival_time"])
        print(f"  Successfully scheduled {len(arrivals):,} cases via Prophet.")
        return arrivals
