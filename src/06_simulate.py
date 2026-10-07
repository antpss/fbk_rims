#!/usr/bin/env python3
"""
src/06_simulate.py

Modernized Config-Driven Discrete-Event Simulator (SimPy) for RIMS+.
Replaces legacy RIMS simulation (predict_simulator.py / token_LSTM.py) with:
  1. Multi-Skilled Shared Worker Pool (59 distinct employees, zero over-allocation).
  2. Dynamic Champion Model Inference (XGBoost & PyTorch TCN from dispatch_config.json).
  3. Agnostic Hybrid Residual Waiting Time Strategy: max(0, W_ML - W_queue).
  4. Advanced Calendar & Business Hours Shift Engine (automatic night & weekend pause).
  5. State-Space Petri Net Token Simulation with XGBoost XOR Decision Routing.
  6. Automatic Simulation vs. Real Log Benchmark Evaluation (Wasserstein, Cycle Time MAE).
"""

import os
import sys
import json
import time
import random
import argparse
import importlib
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import wasserstein_distance
import joblib
import torch
import xgboost as xgb
import simpy
import pm4py
from pm4py.objects.petri_net import semantics

from config_loader import load_config, PROJECT_ROOT

# Dynamic import of 04_train for DurationTCN
train_module = importlib.import_module("04_train")
DurationTCN = train_module.DurationTCN
DurationLSTM = train_module.DurationLSTM
parse_prefix = train_module.parse_prefix


# =====================================================================
# 1. Calendar & Business Hours Manager (Upgraded with Per-Role Calendars)
# =====================================================================
class CalendarManager:
    """
    Controls operational work shifts, night closures, and weekend stops.
    Supports a global business shift as well as per-role / per-activity calendar overrides.
    Ensures human resources only perform work during configured office hours.
    """
    def __init__(self, enabled=True, work_days=(0, 1, 2, 3, 4), hour_start=8, hour_end=17, ref_dt=None, per_role=None):
        self.enabled = enabled
        self.work_days = set(work_days)
        self.hour_start = hour_start
        self.hour_end = hour_end
        self.ref_dt = ref_dt or datetime(2011, 10, 1, 0, 0, 0)
        self.per_role_cfgs = per_role or {}
        self.role_calendars = {}

        # Instantiate specialized per-role / per-activity calendars
        for role_name, r_cfg in self.per_role_cfgs.items():
            self.role_calendars[role_name] = CalendarManager(
                enabled=r_cfg.get("enabled", self.enabled),
                work_days=r_cfg.get("work_days", list(self.work_days)),
                hour_start=r_cfg.get("hour_start", self.hour_start),
                hour_end=r_cfg.get("hour_end", self.hour_end),
                ref_dt=self.ref_dt
            )

    def set_ref_dt(self, ref_dt):
        """Synchronizes reference start timestamp across global and per-role calendars."""
        self.ref_dt = ref_dt
        for rc in self.role_calendars.values():
            rc.set_ref_dt(ref_dt)

    def get_calendar(self, role_or_act=None):
        """Returns specialized calendar if configured for role/activity, otherwise global calendar."""
        if role_or_act and role_or_act in self.role_calendars:
            return self.role_calendars[role_or_act]
        return self

    def to_datetime(self, sim_seconds):
        return self.ref_dt + timedelta(seconds=sim_seconds)

    def is_working_hour(self, dt):
        if not self.enabled:
            return True
        if dt.weekday() not in self.work_days:
            return False
        return self.hour_start <= dt.hour < self.hour_end

    def get_gap_to_next_work_window(self, dt):
        """Calculates seconds until the next working shift begins."""
        if not self.enabled or self.is_working_hour(dt):
            return 0.0

        curr = dt
        while True:
            if curr.weekday() not in self.work_days:
                curr = curr.replace(hour=self.hour_start, minute=0, second=0, microsecond=0) + timedelta(days=1)
                if curr.weekday() in self.work_days:
                    return (curr - dt).total_seconds()
            elif curr.hour < self.hour_start:
                next_start = curr.replace(hour=self.hour_start, minute=0, second=0, microsecond=0)
                return (next_start - dt).total_seconds()
            elif curr.hour >= self.hour_end:
                next_day = curr.replace(hour=self.hour_start, minute=0, second=0, microsecond=0) + timedelta(days=1)
                curr = next_day
                if curr.weekday() in self.work_days:
                    return (curr - dt).total_seconds()
            else:
                return max(0.0, (curr - dt).total_seconds())

    def calculate_calendar_duration(self, sim_seconds, duration_seconds, role_or_act=None):
        """
        Advances working duration across nights and weekends.
        Uses specialized per-role calendar if defined, otherwise falls back to global calendar.
        Returns total elapsed wall-clock seconds in the simulation.
        """
        if role_or_act and role_or_act in self.role_calendars:
            return self.role_calendars[role_or_act].calculate_calendar_duration(sim_seconds, duration_seconds)

        if not self.enabled or duration_seconds <= 0:
            return max(0.0, duration_seconds)

        start_dt = self.to_datetime(sim_seconds)
        initial_pause = self.get_gap_to_next_work_window(start_dt)
        curr_dt = start_dt + timedelta(seconds=initial_pause)

        work_left = duration_seconds
        while work_left > 0:
            day_end = curr_dt.replace(hour=self.hour_end, minute=0, second=0, microsecond=0)
            window_seconds = max(0.0, (day_end - curr_dt).total_seconds())

            if work_left <= window_seconds:
                curr_dt += timedelta(seconds=work_left)
                work_left = 0
            else:
                work_left -= window_seconds
                curr_dt = day_end
                gap = self.get_gap_to_next_work_window(curr_dt)
                curr_dt += timedelta(seconds=gap)

        total_elapsed = max(duration_seconds, (curr_dt - start_dt).total_seconds())
        return total_elapsed



# =====================================================================
# 2. Multi-Skilled Shared Worker Manager (59 Distinct Workers)
# =====================================================================
class WorkerManager:
    """
    Manages the 59 distinct enterprise human resources.
    Eliminates the 266-worker phantom staff bug by modeling shared multi-skilling:
    A worker can only perform one task at a time across all activities.
    """
    def __init__(self, env, role_mapping_path, latency_cfg=None):
        self.env = env
        self.latency_cfg = latency_cfg or {}
        with open(role_mapping_path, "r") as f:
            data = json.load(f)

        self.all_workers = data.get("__all_resources__", [])
        if not self.all_workers:
            self.all_workers = sorted(list(set.union(*[set(data[k]) for k in data if not k.startswith("__")])))

        self.activity_pools = {k: data[k] for k in data if not k.startswith("__")}
        self.worker_busy = {w: False for w in self.all_workers}
        self.worker_activity = {w: None for w in self.all_workers}
        self.pending_requests = []

    def get_role_occupancy(self, activities_list):
        """Returns concurrent activity utilization vector matching Role_*_OC."""
        oc = []
        for act in activities_list:
            eligible = self.activity_pools.get(act, self.all_workers)
            busy_on_act = sum(1 for w in eligible if self.worker_busy[w] and self.worker_activity[w] == act)
            oc.append(busy_on_act / max(1, len(eligible)))
        return oc

    def request_worker(self, activity):
        """Requests an available worker from the activity's eligible skill pool."""
        eligible = self.activity_pools.get(activity, self.all_workers)
        free_workers = [w for w in eligible if not self.worker_busy[w]]

        if free_workers:
            chosen = random.choice(free_workers)
            self.worker_busy[chosen] = True
            self.worker_activity[chosen] = activity
            ev = self.env.event()
            ev.succeed(value=chosen)
            return ev
        else:
            ev = self.env.event()
            self.pending_requests.append({
                "activity": activity,
                "eligible": set(eligible),
                "event": ev,
                "req_time": self.env.now
            })
            return ev

    def release_worker(self, worker_id):
        """Releases worker; if human latency modeling is active, worker enters refractory buffer."""
        self.worker_activity[worker_id] = None
        if self.latency_cfg and self.latency_cfg.get("enabled", True):
            cooldown = self._sample_cooldown(worker_id)
            if cooldown > 0:
                self.env.process(self._cooldown_and_dispatch(worker_id, cooldown))
                return
        self._complete_release(worker_id)

    def _sample_cooldown(self, worker_id):
        profiles = self.latency_cfg.get("per_worker_profiles", {})
        prof = profiles.get(worker_id) or self.latency_cfg
        dist = prof.get("best_distribution") or prof.get("global_distribution", "lognormal")
        params = prof.get("distribution_params") or prof.get("global_params", {})

        try:
            if dist == "lognormal":
                val = float(st.lognorm.rvs(params["shape"], loc=params.get("loc", 0.0), scale=params["scale"]))
            elif dist == "gamma":
                val = float(st.gamma.rvs(params["shape"], loc=params.get("loc", 0.0), scale=params["scale"]))
            else:
                mean_s = float(prof.get("mean_seconds", prof.get("global_mean_seconds", 300.0)))
                val = float(random.expovariate(1.0 / max(1.0, mean_s)))
            return min(max(5.0, val), 1800.0)  # Clamp between 5s and 30 mins
        except Exception:
            return float(random.expovariate(1.0 / 300.0))

    def _cooldown_and_dispatch(self, worker_id, cooldown_seconds):
        yield self.env.timeout(cooldown_seconds)
        self._complete_release(worker_id)

    def _complete_release(self, worker_id):
        self.worker_busy[worker_id] = False
        for i, req in enumerate(self.pending_requests):
            if worker_id in req["eligible"] and not req["event"].triggered:
                self.pending_requests.pop(i)
                self.worker_busy[worker_id] = True
                self.worker_activity[worker_id] = req["activity"]
                req["event"].succeed(value=worker_id)
                return


# =====================================================================
# 3. Dynamic Champion Model Predictor (XGBoost + PyTorch TCN)
# =====================================================================
class ChampionPredictor:
    """
    Unified Runtime Inference Engine.
    Loads and serves duration, waiting time, and XOR routing champion models.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.models_dir = cfg["paths"]["models_dir"]
        self.target_acts = cfg.get("target_activities", [])
        self.max_seq_len = cfg.get("features", {}).get("prefix_window_size", 10)

        # 1. Load Duration Champions
        dur_dir = os.path.join(self.models_dir, "duration", "hybrid_champion")
        with open(os.path.join(dur_dir, "dispatch_config.json")) as f:
            self.dur_dispatch = json.load(f)["dispatch"]
        self.dur_models = self._load_model_suite(dur_dir, self.dur_dispatch)

        # 2. Load Waiting Time Champions
        wait_dir = os.path.join(self.models_dir, "waiting_time", "hybrid_champion")
        with open(os.path.join(wait_dir, "dispatch_config.json")) as f:
            self.wait_dispatch = json.load(f)["dispatch"]
        self.wait_models = self._load_model_suite(wait_dir, self.wait_dispatch)

        # 3. Load XOR Routing Decision Classifiers
        routing_dir = os.path.join(PROJECT_ROOT, cfg.get("tasks", {}).get("routing", {}).get("output_dir", "models/routing"))
        routing_manifest_path = os.path.join(routing_dir, "routing_decisions.json")
        with open(routing_manifest_path) as f:
            self.routing_manifest = json.load(f)["decision_points"]
        self.routing_models = {}
        for p_id, p_info in self.routing_manifest.items():
            if p_info.get("strategy") == "machine_learning" and p_info.get("model_file"):
                m_path = os.path.join(routing_dir, p_info["model_file"])
                self.routing_models[p_id] = joblib.load(m_path)

        # Cache feature columns from training dataset
        dur_ds_path = os.path.join(cfg["paths"]["output_base_dir"], cfg["tasks"]["duration"]["output_file"])
        sample_df = pd.read_csv(dur_ds_path, nrows=5)
        self.feature_cols = [c for c in sample_df.columns if c not in [
            "Case_ID", "case_id", "Prefix", "Duration_Seconds", "Wait_Time_Seconds",
            "Target_Duration", "Resource_ID", "Sample_Weight", "Target_Activity"
        ]]

    def _load_model_suite(self, base_dir, dispatch_dict):
        suite = {}
        v_path = os.path.join(base_dir, "vocab.json")
        vocab = {}
        if os.path.exists(v_path):
            with open(v_path) as f:
                vocab = json.load(f)

        scalers = {}
        for fn in os.listdir(base_dir):
            if fn.startswith("scaler_") and fn.endswith(".pkl"):
                scalers[fn] = joblib.load(os.path.join(base_dir, fn))

        suite["vocab"] = vocab
        suite["scalers"] = scalers
        suite["models"] = {}

        for act, entry in dispatch_dict.items():
            m_type = entry.get("model_type", "xgboost")
            m_file = entry.get("file")
            if not m_file:
                continue
            m_path = os.path.join(base_dir, m_file)
            if not os.path.exists(m_path):
                # Fallback to candidate source dir if not co-located
                s_dir = entry.get("source_dir", "")
                parent_models_dir = os.path.dirname(base_dir)
                alt_path = os.path.join(parent_models_dir, s_dir, m_file)
                if os.path.exists(alt_path):
                    m_path = alt_path

            if m_type == "xgboost":
                booster = xgb.XGBRegressor()
                booster.load_model(m_path)
                suite["models"][act] = {"type": "xgboost", "model": booster}
            elif m_type == "tcn":
                sd = torch.load(m_path, map_location="cpu")
                vocab_size = sd["embedding.weight"].shape[0]
                embed_dim = sd["embedding.weight"].shape[1]
                fc_in = sd["fc.0.weight"].shape[1]
                hidden = sd["fc.0.weight"].shape[0]
                num_acts = sd["act_embedding.weight"].shape[0] if "act_embedding.weight" in sd else 1
                has_act_emb = ("act_embedding.weight" in sd) and (num_acts > 1)
                num_feats = fc_in - hidden - (embed_dim if has_act_emb else 0)

                tcn = DurationTCN(vocab_size=vocab_size, embed_dim=embed_dim, num_acts=num_acts,
                                  num_features=num_feats, hidden_dim=hidden, num_layers=3)
                tcn.load_state_dict(sd)
                tcn.eval()
                suite["models"][act] = {"type": "tcn", "model": tcn}
            elif m_type == "lstm":
                sd = torch.load(m_path, map_location="cpu")
                vocab_size = sd["embedding.weight"].shape[0]
                embed_dim = sd["embedding.weight"].shape[1]
                fc_in = sd["fc.0.weight"].shape[1]
                hidden = sd["fc.0.weight"].shape[0]
                num_acts = sd["act_embedding.weight"].shape[0] if "act_embedding.weight" in sd else 1
                has_act_emb = ("act_embedding.weight" in sd) and (num_acts > 1)
                num_feats = fc_in - hidden - (embed_dim if has_act_emb else 0)

                lstm = DurationLSTM(vocab_size=vocab_size, embed_dim=embed_dim, num_acts=num_acts,
                                    num_features=num_feats, hidden_dim=hidden, num_layers=2)
                lstm.load_state_dict(sd)
                lstm.eval()
                suite["models"][act] = {"type": "lstm", "model": lstm}
        return suite

    def _build_feature_dict(self, target_act, prefix, wip, ac_wip, dt, requested_amount,
                            milestones_seen, offer_count, has_offer, role_oc_array):
        feats = {
            "Prev_Proc_Time": 0.0,
            "WIP": float(wip),
            "AC_WIP": float(ac_wip.get(target_act, 0)),
            "Daytime": (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400.0,
            "Offer_Count": float(offer_count),
            "Has_Offer": float(has_offer),
            "RequestedAmount": float(requested_amount)
        }
        for d in range(7):
            feats[f"Weekday_{d}"] = 1.0 if dt.weekday() == d else 0.0
        for i, val in enumerate(role_oc_array):
            feats[f"Role_{i}_OC"] = float(val)

        for col in self.feature_cols:
            if col.startswith("Milestone_"):
                m_name = col.replace("Milestone_", "")
                feats[col] = 1.0 if m_name in milestones_seen else 0.0
            elif col not in feats:
                feats[col] = 0.0
        return feats


    def predict_duration(self, activity, prefix, wip, ac_wip, dt, requested_amount,
                         milestones_seen, offer_count, has_offer, role_oc_array):
        return self._predict_time(self.dur_models, self.dur_dispatch, activity, prefix,
                                  wip, ac_wip, dt, requested_amount, milestones_seen,
                                  offer_count, has_offer, role_oc_array, default_fallback=600.0)

    def predict_waiting(self, activity, prefix, wip, ac_wip, dt, requested_amount,
                        milestones_seen, offer_count, has_offer, role_oc_array):
        return self._predict_time(self.wait_models, self.wait_dispatch, activity, prefix,
                                  wip, ac_wip, dt, requested_amount, milestones_seen,
                                  offer_count, has_offer, role_oc_array, default_fallback=3600.0)

    def _predict_time(self, suite, dispatch_table, activity, prefix, wip, ac_wip, dt,
                      requested_amount, milestones_seen, offer_count, has_offer,
                      role_oc_array, default_fallback):
        entry = suite["models"].get(activity) or suite["models"].get("__default__")
        if not entry:
            return default_fallback

        m_type = entry["type"]
        model = entry["model"]
        f_dict = self._build_feature_dict(activity, prefix, wip, ac_wip, dt, requested_amount,
                                          milestones_seen, offer_count, has_offer, role_oc_array)

        if m_type == "xgboost":
            p_features = prefix[-self.max_seq_len:] if len(prefix) > self.max_seq_len else prefix + ["NONE"] * (self.max_seq_len - len(prefix))
            row_dict = {}
            for i, p_act in enumerate(p_features):
                row_dict[f"Act_Minus_{self.max_seq_len - i}"] = p_act
            row_dict["Target_Activity"] = activity
            for c in self.feature_cols:
                row_dict[c] = f_dict[c]

            row_df = pd.DataFrame([row_dict])
            for col in row_df.columns:
                if col.startswith("Act_Minus_") or col == "Target_Activity":
                    row_df[col] = row_df[col].astype("category")

            pred_log1p = model.predict(row_df)[0]
            val = float(np.expm1(np.clip(pred_log1p, 0.0, 25.0)))
            return max(1.0, val)

        elif m_type in ["tcn", "lstm"]:
            vocab = suite["vocab"]
            seq = [vocab.get(a, 1) for a in prefix[-self.max_seq_len:]]
            if len(seq) < self.max_seq_len:
                seq = [0] * (self.max_seq_len - len(seq)) + seq
            seq_t = torch.tensor([seq], dtype=torch.long)

            act_idx = self.target_acts.index(activity) if activity in self.target_acts else 0
            act_t = torch.tensor([act_idx], dtype=torch.long)

            scaler = suite["scalers"].get("scaler_local.pkl") or suite["scalers"].get("scaler_global.pkl")
            cols = list(scaler.feature_names_in_) if (scaler and hasattr(scaler, "feature_names_in_")) else self.feature_cols
            feat_df = pd.DataFrame([{c: f_dict.get(c, 0.0) for c in cols}])
            if scaler:
                feat_vals = scaler.transform(feat_df)
            else:
                feat_vals = feat_df.values.astype(np.float32)
            feat_t = torch.tensor(feat_vals, dtype=torch.float32)

            with torch.no_grad():
                out = model(seq_t, act_t, feat_t).item()
            val = float(np.expm1(np.clip(out, 0.0, 25.0)))
            return max(1.0, val)

        return default_fallback

    def predict_routing(self, place_id, prefix, requested_amount, milestones_seen, offer_count, has_offer, curr_dt=None):
        """Predicts outgoing transition branch at an XOR split place."""
        p_info = self.routing_manifest.get(place_id)
        if not p_info:
            return None

        # 1. ML Classifier Prediction
        if p_info.get("strategy") == "machine_learning" and place_id in self.routing_models:
            clf = self.routing_models[place_id]
            feat_cols = p_info["features"]
            recent_p = prefix[-self.max_seq_len:] if len(prefix) >= self.max_seq_len else ["<START>"] * (self.max_seq_len - len(prefix)) + prefix

            row = {}
            for k, p_act in enumerate(recent_p):
                row[f"prefix_{k+1}"] = p_act
            row["RequestedAmount"] = float(requested_amount)
            row["amount"] = float(requested_amount)
            row["Offer_Count"] = float(offer_count)
            row["Has_Offer"] = float(has_offer)
            row["prev_activity"] = prefix[-1] if prefix else "<START>"
            row["hour"] = curr_dt.hour if curr_dt is not None else 12
            row["weekday"] = curr_dt.weekday() if curr_dt is not None else 0

            for col in feat_cols:
                if col.startswith("Milestone_"):
                    m_name = col.replace("Milestone_", "")
                    row[col] = 1.0 if m_name in milestones_seen else 0.0
                elif col.startswith("has_"):
                    act_name = col.replace("has_", "")
                    row[col] = 1.0 if (act_name in prefix or act_name in milestones_seen) else 0.0
                elif col not in row:
                    row[col] = 0.0

            row_df = pd.DataFrame([row])[feat_cols]
            for c in row_df.columns:
                if c.startswith("prefix_") or c == "prev_activity":
                    row_df[c] = row_df[c].astype("category")

            label_enc = p_info["label_encoding"]
            idx_to_label = {v: k for k, v in label_enc.items()}

            try:
                if hasattr(clf, "predict_proba"):
                    probas = clf.predict_proba(row_df)[0]
                    p_sum = probas.sum()
                    if p_sum > 0:
                        probas = probas / p_sum
                        pred_code = np.random.choice(clf.classes_, p=probas)
                    else:
                        pred_code = clf.predict(row_df)[0]
                else:
                    pred_code = clf.predict(row_df)[0]

                return idx_to_label.get(pred_code)
            except Exception:
                pass  # Fall through to empirical probability fallback if unseen sequence category occurs

        # 2. Empirical Probability Fallback
        probs = p_info.get("probabilities", {})
        if probs:
            targets = list(probs.keys())
            weights = list(probs.values())
            return random.choices(targets, weights=weights, k=1)[0]

        # 3. Default to first observed branch
        branches = list(p_info.get("classes_observed", {}).keys())
        return branches[0] if branches else "<END>"


# =====================================================================
# 4. Discrete-Event Simulation Engine (PM4Py Petri Net Marking Semantics)
# =====================================================================
class DiscreteEventSimulation:
    def __init__(self, cfg, num_cases=None, mode=None, arrival_mode=None, export_xes=None, pnml_path=None):
        self.cfg = cfg
        self.mode = mode or cfg.get("simulation", {}).get("mode", "hybrid_residual")
        self.num_cases = num_cases
        self.max_events = cfg.get("simulation", {}).get("max_events_per_case", 80)
        self.sim_log_path = os.path.join(
            PROJECT_ROOT,
            f"data/processed/simulated_log_{self.mode}.csv" if self.mode != "hybrid_residual" else
            cfg.get("simulation", {}).get("output_simulated_log", "data/processed/simulated_log.csv")
        )

        # Arrival Mode (Replay vs. Generative)
        arr_cfg = cfg.get("simulation", {}).get("arrival", {})
        self.arrival_mode = arrival_mode or arr_cfg.get("mode", "replay")
        self.arr_cfg = arr_cfg

        # XES Export Configuration
        self.export_xes = export_xes if export_xes is not None else cfg.get("simulation", {}).get("export_xes", True)
        self.output_simulated_xes = os.path.join(
            PROJECT_ROOT,
            cfg.get("simulation", {}).get("output_simulated_xes", "data/processed/simulated_log.xes")
        )

        # Load Petri Net & Semantics
        target_pnml = pnml_path or os.path.join(PROJECT_ROOT, cfg.get("paths", {}).get("petri_net", "models/petri_nets/discovered_model_split.pnml"))
        if not os.path.exists(target_pnml):
            raise FileNotFoundError(f"Petri Net PNML file not found at: {target_pnml}")
        self.net, self.im, self.fm = pm4py.read_pnml(target_pnml)
        self.pnml_path = target_pnml
        self.branch_target_map = self._build_branch_target_map()

        # Setup Calendar Manager (Global + Per-Role / Per-Activity Overrides)
        cal_cfg = cfg.get("simulation", {}).get("calendar", {})
        self.calendar = CalendarManager(
            enabled=cal_cfg.get("enabled", True),
            work_days=cal_cfg.get("work_days", [0, 1, 2, 3, 4]),
            hour_start=cal_cfg.get("hour_start", 8),
            hour_end=cal_cfg.get("hour_end", 17),
            per_role=cal_cfg.get("per_role", {})
        )

        # Load Causal Delays & Human Latency Manifest
        delay_manifest_path = os.path.join(PROJECT_ROOT, "models/delays/delay_manifest.json")
        self.delay_manifest = None
        self.external_delays = {}
        if os.path.exists(delay_manifest_path):
            try:
                with open(delay_manifest_path, "r") as f:
                    self.delay_manifest = json.load(f)
                for d in self.delay_manifest.get("external_uncoupled_delays", []):
                    self.external_delays[(d["trigger_activity"], d["resume_activity"])] = d
            except Exception as e:
                print(f"[!] Warning: Failed to load delay manifest: {e}")

        # Load Process Elements (Multi-Skilled Shared Worker Pool)
        role_mapping_path = os.path.join(PROJECT_ROOT, "models/role_mapping.json")
        self.env = simpy.Environment()
        latency_cfg = self.delay_manifest.get("human_inter_ticket_latency") if self.delay_manifest else None
        self.workers = WorkerManager(self.env, role_mapping_path, latency_cfg=latency_cfg)
        self.predictor = ChampionPredictor(cfg)

        self.simulated_records = []
        self.active_cases_count = 0
        self.ac_wip = defaultdict(int)

    def _build_branch_target_map(self):
        """
        Dynamically traces Petri net arcs and silent transitions downstream
        to map (decision_place, transition_name) -> target observable activity or '<END>'.
        """
        trans_map = {t.name: t.label for t in self.net.transitions}
        is_silent = {t.name: (t.label is None) for t in self.net.transitions}
        out_arcs = defaultdict(list)
        for a in self.net.arcs:
            out_arcs[a.source.name].append(a.target.name)

        def resolve_target(t_name, visited=None):
            if visited is None:
                visited = set()
            if t_name in visited:
                return None
            visited.add(t_name)
            if not is_silent.get(t_name, False):
                return trans_map.get(t_name, t_name)
            for p in out_arcs.get(t_name, []):
                if p == "sink":
                    return "<END>"
                for dt in out_arcs.get(p, []):
                    if not is_silent.get(dt, False):
                        return trans_map.get(dt, dt)
                    res = resolve_target(dt, visited.copy())
                    if res:
                        return res
            return trans_map.get(t_name, t_name)

        branch_map = {}
        for p in self.net.places:
            if len(p.out_arcs) >= 2:
                for arc in p.out_arcs:
                    t = arc.target
                    branch_map[(p.name, t.name)] = resolve_target(t.name)
        return branch_map

    def _sample_external_delay(self, trigger_act, resume_act):
        """Samples from empirical causal distribution for external uncoupled delay."""
        d = self.external_delays.get((trigger_act, resume_act))
        if not d:
            return 0.0
        dist = d.get("best_distribution", "lognormal")
        params = d.get("distribution_params", {})
        try:
            if dist == "gamma":
                val = float(st.gamma.rvs(params["shape"], loc=params.get("loc", 0.0), scale=params["scale"]))
            elif dist == "exponential":
                val = float(st.expon.rvs(loc=params.get("loc", 0.0), scale=params["scale"]))
            elif dist == "lognormal":
                val = float(st.lognorm.rvs(params["shape"], loc=params.get("loc", 0.0), scale=params["scale"]))
            else:
                val = float(d.get("median_hours", 0.0) * 3600.0)
            return max(0.0, val)
        except Exception:
            return max(0.0, float(d.get("median_hours", 0.0) * 3600.0))

    def _update_milestones(self, milestones_seen, act):
        """Updates milestone progress context dynamically if milestone prefixes are configured."""
        ms_prefixes = tuple(self.cfg.get("preprocessing", {}).get("milestone_prefixes", []))
        if not ms_prefixes:
            return
        prog = self.cfg.get("preprocessing", {}).get("milestone_progression", {})
        if prog and act in prog:
            item = prog[act]
            if isinstance(item, dict):
                for m, prob in item.items():
                    if random.random() < float(prob):
                        milestones_seen.add(m)
            elif isinstance(item, (list, tuple, set)):
                milestones_seen.update(item)
        elif not prog and any(m.startswith("A_") for m in ms_prefixes):
            if act == "W_Complete application":
                milestones_seen.update(["A_ACCEPTED", "O_SELECTED", "A_FINALIZED", "O_CREATED", "O_SENT"])
            elif act == "W_Call after offers":
                milestones_seen.add("O_SENT_BACK")
            elif act == "W_Validate application":
                milestones_seen.update(["A_APPROVED", "A_REGISTERED", "A_ACTIVATED", "O_ACCEPTED"])
            elif act == "W_Handle leads":
                milestones_seen.add("A_PREACCEPTED")

    def _prepare_arrivals(self):
        """
        Prepares arrival schedule:
          - Replay Mode: Replays historical timestamps & case payloads from aligned XES log.
          - Generative Mode: Generates synthetic arrivals from exponential/uniform distributions,
            bounded by working calendar and empirical case attribute distributions.
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
            n_cases = self.num_cases or 1000
            print(f"Generating {n_cases:,} purely synthetic case arrivals (Generative Arrival Mode)...")
            start_str = self.arr_cfg.get("start_timestamp", "2012-01-01 08:00:00")
            cur_time = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
            self.calendar.set_ref_dt(cur_time)

            dist_name = self.arr_cfg.get("distribution", "exponential")
            scale = float(self.arr_cfg.get("scale", 300.0))
            calendar_bounded = self.arr_cfg.get("calendar_bounded", True)
            ms_prefixes = tuple(self.cfg.get("preprocessing", {}).get("milestone_prefixes", []))

            for i in range(n_cases):
                cid = f"synth_case_{i+1:05d}"
                if dist_name == "exponential":
                    interval = float(np.random.exponential(scale))
                elif dist_name == "uniform":
                    interval = float(np.random.uniform(scale * 0.5, scale * 1.5))
                else:
                    interval = scale

                cur_time += timedelta(seconds=interval)
                if calendar_bounded:
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


    def run(self):
        print("\n" + "=" * 80)
        print("      RIMS+ MODERN DISCRETE-EVENT SIMULATOR (SIMPY)")
        print(f"      Mode:        [{self.mode.upper()}]")
        print(f"      Arrivals:    [{self.arrival_mode.upper()}] Mode")
        print(f"      Petri Net:   {os.path.basename(self.pnml_path)}")
        print(f"      Workers:     {len(self.workers.all_workers)} Distinct Multi-Skilled Human Agents")
        print(f"      Calendar:    Working Hours {self.calendar.hour_start}:00 - {self.calendar.hour_end}:00 (Mon-Fri)")
        if self.calendar.role_calendars:
            print(f"      Per-Role:    {len(self.calendar.role_calendars)} Custom Role Schedules Active")
        ext_count = len(self.external_delays)
        print(f"      Delays:      {ext_count} Causal External Arcs Active")
        hl_cfg = self.delay_manifest.get("human_inter_ticket_latency", {}) if self.delay_manifest else {}
        hl_str = f"Active ({hl_cfg.get('global_mean_seconds', 325.4)}s mean gap)" if hl_cfg.get("enabled") else "Disabled"
        print(f"      Latency:     Human Inter-Ticket Latency {hl_str}")
        print("=" * 80)

        # Prepare Case Arrivals
        arrivals, df_real = self._prepare_arrivals()
        print(f"Scheduled {len(arrivals):,} cases for simulation.")

        # Reference start datetime
        ref_dt = arrivals[0]["arrival_time"]
        self.calendar.set_ref_dt(ref_dt)

        # Launch SimPy Arrival Processes
        for arr in arrivals:
            offset_seconds = (arr["arrival_time"] - ref_dt).total_seconds()
            self.env.process(self._case_process(arr["case_id"], offset_seconds, arr["requested_amount"],
                                                arr["init_ms"], arr["ms_states"]))

        start_exec = time.time()
        print("\nExecuting simulation engine...")
        self.env.run()
        elapsed = time.time() - start_exec
        print(f"\n[✓] Simulation completed in {elapsed:.2f} seconds ({len(self.simulated_records):,} events generated).")

        # Save simulated event log (CSV)
        df_sim = pd.DataFrame(self.simulated_records)
        os.makedirs(os.path.dirname(self.sim_log_path), exist_ok=True)
        df_sim.to_csv(self.sim_log_path, index=False)
        print(f"[✓] Saved simulated event log (CSV) to: {self.sim_log_path}")

        # Save simulated event log (Standard IEEE XES)
        if self.export_xes and len(df_sim) > 0:
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

        # Run Validation & Benchmark Evaluation
        return self._evaluate(df_real, df_sim, elapsed)

    def _case_process(self, case_id, arrival_offset, requested_amount, init_ms=None, ms_states=None):
        """
        Simulates single case token lifecycle across Petri net places, transitions,
        and SimPy contention queues using PM4Py token marking semantics.
        """
        yield self.env.timeout(arrival_offset)

        self.active_cases_count += 1
        prefix = []
        ms_prefixes = tuple(self.cfg.get("preprocessing", {}).get("milestone_prefixes", []))
        cfg_init_ms = set(self.cfg.get("preprocessing", {}).get("initial_milestones", []))
        milestones_seen = set(init_ms or (cfg_init_ms if cfg_init_ms else ({"A_SUBMITTED", "A_PARTLYSUBMITTED"} if any(m.startswith("A_") for m in ms_prefixes) else set())))
        offer_count = sum(1 for m in milestones_seen if m.startswith("O_")) if ("Offer_Count" in self.predictor.feature_cols or "O_" in ms_prefixes) else 0
        has_offer = 1 if offer_count > 0 else 0

        events_done = 0
        ms_states = ms_states or []
        prev_act = None

        # Initialize Token Marking at Source
        marking = self.im.copy()

        # Step through Petri Net until final marking or safety ceiling
        while marking and marking != self.fm and events_done < self.max_events:
            enabled = list(semantics.enabled_transitions(self.net, marking))
            if not enabled:
                # Deadlock or process termination reached
                break

            # 1. Resolve Transition Choice at Decision Points
            if len(enabled) == 1:
                chosen_t = enabled[0]
            else:
                # XOR Decision Point: identify place in marking with out-degree >= 2
                decision_places = [p for p in marking if len(p.out_arcs) >= 2]
                if decision_places:
                    dp = decision_places[0].name
                    predicted_target = self.predictor.predict_routing(
                        dp, prefix, requested_amount, milestones_seen, offer_count, has_offer,
                        curr_dt=self.calendar.to_datetime(self.env.now)
                    )
                    # Match predicted target to an enabled transition
                    matched_t = None
                    for cand_t in enabled:
                        tgt = self.branch_target_map.get((dp, cand_t.name))
                        if tgt == predicted_target:
                            matched_t = cand_t
                            break
                    chosen_t = matched_t if matched_t else enabled[0]
                else:
                    chosen_t = enabled[0]

            # 2. Check if chosen transition is Silent vs. Observable Work Item
            if chosen_t.label is None:
                # Silent transition: instant state transition
                marking = semantics.execute(chosen_t, self.net, marking)
                continue

            # Observable Work Item Activity
            act = chosen_t.label
            events_done += 1

            # 3. External Uncoupled Delay (e.g. customer postal / response transit)
            if prev_act and (prev_act, act) in self.external_delays:
                ext_delay = self._sample_external_delay(prev_act, act)
                if ext_delay > 0:
                    yield self.env.timeout(ext_delay)

            # 4. Physical Queue Contention (SimPy)
            t_req = self.env.now
            worker_event = self.workers.request_worker(act)
            worker_id = yield worker_event
            t_acquired = self.env.now
            w_queue = max(0.0, t_acquired - t_req)

            # 5. Predicted Waiting Time & Agnostic Hybrid Residual
            role_oc = self.workers.get_role_occupancy(self.cfg.get("target_activities", []))
            curr_dt = self.calendar.to_datetime(self.env.now)
            if self.mode != "pure_physics":
                w_ml = self.predictor.predict_waiting(act, prefix, self.active_cases_count,
                                                      self.ac_wip, curr_dt, requested_amount,
                                                      milestones_seen, offer_count, has_offer, role_oc)
            else:
                w_ml = 0.0

            if self.mode == "hybrid_residual":
                w_residual = max(0.0, w_ml - w_queue)
                if w_residual > 0:
                    yield self.env.timeout(w_residual)
                total_wait = w_queue + w_residual
            elif self.mode == "pure_ml":
                yield self.env.timeout(w_ml)
                total_wait = w_ml
            elif self.mode == "pure_physics":
                total_wait = w_queue

            # 6. Execution Duration with Calendar Working Hours (Per-Role or Global)
            t_start = self.env.now
            self.ac_wip[act] += 1
            dur_pred = self.predictor.predict_duration(act, prefix, self.active_cases_count,
                                                       self.ac_wip, curr_dt, requested_amount,
                                                       milestones_seen, offer_count, has_offer, role_oc)

            actual_dur = self.calendar.calculate_calendar_duration(self.env.now, dur_pred, role_or_act=act)
            yield self.env.timeout(actual_dur)
            t_comp = self.env.now

            # 7. Release Worker
            self.ac_wip[act] = max(0, self.ac_wip[act] - 1)
            self.workers.release_worker(worker_id)

            # 8. Fire Transition in Petri Net Marking
            marking = semantics.execute(chosen_t, self.net, marking)

            # 9. Log Event
            start_ts = self.calendar.to_datetime(t_start).strftime("%Y-%m-%d %H:%M:%S")
            comp_ts = self.calendar.to_datetime(t_comp).strftime("%Y-%m-%d %H:%M:%S")
            self.simulated_records.append({
                "case_id": case_id,
                "activity": act,
                "start_timestamp": start_ts,
                "complete_timestamp": comp_ts,
                "resource_id": worker_id,
                "duration_seconds": round(actual_dur, 2),
                "wait_time_seconds": round(total_wait, 2),
                "queue_wait_seconds": round(w_queue, 2),
                "wip": self.active_cases_count
            })

            prefix.append(act)
            prev_act = act

            # Advance milestone state if available from case history, else apply progression
            if events_done < len(ms_states):
                milestones_seen = set(ms_states[events_done])
            else:
                self._update_milestones(milestones_seen, act)

            offer_count = sum(1 for m in milestones_seen if m.startswith("O_")) if ("Offer_Count" in self.predictor.feature_cols or "O_" in ms_prefixes) else 0
            has_offer = 1 if offer_count > 0 else 0


        self.active_cases_count = max(0, self.active_cases_count - 1)

    def _evaluate(self, df_real, df_sim, elapsed=0.0):
        """Compares simulated event log against real historical log (or summarizes simulated stats)."""
        print("\n" + "=" * 80)
        print("          RIMS+ SIMULATION VS. REAL LOG BENCHMARK")
        print("=" * 80)

        # Simulated cycle times
        df_sim["start_dt"] = pd.to_datetime(df_sim["start_timestamp"])
        df_sim["comp_dt"] = pd.to_datetime(df_sim["complete_timestamp"])
        sim_cases = df_sim.groupby("case_id").agg(
            start=("start_dt", "min"),
            end=("comp_dt", "max")
        )
        sim_cycles = (sim_cases["end"] - sim_cases["start"]).dt.total_seconds().values
        sim_mean = float(np.mean(sim_cycles)) if len(sim_cycles) > 0 else 0.0
        sim_median = float(np.median(sim_cycles)) if len(sim_cycles) > 0 else 0.0

        def format_dur(sec):
            if sec >= 86400:
                return f"{sec/86400:.1f} days"
            elif sec >= 3600:
                return f"{sec/3600:.1f} hours"
            elif sec >= 60:
                return f"{sec/60:.1f} mins"
            return f"{sec:.1f}s"

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

            print(f"Metric                      | Real Historical | Simulated RIMS+ | Difference")
            print("-" * 75)
            print(f"Cases Evaluated             | {len(real_cases):>15,} | {len(sim_cases):>15,} | ---")
            print(f"Total Work Events           | {len(real_w):>15,} | {len(df_sim):>15,} | ---")
            print(f"Mean Case Cycle Time        | {format_dur(real_mean):>15} | {format_dur(sim_mean):>15} | {format_dur(mae_cycle)}")
            print(f"Median Case Cycle Time      | {format_dur(real_median):>15} | {format_dur(sim_median):>15} | ---")
            print(f"Cycle Time Wasserstein Dist |             --- |             --- | {format_dur(wd)}")
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
        for act in self.cfg.get("target_activities", []):
            cnt = int(sim_act_stats.loc[act, "count"]) if act in sim_act_stats.index else 0
            s_dur = sim_act_stats.loc[act, "sim_dur"] if act in sim_act_stats.index else 0.0
            s_wait = sim_act_stats.loc[act, "sim_wait"] if act in sim_act_stats.index else 0.0
            s_queue = sim_act_stats.loc[act, "sim_queue"] if act in sim_act_stats.index else 0.0
            print(f"{act:<32} | {cnt:>10,} | {format_dur(s_dur):>14} | {format_dur(s_wait):>14} | {format_dur(s_queue):>14}")
        print("=" * 92 + "\n")

        # Save Comprehensive Markdown Report
        rep_path = os.path.join(PROJECT_ROOT, "data/processed/simulation_report.md")
        with open(rep_path, "w") as f:
            f.write("# RIMS+ Simulation Benchmark Report\n\n")
            f.write(f"- **Simulation Mode:** `{self.mode}`\n")
            f.write(f"- **Arrival Mode:** `{self.arrival_mode}`\n")
            f.write(f"- **Simulated Cases:** {len(sim_cases):,}\n")
            f.write(f"- **Total Work Events:** {len(df_sim):,}\n")
            f.write(f"- **Workers in Multi-Skilled Pool:** {len(self.workers.all_workers)} human resources\n")
            f.write(f"- **Calendar Engine:** Working Hours {self.calendar.hour_start}:00 - {self.calendar.hour_end}:00 (Mon-Fri)\n")
            f.write(f"- **Simulated Log Path (CSV):** `{self.sim_log_path}`\n")
            if self.export_xes:
                f.write(f"- **Simulated Log Path (XES):** `{self.output_simulated_xes}`\n\n")
            else:
                f.write("\n")
            f.write("## Overall Cycle Time Benchmark\n\n")
            f.write("| Metric | Real Historical | Simulated RIMS+ | Difference |\n")
            f.write("|---|---|---|---|\n")
            f.write(f"| **Cases Evaluated** | {len(real_cases) if df_real is not None else 'N/A'} | {len(sim_cases):,} | --- |\n")
            f.write(f"| **Total Work Events** | {len(real_w) if df_real is not None else 'N/A'} | {len(df_sim):,} | --- |\n")
            f.write(f"| **Mean Case Cycle Time** | {format_dur(real_mean)} | {format_dur(sim_mean)} | {format_dur(mae_cycle)} |\n")
            f.write(f"| **Median Case Cycle Time** | {format_dur(real_median)} | {format_dur(sim_median)} | --- |\n")
            f.write(f"| **Cycle Time Wasserstein Dist** | --- | --- | {format_dur(wd)} |\n\n")
            f.write("## Activity Execution Breakdown\n\n")
            f.write("| Activity | Simulated Count | Sim Mean Duration | Sim Mean Wait | Sim Mean Queue Wait |\n")
            f.write("|---|---|---|---|---|\n")
            for act in self.cfg.get("target_activities", []):
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
            "elapsed": elapsed
        }


# =====================================================================
# Main CLI Entrypoint
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="RIMS+ Modern Discrete-Event Simulator")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--cases", type=str, default=None,
                        help="Number of cases to simulate (e.g. 1000, 2500, or 'all')")
    parser.add_argument("--all", action="store_true",
                        help="Simulate all cases available in the event log")
    parser.add_argument("--mode", type=str, default=None, choices=["hybrid_residual", "pure_physics", "pure_ml"],
                        help="Simulation waiting time mode")
    parser.add_argument("--arrival_mode", type=str, default=None, choices=["replay", "generative"],
                        help="Arrival generation mode: 'replay' from aligned log, or 'generative' distribution")
    parser.add_argument("--export_xes", action="store_true", default=None,
                        help="Export standard IEEE XES log (.xes)")
    parser.add_argument("--pnml", type=str, default=None,
                        help="Path to override Petri net PNML file")
    parser.add_argument("--compare_modes", action="store_true",
                        help="Run 3-way ablation study comparing all simulation modes")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.all or (args.cases and str(args.cases).lower() == "all"):
        num_cases = None
    elif args.cases is not None:
        num_cases = int(args.cases)
    else:
        num_cases = cfg.get("simulation", {}).get("num_cases", 1000)

    arrival_mode = args.arrival_mode or cfg.get("simulation", {}).get("arrival", {}).get("mode", "replay")
    export_xes = args.export_xes if args.export_xes is not None else cfg.get("simulation", {}).get("export_xes", True)
    pnml_path = args.pnml or os.path.join(PROJECT_ROOT, cfg.get("paths", {}).get("petri_net", "models/petri_nets/discovered_model_split.pnml"))

    def format_dur(sec):
        if sec >= 86400:
            return f"{sec/86400:.1f} days"
        elif sec >= 3600:
            return f"{sec/3600:.1f} hours"
        elif sec >= 60:
            return f"{sec/60:.1f} mins"
        return f"{sec:.1f}s"

    if args.compare_modes:
        modes = ["pure_physics", "pure_ml", "hybrid_residual"]
        results = []
        case_label = f"{num_cases:,}" if num_cases else "ALL"
        print("\n" + "=" * 105)
        print("                        RIMS+ 3-WAY SIMULATION ABLATION SUITE")
        print(f"                        Evaluating {case_label} cases across all 3 modes")
        print("=" * 105)

        for m in modes:
            print(f"\n>>> Running Mode: [{m.upper()}] ...")
            sim = DiscreteEventSimulation(cfg, num_cases=num_cases, mode=m,
                                          arrival_mode=arrival_mode, export_xes=export_xes, pnml_path=pnml_path)
            res = sim.run()
            results.append(res)

        print("\n" + "=" * 105)
        print("                           RIMS+ SIMULATION ABLATION LEADERBOARD")
        print("=" * 105)
        print(f"{'Simulation Mode':<18} | {'Mean Cycle':>14} | {'Median Cycle':>14} | {'Cycle Time MAE':>16} | {'Wasserstein':>14} | {'Run Time':>10}")
        print("-" * 105)
        real_m = results[0]["real_mean"]
        real_med = results[0]["real_median"]
        print(f"{'REAL HISTORICAL':<18} | {format_dur(real_m):>14} | {format_dur(real_med):>14} | {'---':>16} | {'---':>14} | {'---':>10}")
        print("-" * 105)

        # Sort by lowest Wasserstein distance
        sorted_res = sorted(results, key=lambda x: x["wd"])
        best_mode = sorted_res[0]["mode"]
        for r in results:
            tag = " ★ (Best)" if r["mode"] == best_mode else ""
            print(f"{r['mode']:<18} | {format_dur(r['sim_mean']):>14} | {format_dur(r['sim_median']):>14} | {format_dur(r['mae_cycle']):>16} | {format_dur(r['wd']):>14} | {r['elapsed']:>9.1f}s{tag}")
        print("=" * 105 + "\n")

        # Save Ablation Report
        abl_path = os.path.join(PROJECT_ROOT, "data/processed/ablation_report.md")
        with open(abl_path, "w") as f:
            f.write("# RIMS+ Simulation Ablation Benchmark Report\n\n")
            f.write(f"- **Evaluated Cases per Mode:** {case_label}\n")
            f.write(f"- **Real Historical Mean Cycle Time:** {format_dur(real_m)}\n\n")
            f.write("| Simulation Mode | Mean Cycle Time | Median Cycle Time | Cycle Time MAE | Wasserstein Dist | Run Time |\n")
            f.write("|---|---|---|---|---|---|\n")
            f.write(f"| **REAL HISTORICAL** | {format_dur(real_m)} | {format_dur(real_med)} | --- | --- | --- |\n")
            for r in sorted_res:
                tag = " ★ (Best)" if r["mode"] == best_mode else ""
                f.write(f"| `{r['mode']}`{tag} | {format_dur(r['sim_mean'])} | {format_dur(r['sim_median'])} | {format_dur(r['mae_cycle'])} | {format_dur(r['wd'])} | {r['elapsed']:.1f}s |\n")
        print(f"[✓] Saved comprehensive ablation report to: {abl_path}\n")

    else:
        mode = args.mode or cfg.get("simulation", {}).get("mode", "hybrid_residual")
        sim = DiscreteEventSimulation(cfg, num_cases=num_cases, mode=mode,
                                      arrival_mode=arrival_mode, export_xes=export_xes, pnml_path=pnml_path)
        sim.run()


if __name__ == "__main__":
    main()

