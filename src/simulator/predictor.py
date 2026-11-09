#!/usr/bin/env python3
"""
src/simulator/predictor.py

Unified Runtime Model Serving & Inference Wrapper.
Loads Champion dispatch configurations and serves:
  - Duration predictions (XGBoost, TCN, LSTM)
  - Waiting time predictions (XGBoost, TCN, LSTM)
  - XOR routing decision branching (White-Box Hyperopt Decision Trees, Empirical Probabilities)
"""

import os
import sys
import json
import random
import logging
import importlib
import warnings

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

logger = logging.getLogger("rims_predictor")

import numpy as np
import pandas as pd
import joblib
import torch
import xgboost as xgb

torch.set_num_threads(1)

try:
    from config_loader import PROJECT_ROOT
except ImportError:
    from src.config_loader import PROJECT_ROOT

try:
    from features import FeatureManager, FeatureContext
except ImportError:
    from src.features import FeatureManager, FeatureContext

# Dynamic import of 04_train architectures
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    train_module = importlib.import_module("04_train")
    DurationTCN = getattr(train_module, "DurationTCN", None)
    DurationLSTM = getattr(train_module, "DurationLSTM", None)
except Exception:
    DurationTCN = None
    DurationLSTM = None


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

        # Load Activity Profiles (Human Work Items vs. Automated Tasks vs. Instant Gateways)
        try:
            from activity_profiler import load_or_discover_activity_profiles
        except ImportError:
            from src.activity_profiler import load_or_discover_activity_profiles

        prof_data = load_or_discover_activity_profiles(cfg)
        self.activity_profiles = prof_data.get("activity_types", {})

        # Trace history tracked activities
        th_cfg = cfg.get("features", {}).get("trace_history", {})
        th_mode = th_cfg.get("mode", "profile_aware")
        if th_mode == "whitelist":
            self.tracked_activities = th_cfg.get("whitelist", [])
        elif th_mode == "all_activities":
            self.tracked_activities = list(self.activity_profiles.keys())
        else:  # "profile_aware"
            self.tracked_activities = [
                a for a, t in self.activity_profiles.items()
                if t in ("INSTANT_GATEWAY", "AUTOMATED_TASK")
            ]

        # Resolve exactly the deployment requested by configuration.
        strategy = cfg.get("model_strategy", {})
        mode = strategy.get("mode", "hybrid")
        deployments = {"hybrid": "hybrid_champion", "global_tournament": "global_champion",
                       "single_global": "global_" + strategy.get("single_global_engine", "xgboost")}
        if mode not in deployments:
            raise ValueError(f"Unknown model strategy: {mode}")
        for task, prefix in [("duration", "dur"), ("waiting_time", "wait")]:
            directory = os.path.join(self.models_dir, task, deployments[mode])
            with open(os.path.join(directory, "dispatch_config.json")) as f:
                manifest = json.load(f)
            if manifest.get("task") != task:
                raise ValueError(f"Wrong task in {directory}")
            if manifest.get("schema_version") != 2:
                raise ValueError(f"Legacy feature/model schema in {directory}; rebuild discovery, datasets and models")
            with open(os.path.join(directory, "training_metadata.json")) as f:
                metadata = json.load(f)
                if metadata.get("feature_config") != cfg.get("features", {}):
                    raise ValueError(f"Feature configuration differs from training in {directory}; retrain models")
                quotas = cfg.get("resources", {}).get("simulation", {}).get("enable_concurrency_quotas", True)
                if metadata.get("capacity_quotas") != quotas:
                    raise ValueError("Worker capacity policy differs from training; rebuild features and models")
                setattr(self, prefix + "_metadata", metadata)
            setattr(self, prefix + "_dispatch", manifest["dispatch"])
            setattr(self, prefix + "_models", self._load_model_suite(directory, manifest["dispatch"]))

        # 3. Load XOR Routing Decision Classifiers
        routing_dir = os.path.join(
            PROJECT_ROOT,
            cfg.get("paths", {}).get("routing_models_dir") or cfg.get("tasks", {}).get("routing", {}).get("output_dir", "models/routing")
        )
        routing_manifest_path = os.path.join(routing_dir, "routing_decisions.json")
        with open(routing_manifest_path) as f:
            self.routing_manifest = json.load(f)["decision_points"]
        self.routing_models = {}
        for p_id, p_info in self.routing_manifest.items():
            if p_info.get("strategy") == "machine_learning":
                m_file = p_info.get("model_file")
                if m_file:
                    m_path = os.path.join(routing_dir, m_file)
                    if os.path.exists(m_path):
                        try:
                            self.routing_models[p_id] = joblib.load(m_path)
                        except Exception as e_job:
                            logger.warning(
                                f"Failed to load routing model for place '{p_id}' from '{m_path}': {e_job}. "
                                "Falling back to empirical probabilities."
                            )
                    else:
                        logger.warning(
                            f"Routing model file '{m_file}' for place '{p_id}' not found at '{m_path}'."
                        )

        # Cache feature columns and empirical percentile training bounds to prevent OOD extrapolation
        self.feature_mgr = FeatureManager(cfg)
        self.feature_cols = self.dur_metadata["feature_columns"]
        self.feature_max_caps = self.dur_metadata["feature_caps"]
        self.dur_max_caps = self.dur_metadata["activity_caps"]
        self.dur_global_cap = self.dur_metadata["global_cap"]
        self.wait_max_caps = self.wait_metadata["activity_caps"]
        self.wait_global_cap = self.wait_metadata["global_cap"]

    def _load_model_suite(self, base_dir, dispatch_dict):
        if any("bundle" in e for e in dispatch_dict.values()):
            loaded = {}
            result = {"models": {}}
            for act, entry in dispatch_dict.items():
                bundle = entry["bundle"]
                if bundle not in loaded:
                    directory = os.path.join(base_dir, bundle)
                    with open(os.path.join(directory, "dispatch_config.json")) as f:
                        child = json.load(f)
                    loaded[bundle] = self._load_model_suite(directory, child["dispatch"])
                sub = loaded[bundle]
                model = sub["models"].get(act) or sub["models"].get("__default__")
                if model is None:
                    raise ValueError(f"Missing model for {act} in {bundle}")
                result["models"][act] = {**model, "suite": sub}
            return result
        suite = {}
        meta_path = os.path.join(base_dir, "training_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                suite["metadata"] = json.load(f)
        v_path = os.path.join(base_dir, "vocab.json")
        vocab = {}
        act_to_idx = {}
        if os.path.exists(v_path):
            with open(v_path) as f:
                v_data = json.load(f)
                if isinstance(v_data, dict):
                    vocab = v_data.get("vocab", v_data)
                    act_to_idx = v_data.get("act_to_idx", {})

        scalers = {}
        for fn in os.listdir(base_dir):
            if fn.startswith("scaler_") and fn.endswith(".pkl"):
                scalers[fn] = joblib.load(os.path.join(base_dir, fn))

        suite["vocab"] = vocab
        suite["act_to_idx"] = act_to_idx
        suite["scalers"] = scalers
        suite["models"] = {}

        # Reconstruct CategoricalDtypes for XGBoost runtime inference parity
        act_categories = []
        res_categories = []
        cat_file = os.path.join(base_dir, "categories.json")
        if os.path.exists(cat_file):
            try:
                with open(cat_file) as f:
                    cat_meta = json.load(f)
                    act_categories = cat_meta.get("activities", [])
                    res_categories = cat_meta.get("resources", [])
            except Exception:
                pass

        if not act_categories:
            if act_to_idx:
                act_categories = sorted(list(act_to_idx.keys()))
            elif vocab:
                act_categories = sorted([k for k in vocab.keys() if not k.startswith("<")])
            elif hasattr(self, "cfg") and self.cfg.get("target_activities"):
                act_categories = sorted(list(self.cfg.get("target_activities")))

        if act_categories:
            suite["prefix_cat_dtype"] = pd.CategoricalDtype(categories=act_categories + ["NONE"], ordered=False)
            suite["target_act_dtype"] = pd.CategoricalDtype(categories=act_categories, ordered=False)
            suite["prefix_code_map"] = {c: float(i) for i, c in enumerate(act_categories + ["NONE"])}
            suite["target_code_map"] = {c: float(i) for i, c in enumerate(act_categories)}
        else:
            suite["prefix_cat_dtype"] = None
            suite["target_act_dtype"] = None
            suite["prefix_code_map"] = {}
            suite["target_code_map"] = {}

        if res_categories:
            suite["resource_cat_dtype"] = pd.CategoricalDtype(categories=res_categories + ["UNKNOWN"], ordered=False)
            suite["resource_code_map"] = {c: float(i) for i, c in enumerate(res_categories + ["UNKNOWN"])}
        else:
            suite["resource_cat_dtype"] = None
            suite["resource_code_map"] = {}

        def _resolve_scaler(s_dir, num_feats):
            cand_scalers = []
            if s_dir:
                parent_models_dir = os.path.dirname(base_dir)
                cand_dir = os.path.join(parent_models_dir, s_dir)
                if os.path.isdir(cand_dir):
                    for fn in sorted(os.listdir(cand_dir)):
                        if fn.startswith("scaler_") and fn.endswith(".pkl"):
                            try:
                                cand_scalers.append(joblib.load(os.path.join(cand_dir, fn)))
                            except Exception:
                                pass
            for fn in sorted(scalers.keys()):
                cand_scalers.append(scalers[fn])

            for sc in cand_scalers:
                if hasattr(sc, "feature_names_in_") and len(sc.feature_names_in_) == num_feats:
                    return sc
            for sc in cand_scalers:
                if not hasattr(sc, "feature_names_in_"):
                    return sc
            return cand_scalers[0] if cand_scalers else None

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
                suite["models"][act] = {
                    "type": "xgboost",
                    "model": booster,
                    "booster": booster.get_booster()
                }
            elif m_type == "tcn":
                if DurationTCN is None:
                    raise ImportError("DurationTCN class could not be loaded from 04_train.")
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
                s_dir = entry.get("source_dir", "")
                suite["models"][act] = {
                    "type": "tcn",
                    "model": tcn,
                    "source_dir": s_dir,
                    "num_feats": num_feats,
                    "scaler": _resolve_scaler(s_dir, num_feats)
                }
            elif m_type == "lstm":
                if DurationLSTM is None:
                    raise ImportError("DurationLSTM class could not be loaded from 04_train.")
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
                s_dir = entry.get("source_dir", "")
                suite["models"][act] = {
                    "type": "lstm",
                    "model": lstm,
                    "source_dir": s_dir,
                    "num_feats": num_feats,
                    "scaler": _resolve_scaler(s_dir, num_feats)
                }
        for entry in suite["models"].values():
            if entry["type"] not in ("lstm", "tcn"):
                continue
            model = entry["model"]
            if len(vocab) != model.embedding.num_embeddings:
                raise ValueError(f"Vocabulary does not match neural weights in {base_dir}")
            scaler = entry.get("scaler")
            if scaler is None or scaler.n_features_in_ != entry["num_feats"]:
                raise ValueError(f"Missing/incompatible neural scaler in {base_dir}")
            expected = suite.get("metadata", {}).get("feature_columns")
            if expected is not None and list(scaler.feature_names_in_) != expected:
                raise ValueError(f"Scaler feature order differs from training metadata in {base_dir}")
        return suite

    def _build_feature_dict(self, target_act, prefix, wip, ac_wip, dt, requested_amount,
                            role_oc_array, worker_id=None, prev_dur=0.0,
                            trace_history=None, case_attrs=None):
        ctx = FeatureContext(
            target_act=target_act,
            prefix=prefix,
            wip=wip,
            ac_wip=ac_wip,
            dt=dt,
            requested_amount=requested_amount,
            role_oc=role_oc_array,
            worker_id=worker_id or "UNKNOWN",
            trace_history=trace_history or {},
            tracked_activities=getattr(self, "tracked_activities", []),
            prev_dur=float(prev_dur),
            case_attrs=case_attrs or {},
        )
        feats = self.feature_mgr.build_features(ctx)

        # Merge dynamic domain case attributes if passed
        if case_attrs:
            for k, v in case_attrs.items():
                try:
                    feats[k] = float(v)
                except (ValueError, TypeError):
                    feats[k] = v

        if self.feature_mgr.is_feature_active("case_attributes") or "RequestedAmount" in self.feature_cols:
            feats.setdefault("RequestedAmount", float(requested_amount))
        feats["Prev_Proc_Time"] = float(prev_dur)
        if worker_id:
            feats.setdefault("Resource_ID", str(worker_id))

        for col in self.feature_cols:
            if col not in feats:
                if col == "Resource_ID":
                    feats[col] = str(worker_id) if worker_id else "UNKNOWN"
                elif col == "Prev_Proc_Time":
                    feats[col] = float(prev_dur)
                else:
                    feats[col] = 0.0

        return feats

    def predict_duration(self, activity, prefix, wip, ac_wip, dt, requested_amount=0.0,
                         role_oc_array=None, worker_id=None, prev_dur=0.0,
                         trace_history=None, case_attrs=None):
        """Predicts activity execution duration in seconds using champion model."""
        act_type = self.activity_profiles.get(activity)
        if act_type in ("AUTOMATED_TASK", "INSTANT_GATEWAY"):
            return 0.0

        val = self._predict_time(self.dur_models, self.dur_dispatch, activity, prefix,
                                 wip, ac_wip, dt, requested_amount,
                                 role_oc_array or [], default_fallback=600.0,
                                 worker_id=worker_id, prev_dur=prev_dur,
                                 trace_history=trace_history, case_attrs=case_attrs)
        cap = self.dur_max_caps.get(activity, self.dur_global_cap)
        return min(val, cap)

    def predict_waiting(self, activity, prefix, wip, ac_wip, dt, requested_amount=0.0,
                        role_oc_array=None, worker_id=None, prev_dur=0.0,
                        trace_history=None, case_attrs=None):
        """Predicts activity waiting time in seconds using champion model."""
        act_type = self.activity_profiles.get(activity)
        if act_type == "INSTANT_GATEWAY":
            return 0.0

        val = self._predict_time(self.wait_models, self.wait_dispatch, activity, prefix,
                                 wip, ac_wip, dt, requested_amount,
                                 role_oc_array or [], default_fallback=3600.0,
                                 worker_id=worker_id, prev_dur=prev_dur,
                                 trace_history=trace_history, case_attrs=case_attrs)
        cap = self.wait_max_caps.get(activity, self.wait_global_cap)
        return min(val, cap)

    def _predict_time(self, suite, dispatch_table, activity, prefix, wip, ac_wip, dt,
                      requested_amount, role_oc_array, default_fallback, worker_id=None, prev_dur=0.0,
                      trace_history=None, case_attrs=None):
        entry = suite["models"].get(activity) or suite["models"].get("__default__")
        if not entry:
            return default_fallback

        suite = entry.get("suite", suite)
        m_type = entry["type"]
        model = entry["model"]
        f_dict = self._build_feature_dict(activity, prefix, wip, ac_wip, dt, requested_amount,
                                          role_oc_array,
                                          worker_id=worker_id, prev_dur=prev_dur,
                                          trace_history=trace_history, case_attrs=case_attrs)

        # Waiting and duration observe different times, so use their own fitted bounds.
        for key, ceiling in suite.get("metadata", {}).get("feature_caps", {}).items():
            if key in f_dict and isinstance(f_dict[key], (int, float)):
                f_dict[key] = min(float(f_dict[key]), ceiling)

        if m_type == "xgboost":
            p_features = prefix[-self.max_seq_len:] if len(prefix) > self.max_seq_len else prefix + ["NONE"] * (self.max_seq_len - len(prefix))
            row_dict = {}
            for i, p_act in enumerate(p_features):
                row_dict[f"Act_Minus_{self.max_seq_len - i}"] = p_act
            row_dict["Target_Activity"] = activity
            for k, v in f_dict.items():
                row_dict[k] = v

            # Determine expected columns from model booster if available
            if "expected_cols" not in entry:
                expected_cols = None
                if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
                    expected_cols = list(model.feature_names_in_)
                elif hasattr(model, "get_booster") and model.get_booster().feature_names:
                    expected_cols = list(model.get_booster().feature_names)
                elif hasattr(model, "feature_names") and model.feature_names:
                    expected_cols = list(model.feature_names)
                entry["expected_cols"] = expected_cols
            expected_cols = entry["expected_cols"]

            if expected_cols:
                for ec in expected_cols:
                    if ec not in row_dict:
                        if ec.startswith("Role_") and ec.endswith("_OC"):
                            try:
                                r_idx = int(ec.split("_")[1])
                                row_dict[ec] = float(role_oc_array[r_idx]) if r_idx < len(role_oc_array) else 0.0
                            except Exception:
                                row_dict[ec] = 0.0
                        elif ec.startswith("Weekday_"):
                            try:
                                w_idx = int(ec.split("_")[1])
                                row_dict[ec] = 1.0 if dt and dt.weekday() == w_idx else 0.0
                            except Exception:
                                row_dict[ec] = 0.0
                        elif ec == "Daytime":
                            row_dict[ec] = (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400.0 if dt else 0.0
                        elif ec == "WIP":
                            max_wip = self.feature_max_caps.get("WIP", 888.0)
                            row_dict[ec] = min(float(wip), max_wip)
                        elif ec == "AC_WIP":
                            max_ac = self.feature_max_caps.get("AC_WIP", 6.0)
                            val_ac = float(ac_wip.get(activity, 0)) if isinstance(ac_wip, dict) else 0.0
                            row_dict[ec] = min(val_ac, max_ac)
                        elif ec.startswith("Act_Minus_") or ec in ["Target_Activity", "Resource_ID"]:
                            row_dict[ec] = "NONE"
                        else:
                            row_dict[ec] = 0.0
                cols_to_use = expected_cols
            else:
                cols_to_use = [c for c in row_dict.keys() if c in self.feature_cols or c.startswith("Act_Minus_") or c == "Target_Activity"]

            prefix_map = suite.get("prefix_code_map", {})
            target_map = suite.get("target_code_map", {})
            res_map = suite.get("resource_code_map", {})

            row_vals = []
            for col in cols_to_use:
                if col.startswith("Act_Minus_"):
                    row_vals.append(prefix_map.get(row_dict.get(col, "NONE"), np.nan))
                elif col == "Target_Activity":
                    row_vals.append(target_map.get(row_dict.get(col, activity), np.nan))
                elif col == "Resource_ID":
                    row_vals.append(res_map.get(row_dict.get(col, "UNKNOWN"), np.nan))
                else:
                    row_vals.append(float(row_dict.get(col, 0.0)))

            x_arr = np.array([row_vals], dtype=np.float32)
            booster_obj = entry.get("booster") or model.get_booster()
            entry["booster"] = booster_obj
            best = booster_obj.attr("best_iteration")
            iteration_range = (0, int(best) + 1) if best is not None else (0, 0)
            pred_log1p = float(booster_obj.inplace_predict(x_arr, iteration_range=iteration_range)[0])
            val = float(np.expm1(np.clip(pred_log1p, 0.0, 20.0)))
            return max(1.0, val)

        elif m_type in ["tcn", "lstm"]:
            vocab = suite["vocab"]
            act_to_idx = suite.get("act_to_idx", {})
            seq = [vocab.get(a, 1) for a in prefix[-self.max_seq_len:]]
            if len(seq) < self.max_seq_len:
                seq = [0] * (self.max_seq_len - len(seq)) + seq
            seq_t = torch.tensor([seq], dtype=torch.long)

            if activity in act_to_idx:
                act_idx = act_to_idx[activity]
            elif activity in self.target_acts:
                act_idx = self.target_acts.index(activity)
            else:
                act_idx = 0
            act_t = torch.tensor([act_idx], dtype=torch.long)

            scaler = entry.get("scaler")
            target_n = entry.get("num_feats")
            if scaler is None:
                s_dir = entry.get("source_dir", "")
                if "global" in s_dir:
                    scaler = suite["scalers"].get("scaler_global.pkl")
                elif "local" in s_dir:
                    scaler = suite["scalers"].get(f"scaler_local_{activity}.pkl") or suite["scalers"].get("scaler_local.pkl")

                if scaler is None and target_n is not None:
                    for sc in suite["scalers"].values():
                        if hasattr(sc, "feature_names_in_") and len(sc.feature_names_in_) == target_n:
                            scaler = sc
                            break

                if scaler is None:
                    scaler = suite["scalers"].get("scaler_global.pkl") or suite["scalers"].get("scaler_local.pkl")
                entry["scaler"] = scaler

            if "scaler_mean" not in entry and scaler is not None and hasattr(scaler, "mean_"):
                entry["scaler_mean"] = scaler.mean_.astype(np.float32)
                entry["scaler_scale"] = scaler.scale_.astype(np.float32)

            if "cols" not in entry:
                entry["cols"] = list(scaler.feature_names_in_) if (scaler and hasattr(scaler, "feature_names_in_")) else self.feature_cols
            cols = entry["cols"]

            if "scaler_mean" in entry:
                feat_raw = []
                for c in cols:
                    if c in f_dict:
                        feat_raw.append(float(f_dict[c]))
                    else:
                        feat_raw.append(0.0)
                feat_arr = (np.array([feat_raw], dtype=np.float32) - entry["scaler_mean"]) / entry["scaler_scale"]
                feat_t = torch.from_numpy(feat_arr)
            else:
                feat_dict = {}
                for c in cols:
                    if c in f_dict:
                        feat_dict[c] = f_dict[c]
                    else:
                        feat_dict[c] = 0.0
                feat_df = pd.DataFrame([feat_dict])[cols]
                if scaler:
                    feat_vals = scaler.transform(feat_df)
                else:
                    feat_vals = feat_df.values.astype(np.float32)
                feat_t = torch.tensor(feat_vals, dtype=torch.float32)

            if target_n is not None and feat_t.shape[1] != target_n:
                if feat_t.shape[1] > target_n:
                    feat_t = feat_t[:, :target_n]
                else:
                    pad = torch.zeros((feat_t.shape[0], target_n - feat_t.shape[1]), dtype=feat_t.dtype)
                    feat_t = torch.cat([feat_t, pad], dim=1)

            with torch.no_grad():
                out = model(seq_t, act_t, feat_t).item()
            val = float(np.expm1(np.clip(out, 0.0, 15.0)))
            return max(1.0, val)

        return default_fallback

    def predict_routing(self, place_id, prefix, requested_amount=0.0, trace_history=None, curr_dt=None, loop_count=0.0,
                        inter_arrival_seconds=0.0, case_attrs=None):
        """Predicts outgoing transition branch at an XOR split place."""
        p_info = self.routing_manifest.get(place_id)
        if not p_info:
            return None

        # 1. ML Classifier Prediction
        force_prob = set(self.cfg.get("tasks", {}).get("routing", {}).get("force_probability_places", []))
        if place_id not in force_prob and p_info.get("strategy") == "machine_learning" and place_id in self.routing_models:
            clf = self.routing_models[place_id]
            feat_cols = p_info["features"]

            row = {
                "RequestedAmount": float(requested_amount),
                "loop_count": float(loop_count),
                "prev_activity": prefix[-1] if prefix else "<START>",
                "hour": curr_dt.hour if curr_dt is not None else 12,
                "weekday": curr_dt.weekday() if curr_dt is not None else 0,
                "inter_arrival_seconds": float(inter_arrival_seconds),
            }

            if case_attrs:
                for k, v in case_attrs.items():
                    row[k] = v

            for col in feat_cols:
                if col.startswith("has_"):
                    act_name = col.replace("has_", "")
                    act_clean = act_name.replace(" ", "_").replace("-", "_")
                    has_fired = (
                        act_name in prefix
                        or act_clean in prefix
                        or (trace_history and (
                            trace_history.get(act_name, 0) > 0
                            or trace_history.get(act_clean, 0) > 0
                            or trace_history.get(col[4:], 0) > 0
                        ))
                    )
                    row[col] = 1.0 if has_fired else 0.0
                elif col.startswith("count_"):
                    act_name = col.replace("count_", "")
                    act_clean = act_name.replace(" ", "_").replace("-", "_")
                    cnt = prefix.count(act_name)
                    if trace_history:
                        cnt = max(cnt, trace_history.get(act_name, 0), trace_history.get(act_clean, 0))
                    row[col] = float(cnt)
                elif col not in row:
                    row[col] = 0.0

            cat_map = p_info.get("cat_mappings", {})
            for c in feat_cols:
                if c in cat_map:
                    cats = cat_map[c]
                    val = row.get(c, "<UNKNOWN>")
                    row[c] = cats.index(val) if val in cats else -1

            try:
                label_enc = p_info["label_encoding"]
                idx_to_label = {v: k for k, v in label_enc.items()}
                inference_mode = self.cfg.get("tasks", {}).get("routing", {}).get("inference_mode", "sample")

                is_xgb = "xgb" in str(type(clf)).lower()
                if is_xgb:
                    row_df = pd.DataFrame([row])[feat_cols]
                    for c in row_df.columns:
                        if (c.startswith("prefix_") or c == "prev_activity") and c not in cat_map:
                            row_df[c] = row_df[c].astype("category")
                    if inference_mode == "argmax":
                        pred_code = clf.predict(row_df)[0]
                    elif hasattr(clf, "predict_proba"):
                        probas = clf.predict_proba(row_df)[0]
                        p_sum = probas.sum()
                        pred_code = np.random.choice(clf.classes_, p=probas / p_sum) if p_sum > 0 else clf.predict(row_df)[0]
                    else:
                        pred_code = clf.predict(row_df)[0]
                else:
                    x_arr = np.array([[float(row.get(c, 0.0)) for c in feat_cols]], dtype=np.float32)
                    if inference_mode == "argmax":
                        pred_code = clf.predict(x_arr)[0]
                    elif hasattr(clf, "predict_proba"):
                        probas = clf.predict_proba(x_arr)[0]
                        p_sum = probas.sum()
                        pred_code = np.random.choice(clf.classes_, p=probas / p_sum) if p_sum > 0 else clf.predict(x_arr)[0]
                    else:
                        pred_code = clf.predict(x_arr)[0]

                return idx_to_label.get(pred_code)
            except Exception as e:
                logger.warning(
                    f"[Routing] ML inference failed for decision place '{place_id}': {e}. "
                    "Falling back to empirical probability."
                )

        # 2. Empirical Probability Fallback
        probs = p_info.get("probabilities", {})
        if not probs and "classes_observed" in p_info:
            tot = sum(p_info["classes_observed"].values())
            if tot > 0:
                probs = {k: v / tot for k, v in p_info["classes_observed"].items()}
        if probs:
            targets = list(probs.keys())
            weights = list(probs.values())
            return random.choices(targets, weights=weights, k=1)[0]

        # 3. Default to first observed branch
        branches = list(p_info.get("classes_observed", {}).keys())
        return branches[0] if branches else "<END>"