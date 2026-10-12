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
from datetime import datetime

logger = logging.getLogger("rims_predictor")

import numpy as np
import pandas as pd
import joblib
import torch
import xgboost as xgb

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

        # 1. Load Duration Models (hybrid_champion or active candidate directory)
        dur_base = os.path.join(self.models_dir, "duration")
        dur_dir = os.path.join(dur_base, "hybrid_champion")
        if not os.path.exists(dur_dir) and os.path.exists(dur_base):
            for cand in sorted(os.listdir(dur_base)):
                alt = os.path.join(dur_base, cand)
                if os.path.isdir(alt) and os.path.exists(os.path.join(alt, "dispatch_config.json")):
                    dur_dir = alt
                    break
        with open(os.path.join(dur_dir, "dispatch_config.json")) as f:
            self.dur_dispatch = json.load(f)["dispatch"]
        self.dur_models = self._load_model_suite(dur_dir, self.dur_dispatch)

        # 2. Load Waiting Time Models (hybrid_champion or active candidate directory)
        wait_base = os.path.join(self.models_dir, "waiting_time")
        wait_dir = os.path.join(wait_base, "hybrid_champion")
        if not os.path.exists(wait_dir) and os.path.exists(wait_base):
            for cand in sorted(os.listdir(wait_base)):
                alt = os.path.join(wait_base, cand)
                if os.path.isdir(alt) and os.path.exists(os.path.join(alt, "dispatch_config.json")):
                    wait_dir = alt
                    break
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
            if p_info.get("strategy") == "machine_learning":
                m_file = p_info.get("model_file") or p_info.get("legacy_pkl_file")
                if m_file:
                    m_path = os.path.join(routing_dir, m_file)
                    if os.path.exists(m_path):
                        try:
                            self.routing_models[p_id] = joblib.load(m_path)
                        except Exception as e_job:
                            try:
                                import pickle
                                with open(m_path, "rb") as pf:
                                    self.routing_models[p_id] = pickle.load(pf)
                            except Exception as e_pkl:
                                logger.warning(
                                    f"Failed to load routing model for place '{p_id}' from '{m_path}'. "
                                    f"joblib error: {e_job}; pickle error: {e_pkl}. Falling back to empirical probabilities."
                                )
                    else:
                        logger.warning(
                            f"Routing model file '{m_file}' for place '{p_id}' not found at '{m_path}'."
                        )

        # Cache feature columns from training dataset
        self.feature_mgr = FeatureManager(cfg)
        dur_ds_path = os.path.join(cfg["paths"]["output_base_dir"], cfg["tasks"]["duration"]["output_file"])
        sample_df = pd.read_csv(dur_ds_path, nrows=5)
        ignored = {
            "Case_ID", "case_id", "Prefix", "Duration_Seconds", "Wait_Time_Seconds",
            "Target_Duration", "Sample_Weight", "Target_Activity"
        }
        if not self.feature_mgr.is_feature_active("resource_id"):
            ignored.add("Resource_ID")
        self.feature_cols = [c for c in sample_df.columns if c not in ignored]
        self.all_milestones = [c.replace("Milestone_", "") for c in self.feature_cols if c.startswith("Milestone_")]

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
                suite["models"][act] = {"type": "tcn", "model": tcn}
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
                suite["models"][act] = {"type": "lstm", "model": lstm}
        return suite

    def _build_feature_dict(self, target_act, prefix, wip, ac_wip, dt, requested_amount,
                            milestones_seen, offer_count, has_offer, role_oc_array, worker_id=None, prev_dur=0.0):
        ctx = FeatureContext(
            target_act=target_act,
            prefix=prefix,
            wip=wip,
            ac_wip=ac_wip,
            dt=dt,
            requested_amount=requested_amount,
            milestones_seen=milestones_seen,
            offer_count=offer_count,
            has_offer=has_offer,
            role_oc=role_oc_array,
            worker_id=worker_id or "UNKNOWN",
            all_milestones=self.all_milestones,
            prev_dur=float(prev_dur),
        )
        feats = self.feature_mgr.build_features(ctx)

        # Ensure universal context variables are available in feats dictionary
        feats.setdefault("Offer_Count", float(offer_count))
        feats.setdefault("Has_Offer", float(has_offer))
        feats.setdefault("RequestedAmount", float(requested_amount))
        feats["Prev_Proc_Time"] = float(prev_dur)
        if worker_id:
            feats.setdefault("Resource_ID", str(worker_id))

        for col in self.feature_cols:
            if col not in feats:
                if col.startswith("Milestone_"):
                    m_name = col.replace("Milestone_", "")
                    feats[col] = 1.0 if m_name in milestones_seen else 0.0
                elif col == "Resource_ID":
                    feats[col] = str(worker_id) if worker_id else "UNKNOWN"
                elif col == "Prev_Proc_Time":
                    feats[col] = float(prev_dur)
                else:
                    feats[col] = 0.0
        return feats

    def predict_duration(self, activity, prefix, wip, ac_wip, dt, requested_amount,
                         milestones_seen, offer_count, has_offer, role_oc_array, worker_id=None, prev_dur=0.0):
        """Predicts activity execution duration in seconds using champion model."""
        return self._predict_time(self.dur_models, self.dur_dispatch, activity, prefix,
                                  wip, ac_wip, dt, requested_amount, milestones_seen,
                                  offer_count, has_offer, role_oc_array, default_fallback=600.0,
                                  worker_id=worker_id, prev_dur=prev_dur)

    def predict_waiting(self, activity, prefix, wip, ac_wip, dt, requested_amount,
                        milestones_seen, offer_count, has_offer, role_oc_array, worker_id=None, prev_dur=0.0):
        """Predicts activity waiting time in seconds using champion model."""
        return self._predict_time(self.wait_models, self.wait_dispatch, activity, prefix,
                                  wip, ac_wip, dt, requested_amount, milestones_seen,
                                  offer_count, has_offer, role_oc_array, default_fallback=3600.0,
                                  worker_id=worker_id, prev_dur=prev_dur)

    def _predict_time(self, suite, dispatch_table, activity, prefix, wip, ac_wip, dt,
                      requested_amount, milestones_seen, offer_count, has_offer,
                      role_oc_array, default_fallback, worker_id=None, prev_dur=0.0):
        entry = suite["models"].get(activity) or suite["models"].get("__default__")
        if not entry:
            return default_fallback

        m_type = entry["type"]
        model = entry["model"]
        f_dict = self._build_feature_dict(activity, prefix, wip, ac_wip, dt, requested_amount,
                                          milestones_seen, offer_count, has_offer, role_oc_array,
                                          worker_id=worker_id, prev_dur=prev_dur)

        if m_type == "xgboost":
            p_features = prefix[-self.max_seq_len:] if len(prefix) > self.max_seq_len else prefix + ["NONE"] * (self.max_seq_len - len(prefix))
            row_dict = {}
            for i, p_act in enumerate(p_features):
                row_dict[f"Act_Minus_{self.max_seq_len - i}"] = p_act
            row_dict["Target_Activity"] = activity
            for k, v in f_dict.items():
                row_dict[k] = v

            # Determine expected columns from model booster if available
            expected_cols = None
            if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
                expected_cols = list(model.feature_names_in_)
            elif hasattr(model, "get_booster") and model.get_booster().feature_names:
                expected_cols = list(model.get_booster().feature_names)
            elif hasattr(model, "feature_names") and model.feature_names:
                expected_cols = list(model.feature_names)

            if expected_cols:
                for ec in expected_cols:
                    if ec not in row_dict:
                        if ec.startswith("Milestone_"):
                            m_name = ec[len("Milestone_"):]
                            row_dict[ec] = 1.0 if m_name in milestones_seen else 0.0
                        elif ec.startswith("Role_") and ec.endswith("_OC"):
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
                            row_dict[ec] = float(wip)
                        elif ec == "AC_WIP":
                            row_dict[ec] = float(ac_wip.get(activity, 0)) if isinstance(ac_wip, dict) else 0.0
                        elif ec.startswith("Act_Minus_") or ec in ["Target_Activity", "Resource_ID"]:
                            row_dict[ec] = "NONE"
                        else:
                            row_dict[ec] = 0.0
                cols_to_use = expected_cols
            else:
                cols_to_use = [c for c in row_dict.keys() if c in self.feature_cols or c.startswith("Act_Minus_") or c == "Target_Activity"]

            row_df = pd.DataFrame([{c: row_dict.get(c, 0.0) for c in cols_to_use}])[cols_to_use]
            for col in row_df.columns:
                if col.startswith("Act_Minus_") or col in ["Target_Activity", "Resource_ID"]:
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
            feat_dict = {}
            for c in cols:
                if c in f_dict:
                    feat_dict[c] = f_dict[c]
                elif c.startswith("Milestone_"):
                    m_name = c[len("Milestone_"):]
                    feat_dict[c] = 1.0 if m_name in milestones_seen else 0.0
                elif c == "Offer_Count":
                    feat_dict[c] = float(offer_count)
                elif c == "Has_Offer":
                    feat_dict[c] = float(has_offer)
                else:
                    feat_dict[c] = 0.0
            feat_df = pd.DataFrame([feat_dict])[cols]
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
            cat_map = p_info.get("cat_mappings", {})
            for c in feat_cols:
                if c in cat_map:
                    cats = cat_map[c]
                    val = row.get(c, "<UNKNOWN>")
                    row[c] = cats.index(val) if val in cats else -1

            row_df = pd.DataFrame([row])[feat_cols]
            for c in row_df.columns:
                if (c.startswith("prefix_") or c == "prev_activity") and c not in cat_map:
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
            except Exception as e:
                logger.warning(
                    f"[Routing] ML inference failed for decision place '{place_id}': {e}. "
                    "Falling back to empirical probability."
                )

        # 2. Empirical Probability Fallback
        probs = p_info.get("probabilities", {})
        if probs:
            targets = list(probs.keys())
            weights = list(probs.values())
            return random.choices(targets, weights=weights, k=1)[0]

        # 3. Default to first observed branch
        branches = list(p_info.get("classes_observed", {}).keys())
        return branches[0] if branches else "<END>"