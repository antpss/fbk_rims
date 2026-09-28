#!/usr/bin/env python3
"""
src/05_train_routing.py

Modernized Config-Driven XOR Decision Point Mining & Routing Classifier for RIMS+.
Replaces legacy RIMS_decision_points/decision_mining.py with:
  1. Config-driven domain-agnostic feature handling (no hardcoded dictionaries).
  2. Direct ingestion of pre-aligned / repaired logs (no trace-by-trace alignment recomputation).
  3. High-performance Gradient Boosting (XGBoost) and regularized Decision Trees.
  4. Automatic fallback to empirical branching probabilities when F1 < min_f1_threshold.
  5. Exact backwards-compatibility with RIMS simulation: saves {place_id}.pkl and {project}_decision_points.json,
     as well as the modern routing_decisions.json manifest.
"""

import os
import sys
import json
import pickle
import argparse
import joblib
import numpy as np
import pandas as pd
from lxml import etree
from collections import defaultdict

import pm4py
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import xgboost as xgb

from config_loader import load_config, PROJECT_ROOT


# =====================================================================
# Petri Net XOR Decision Point Discovery
# =====================================================================
def discover_xor_decision_points(pnml_path):
    """
    Parses a PNML Petri net file and identifies all places with >= 2 outgoing arcs (XOR splits).
    Traces silent transitions to downstream observable activities where possible.
    """
    if not os.path.exists(pnml_path):
        raise FileNotFoundError(f"Petri net not found at: {pnml_path}")

    tree = etree.parse(pnml_path)
    root = tree.getroot()
    net = root.find("net")
    page = net.find("page") if net.find("page") is not None else net

    # 1. Map transitions (ID -> display name / label)
    trans_map = {}
    is_silent = {}
    for trans in page.findall("transition"):
        t_id = trans.get("id")
        name_elem = trans.find("name")
        t_text = name_elem.find("text").text if (name_elem is not None and name_elem.find("text") is not None) else t_id

        ts = trans.find("toolspecific")
        if (ts is not None and ts.get("activity") == "$invisible$") or t_text.startswith("sfl_") or t_text == "$invisible$":
            trans_map[t_id] = t_text
            is_silent[t_id] = True
        else:
            trans_map[t_id] = t_text
            is_silent[t_id] = False

    # 2. Map arcs
    out_arcs = defaultdict(list)   # source_id -> list of target_ids
    in_arcs = defaultdict(list)    # target_id -> list of source_ids
    for arc in page.findall("arc"):
        src = arc.get("source")
        tgt = arc.get("target")
        out_arcs[src].append(tgt)
        in_arcs[tgt].append(src)

    # 3. Helper to trace downstream observable activity from a silent transition
    def resolve_target_label(t_id, visited=None):
        if visited is None:
            visited = set()
        if t_id in visited:
            return trans_map.get(t_id, t_id)
        visited.add(t_id)

        if not is_silent.get(t_id, False):
            return trans_map.get(t_id, t_id)

        # Silent transition: check downstream places and their transitions
        downstream_places = out_arcs.get(t_id, [])
        for dp in downstream_places:
            downstream_trans = out_arcs.get(dp, [])
            for dt in downstream_trans:
                if not is_silent.get(dt, False):
                    return trans_map.get(dt, dt)
        return trans_map.get(t_id, t_id)

    # 4. Identify XOR places (out_degree >= 2)
    xor_decision_points = {}
    for place in page.findall("place"):
        p_id = place.get("id")
        targets = out_arcs.get(p_id, [])
        if len(targets) >= 2:
            branch_info = {}
            for t_id in targets:
                resolved_label = resolve_target_label(t_id)
                branch_info[t_id] = {
                    "raw_trans": trans_map.get(t_id, t_id),
                    "resolved_target": resolved_label,
                    "is_silent": is_silent.get(t_id, False)
                }
            xor_decision_points[p_id] = branch_info

    return xor_decision_points


# =====================================================================
# Decision Dataset Extraction from Pre-Aligned Log
# =====================================================================
def extract_decision_samples_from_log(log_path, xor_decision_points, window_size, case_attrs_cfg):
    """
    Scans the pre-aligned log, extracting decision context (prefix window + case attributes)
    whenever an XOR decision point choice is encountered.
    """
    print(f"Loading event log for decision extraction: {log_path}...")
    log = pm4py.read_xes(log_path)

    # Prepare storage per decision place
    decision_data = {p_id: [] for p_id in xor_decision_points}

    # Reverse lookup: for a given place, map possible next activities/targets
    place_to_choices = {}
    for p_id, branches in xor_decision_points.items():
        place_to_choices[p_id] = {b["resolved_target"]: b["raw_trans"] for b in branches.values()}

    for trace in log:
        trace_attrs = trace.attributes
        events = [e["concept:name"] for e in trace]

        # Extract case attributes dynamically from config
        case_features = {}
        for attr in case_attrs_cfg:
            col = attr["column"]
            feat_name = attr["feature_name"]
            fill_val = attr.get("fill_value", 0.0)
            val = trace_attrs.get(col, fill_val)
            case_features[feat_name] = val

        # Walk through trace history
        prefix = []
        for i, act in enumerate(events):
            if i + 1 < len(events):
                next_act = events[i + 1]

                for p_id, choice_map in place_to_choices.items():
                    if next_act in choice_map:
                        recent_prefix = prefix[-window_size:] if len(prefix) >= window_size else ["<START>"] * (window_size - len(prefix)) + prefix
                        
                        sample = {
                            "case_id": trace_attrs.get("concept:name", "unknown"),
                            "current_activity": act,
                            "chosen_target": next_act,
                            **{f"prefix_{k+1}": p_act for k, p_act in enumerate(recent_prefix)},
                            **case_features
                        }
                        decision_data[p_id].append(sample)

            prefix.append(act)

    decision_dfs = {}
    for p_id, samples in decision_data.items():
        if len(samples) > 0:
            decision_dfs[p_id] = pd.DataFrame(samples)

    return decision_dfs


# =====================================================================
# Main Training Engine
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Modernized XOR Decision Point Mining & Routing Classifier for RIMS+")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--classifier", type=str, default=None, choices=["xgboost", "decision_tree"], help="Override classifier type (xgboost or decision_tree)")
    parser.add_argument("--min_samples", type=int, default=None, help="Minimum samples required to train an ML model (default from config or 30)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["project"]["random_seed"]
    project_name = cfg["project"].get("name", "BPI_2012")
    pnml_path = os.path.join(PROJECT_ROOT, cfg["paths"]["petri_net"])
    log_path = os.path.join(PROJECT_ROOT, cfg["paths"]["aligned_log"])

    routing_cfg = cfg.get("tasks", {}).get("routing", {})
    min_f1 = routing_cfg.get("min_f1_threshold", 0.60)
    clf_type = args.classifier or routing_cfg.get("classifier_type", "xgboost")
    min_samples = args.min_samples if args.min_samples is not None else routing_cfg.get("min_samples", 30)
    output_dir = os.path.join(PROJECT_ROOT, routing_cfg.get("output_dir", "models_no_zeros/routing"))
    os.makedirs(output_dir, exist_ok=True)

    window_size = cfg.get("features", {}).get("prefix_window_size", 10)
    case_attrs_cfg = cfg.get("features", {}).get("case_attributes", [])

    print("=" * 85)
    print("  RIMS+ MODERNIZED XOR DECISION MINING & ROUTING CLASSIFIER")
    print(f"  Petri Net:     {pnml_path}")
    print(f"  Event Log:     {log_path}")
    print(f"  Classifier:    {clf_type.upper()}")
    print(f"  Min Samples:   {min_samples}")
    print(f"  F1 Threshold:  {min_f1}")
    print(f"  Output Dir:    {output_dir}")
    print("=" * 85)

    # Step 1: Discover XOR Decision Places
    xor_points = discover_xor_decision_points(pnml_path)
    print(f"\nDiscovered {len(xor_points)} XOR decision point places in Petri net:")
    for p_id, branches in xor_points.items():
        targets = [b["resolved_target"] for b in branches.values()]
        print(f"  - Place [{p_id}]: {len(branches)} outgoing branches -> {targets}")

    # Step 2: Extract Decision Instances from Log
    decision_dfs = extract_decision_samples_from_log(log_path, xor_points, window_size, case_attrs_cfg)
    print(f"\nExtracted training samples across {len(decision_dfs)} decision points.")

    dispatch_manifest = {
        "classifier_type": clf_type,
        "min_f1_threshold": min_f1,
        "window_size": window_size,
        "decision_points": {}
    }

    # Exact legacy RIMS compatibility dictionary ({NAME}_decision_points.json)
    legacy_rims_data = {}

    results_table = []

    # Step 3: Train Classifier or Compute Empirical Probabilities per Decision Point
    for p_id, df in decision_dfs.items():
        safe_name = p_id.replace(" ", "_").replace(":", "_").replace("/", "_")
        targets_available = [b["resolved_target"] for b in xor_points[p_id].values()]
        n_samples = len(df)
        counts = df["chosen_target"].value_counts().to_dict()
        n_classes = len(counts)

        dp_entry = {
            "place_id": p_id,
            "sample_count": n_samples,
            "classes_observed": counts,
            "branches_defined": targets_available,
        }

        legacy_entry = {
            "transitions": targets_available,
            "padding": window_size
        }

        # Case 1: Insufficient data or single observed class -> Fallback to Empirical Probabilities
        if n_samples < min_samples or n_classes < 2:
            probs = {cls: float(cnt / n_samples) for cls, cnt in counts.items()}
            dp_entry["strategy"] = "empirical_probability"
            dp_entry["probabilities"] = probs
            dp_entry["accuracy"] = None
            dp_entry["macro_f1"] = None
            dispatch_manifest["decision_points"][p_id] = dp_entry

            legacy_entry["prediction"] = False
            legacy_entry["probability"] = probs
            legacy_rims_data[p_id] = legacy_entry

            results_table.append({
                "Place": p_id,
                "Samples": n_samples,
                "Classes": n_classes,
                "Strategy": "Empirical Prob",
                "Accuracy": "---",
                "Macro F1": "---",
                "Status": "Probability Fallback"
            })
            continue

        # Case 2: Multi-class choice with sufficient samples -> Train ML Classifier
        y_raw = df["chosen_target"]
        label_to_idx = {lbl: i for i, lbl in enumerate(sorted(y_raw.unique()))}
        idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
        y = y_raw.map(label_to_idx).values

        feat_cols = [c for c in df.columns if c not in ["case_id", "current_activity", "chosen_target"]]
        X = df[feat_cols].copy()

        cat_mappings = {}
        for c in X.columns:
            if X[c].dtype == "object":
                X[c] = X[c].astype("category")
                cat_mappings[c] = list(X[c].cat.categories)

        # Train/Test Split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed
            )

        if clf_type == "xgboost":
            clf = xgb.XGBClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                enable_categorical=True,
                random_state=seed,
                n_jobs=-1,
                eval_metric="mlogloss" if n_classes > 2 else "logloss"
            )
            clf.fit(X_train, y_train)
        else:
            clf = DecisionTreeClassifier(
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=seed
            )
            X_tr_num = X_train.copy()
            X_te_num = X_test.copy()
            for c in cat_mappings:
                X_tr_num[c] = X_tr_num[c].cat.codes
                X_te_num[c] = X_te_num[c].cat.codes
            clf.fit(X_tr_num, y_train)
            X_test = X_te_num

        y_pred = clf.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
        weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
        cm = confusion_matrix(y_test, y_pred).tolist()

        dp_entry["accuracy"] = round(acc, 4)
        dp_entry["macro_f1"] = round(macro_f1, 4)
        dp_entry["weighted_f1"] = round(weighted_f1, 4)
        dp_entry["confusion_matrix"] = cm
        dp_entry["label_encoding"] = label_to_idx
        dp_entry["features"] = feat_cols

        if macro_f1 >= min_f1:
            dp_entry["strategy"] = "machine_learning"
            model_joblib_file = f"routing_{safe_name}.joblib"
            model_pkl_file = f"{safe_name}.pkl"
            
            joblib.dump(clf, os.path.join(output_dir, model_joblib_file))
            with open(os.path.join(output_dir, model_pkl_file), "wb") as pf:
                pickle.dump(clf, pf)

            dp_entry["model_file"] = model_joblib_file
            dp_entry["legacy_pkl_file"] = model_pkl_file

            legacy_entry["prediction"] = True
            legacy_entry["Accuracy"] = acc
            legacy_entry["f1_score_macro"] = macro_f1
            legacy_entry["f1_score_weighted"] = weighted_f1
            legacy_entry["confusion_matrix"] = cm
            legacy_entry["encoding2target"] = idx_to_label

            status_str = "ML Model Saved"
        else:
            dp_entry["strategy"] = "empirical_probability"
            probs = {cls: float(cnt / n_samples) for cls, cnt in counts.items()}
            dp_entry["probabilities"] = probs
            dp_entry["model_file"] = None

            legacy_entry["prediction"] = False
            legacy_entry["probability"] = probs
            status_str = f"F1 < {min_f1} -> Prob Fallback"

        dispatch_manifest["decision_points"][p_id] = dp_entry
        legacy_rims_data[p_id] = legacy_entry

        results_table.append({
            "Place": p_id,
            "Samples": n_samples,
            "Classes": n_classes,
            "Strategy": dp_entry["strategy"],
            "Accuracy": f"{acc * 100:.1f}%",
            "Macro F1": f"{macro_f1:.4f}",
            "Status": status_str
        })

    # Save Manifests (both modern RIMS+ and legacy RIMS format)
    manifest_path = os.path.join(output_dir, "routing_decisions.json")
    with open(manifest_path, "w") as f:
        json.dump(dispatch_manifest, f, indent=2)

    legacy_json_path = os.path.join(output_dir, f"{project_name}_decision_points.json")
    with open(legacy_json_path, "w") as f:
        json.dump(legacy_rims_data, f, indent=2)

    # Print Summary Table
    print("\n" + "=" * 95)
    print(f"  RIMS+ DECISION POINT MINING & ROUTING BENCHMARK TABLE")
    print("=" * 95)
    print(f"{'Decision Point Place':<36} | {'Samples':>7} | {'Classes':>7} | {'Strategy':<15} | {'Accuracy':>8} | {'Macro F1':>8} | {'Status':<18}")
    print("-" * 95)
    for row in results_table:
        print(f"{row['Place']:<36} | {row['Samples']:>7} | {row['Classes']:>7} | {row['Strategy']:<15} | {row['Accuracy']:>8} | {row['Macro F1']:>8} | {row['Status']:<18}")
    print("=" * 95)
    print(f"Modern dispatch manifest saved to: {manifest_path}")
    print(f"Legacy RIMS JSON saved to:         {legacy_json_path}\n")


if __name__ == "__main__":
    main()
