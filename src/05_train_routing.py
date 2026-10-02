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
from pm4py.objects.petri_net import semantics
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
    Traces silent transitions to downstream observable activities or <END>/<START> recursively.
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
    for trans in page.findall(".//transition"):
        t_id = trans.get("id")
        ts = trans.find("toolspecific")
        name_elem = trans.find("name")
        t_text = name_elem.find("text").text if (name_elem is not None and name_elem.find("text") is not None) else t_id

        if (ts is not None and ts.get("activity") == "$invisible$") or t_text.startswith("sfl_") or t_text == "$invisible$" or (len(t_text) == 36 and "-" in t_text):
            trans_map[t_id] = t_text
            is_silent[t_id] = True
        else:
            trans_map[t_id] = t_text
            is_silent[t_id] = False

    # 2. Map arcs
    out_arcs = defaultdict(list)   # source_id -> list of target_ids
    in_arcs = defaultdict(list)    # target_id -> list of source_ids
    for arc in page.findall(".//arc"):
        src = arc.get("source")
        tgt = arc.get("target")
        out_arcs[src].append(tgt)
        in_arcs[tgt].append(src)

    # 3. Helper to trace downstream observable activity from a silent transition
    def resolve_target_label(t_id, visited=None):
        if visited is None:
            visited = set()
        if t_id in visited:
            return None
        visited.add(t_id)

        if not is_silent.get(t_id, False):
            return trans_map.get(t_id, t_id)

        # Silent transition: check downstream places and their transitions
        for p in out_arcs.get(t_id, []):
            if p == "sink":
                return "<END>"
            for dt in out_arcs.get(p, []):
                if not is_silent.get(dt, False):
                    return trans_map.get(dt, dt)
                res = resolve_target_label(dt, visited.copy())
                if res:
                    return res
        return trans_map.get(t_id, t_id)

    # 4. Helper to trace upstream incoming activity for an XOR place
    def resolve_incoming_label(t_id, visited=None):
        if visited is None:
            visited = set()
        if t_id in visited:
            return set()
        visited.add(t_id)

        if not is_silent.get(t_id, False):
            return {trans_map.get(t_id, t_id)}

        res = set()
        for p in in_arcs.get(t_id, []):
            if p == "source":
                res.add("<START>")
            for st in in_arcs.get(p, []):
                if not is_silent.get(st, False):
                    res.add(trans_map.get(st, st))
                else:
                    res.update(resolve_incoming_label(st, visited.copy()))
        return res

    # 5. Identify XOR places (out_degree >= 2)
    xor_decision_points = {}
    for place in page.findall(".//place"):
        p_id = place.get("id")
        targets = out_arcs.get(p_id, [])
        if len(targets) >= 2:
            in_trans = in_arcs.get(p_id, [])
            all_incoming = set()
            for s in in_trans:
                all_incoming.update(resolve_incoming_label(s))

            branch_info = {}
            for t_id in targets:
                resolved_label = resolve_target_label(t_id)
                branch_info[t_id] = {
                    "raw_trans": trans_map.get(t_id, t_id),
                    "resolved_target": resolved_label,
                    "is_silent": is_silent.get(t_id, False)
                }
            xor_decision_points[p_id] = {
                "incoming": list(all_incoming),
                "branches": branch_info
            }

    return xor_decision_points


# =====================================================================
# Decision Dataset Extraction from Pre-Aligned Log
# =====================================================================
def extract_decision_samples_from_log(log_path, xor_decision_points, window_size, case_attrs_cfg, prep_cfg=None, pnml_path=None):
    """
    Scans the pre-aligned log, extracting decision context (prefix window + case attributes
    + dynamic business milestone features) whenever an XOR decision point choice is encountered.
    Integrates Petri net marking semantics to verify active tokens at decision places.
    """
    print(f"Loading event log for decision extraction: {log_path}...")
    log = pm4py.read_xes(log_path)
    df = pm4py.convert_to_dataframe(log) if not isinstance(log, pd.DataFrame) else log

    if prep_cfg is None:
        prep_cfg = {}
    proc_prefixes = tuple(prep_cfg.get("process_activity_prefixes", []))
    milestone_prefixes = tuple(prep_cfg.get("milestone_prefixes", []))

    all_milestones = []
    if milestone_prefixes:
        all_milestones = sorted([m for m in df["concept:name"].dropna().unique() if m.startswith(milestone_prefixes)])
        print(f"Extracting context across {len(all_milestones)} business milestones.")

    # Load Petri net for marking semantics verification if provided
    net = None
    im = None
    place_map = {}
    label_to_trans = defaultdict(list)
    if pnml_path and os.path.exists(pnml_path):
        try:
            net, im, _ = pm4py.read_pnml(pnml_path)
            place_map = {p.name: p for p in net.places}
            for t in net.transitions:
                if t.label is not None:
                    label_to_trans[t.label].append(t)
            print("Petri net marking semantics enabled for decision verification.")
        except Exception as e:
            print(f"[Warning] Could not initialize Petri net semantics: {e}")

    # Prepare storage per decision place
    decision_data = {p_id: [] for p_id in xor_decision_points}

    # Group by case
    cases = df.groupby("case:concept:name", sort=False)

    def advance_silents(m):
        if m is None or net is None:
            return m
        while True:
            enabled = semantics.enabled_transitions(net, m)
            fired = False
            for t in enabled:
                if t.label is None and any(arc.target.name in xor_decision_points for arc in t.out_arcs):
                    m = semantics.execute(t, net, m)
                    fired = True
                    break
            if not fired:
                break
        return m

    for case_id, group in cases:
        grp_sorted = group.sort_values(by="time:timestamp").reset_index(drop=True)
        if len(grp_sorted) == 0:
            continue

        first_row = grp_sorted.iloc[0]

        # Extract case attributes dynamically from config
        case_features = {}
        for attr in case_attrs_cfg:
            col = attr["column"]
            feat_name = attr["feature_name"]
            fill_val = attr.get("fill_value", 0.0)
            val = first_row.get(col, fill_val)
            if pd.isna(val) and col.replace("case:", "") in first_row:
                val = first_row.get(col.replace("case:", ""), fill_val)
            try:
                val = float(val) if attr.get("type") == "continuous" else val
            except Exception:
                val = fill_val
            case_features[feat_name] = val

        # Separate milestones vs process activities
        milestones_seen = set()
        offer_count = 0
        proc_events = []
        milestone_state_at_proc = []

        for _, row in grp_sorted.iterrows():
            act = row["concept:name"]
            trans = str(row.get("lifecycle:transition", "COMPLETE")).upper()

            if milestone_prefixes and act.startswith(milestone_prefixes):
                milestones_seen.add(act)
                if act.startswith("O_"):
                    offer_count += 1
            elif (not proc_prefixes or act.startswith(proc_prefixes)) and trans == "COMPLETE":
                proc_events.append(act)
                m_state = {
                    "Offer_Count": float(offer_count),
                    "Has_Offer": 1.0 if offer_count > 0 else 0.0,
                    **{f"Milestone_{m}": (1.0 if m in milestones_seen else 0.0) for m in all_milestones}
                }
                milestone_state_at_proc.append(m_state)

        if not proc_events:
            continue

        curr_marking = im.copy() if im is not None else None
        curr_marking = advance_silents(curr_marking)

        # 1. Process Start Decision Point (<START>)
        first_act = proc_events[0]
        for p_id, p_info in xor_decision_points.items():
            if "<START>" in p_info["incoming"]:
                branch_targets = [b["resolved_target"] for b in p_info["branches"].values()]
                p_obj = place_map.get(p_id)
                marking_match = (curr_marking is not None and p_obj is not None and curr_marking.get(p_obj, 0) > 0)
                structural_match = True

                if (marking_match or structural_match) and first_act in branch_targets:
                    recent_prefix = ["<START>"] * window_size
                    sample = {
                        "case_id": case_id,
                        "current_activity": "<START>",
                        "chosen_target": first_act,
                        **{f"prefix_{k+1}": p_act for k, p_act in enumerate(recent_prefix)},
                        **case_features,
                        **milestone_state_at_proc[0]
                    }
                    decision_data[p_id].append(sample)

        # 2. Intermediate Decision Points
        prefix = []
        for i, act in enumerate(proc_events):
            next_act = proc_events[i + 1] if i + 1 < len(proc_events) else "<END>"
            m_state = milestone_state_at_proc[i]
            curr_marking = advance_silents(curr_marking)

            for p_id, p_info in xor_decision_points.items():
                branch_targets = [b["resolved_target"] for b in p_info["branches"].values()]
                p_obj = place_map.get(p_id)
                marking_match = (curr_marking is not None and p_obj is not None and curr_marking.get(p_obj, 0) > 0)
                structural_match = act in p_info["incoming"]

                if (marking_match or structural_match) and (next_act in branch_targets):
                    recent_prefix = prefix[-window_size:] if len(prefix) >= window_size else ["<START>"] * (window_size - len(prefix)) + prefix
                    sample = {
                        "case_id": case_id,
                        "current_activity": act,
                        "chosen_target": next_act,
                        **{f"prefix_{k+1}": p_act for k, p_act in enumerate(recent_prefix)},
                        **case_features,
                        **m_state
                    }
                    decision_data[p_id].append(sample)

            if curr_marking is not None and act in label_to_trans:
                enabled = semantics.enabled_transitions(net, curr_marking)
                for t in label_to_trans[act]:
                    if t in enabled:
                        curr_marking = semantics.execute(t, net, curr_marking)
                        break

            prefix.append(act)

    decision_dfs = {}
    for p_id, samples in decision_data.items():
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
    output_dir = os.path.join(PROJECT_ROOT, routing_cfg.get("output_dir", "models/routing"))
    os.makedirs(output_dir, exist_ok=True)

    window_size = cfg.get("features", {}).get("prefix_window_size", 10)
    case_attrs_cfg = cfg.get("features", {}).get("case_attributes", [])
    prep_cfg = cfg.get("preprocessing", {})

    print("=" * 85)
    print("  RIMS+ MODERNIZED XOR DECISION MINING & ROUTING CLASSIFIER")
    print(f"  Petri Net:     {pnml_path}")
    print(f"  Event Log:     {log_path}")
    print(f"  Classifier:    {clf_type.upper()}")
    print(f"  Min Samples:   {min_samples}")
    print(f"  F1 Threshold:  {min_f1}")
    print(f"  Output Dir:    {output_dir}")
    print("=" * 85)

    #Discover XOR Decision Places
    xor_points = discover_xor_decision_points(pnml_path)
    print(f"\nDiscovered {len(xor_points)} XOR decision point places in Petri net:")
    for p_id, p_info in xor_points.items():
        branches = p_info["branches"]
        targets = [b["resolved_target"] for b in branches.values()]
        print(f"  - Place [{p_id}] (After {p_info['incoming']}): {len(branches)} outgoing branches -> {targets}")

    #Extract Decision Instances from Log
    decision_dfs = extract_decision_samples_from_log(log_path, xor_points, window_size, case_attrs_cfg, prep_cfg, pnml_path=pnml_path)
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

    #Train Classifier or Compute Empirical Probabilities per Decision Point
    for p_id, df in decision_dfs.items():
        safe_name = p_id.replace(" ", "_").replace(":", "_").replace("/", "_")
        targets_available = list(set([b["resolved_target"] for b in xor_points[p_id]["branches"].values()]))
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
            if pd.api.types.is_string_dtype(X[c]) or X[c].dtype.name in ("object", "str", "string"):
                X[c] = X[c].astype("category")
                cat_mappings[c] = list(X[c].cat.categories)

        # Case-Level 80/20 Train/Test Split (prevents intra-case cross-event data leakage)
        unique_cases = np.array(list(set(df["case_id"])))
        try:
            tr_cases, te_cases = train_test_split(unique_cases, test_size=0.2, random_state=seed)
            tr_mask = df["case_id"].isin(set(tr_cases)).values
            te_mask = df["case_id"].isin(set(te_cases)).values

            X_train, y_train = X[tr_mask], y[tr_mask]
            X_test, y_test = X[te_mask], y[te_mask]

            if len(np.unique(y_train)) < n_classes or len(np.unique(y_test)) < 2:
                raise ValueError("Sparse class distribution across case split")
        except Exception:
            # Fallback to stratified row split if case split misses minority classes
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

    # Save Manifests
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
