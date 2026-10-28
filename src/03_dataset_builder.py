import os
import json
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
import pm4py
from config_loader import load_config, PROJECT_ROOT
try:
    from features import FeatureManager, FeatureContext
except ImportError:
    from src.features import FeatureManager, FeatureContext
try:
    from simulator.calendar import CalendarManager
except ImportError:
    from src.simulator.calendar import CalendarManager

def main():
    parser = argparse.ArgumentParser(description="Config-Driven Dataset Creation for RIMS++")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--task", type=str, default="duration", choices=["duration", "waiting_time"],
                        help="Task-specific dataset to generate ('duration' or 'waiting_time')")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task_name = args.task
    
    if task_name not in cfg.get("tasks", {}):
        raise ValueError(f"Task '{task_name}' is not configured in config.yaml under 'tasks'")
        
    task_cfg = cfg["tasks"][task_name]
    print(f"============================================================")
    print(f"  RIMS++ Dataset Builder — Task: [{task_name.upper()}]")
    print(f"  Description: {task_cfg.get('description', '')}")
    print(f"============================================================")

    log_path = cfg["paths"]["aligned_log"]
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Aligned log not found at: {log_path}")

    print(f"Loading aligned log: {log_path}")
    log = pm4py.read_xes(log_path)
    df = pm4py.convert_to_dataframe(log)

    prep_cfg = cfg.get("preprocessing", {})
    proc_prefixes = tuple(prep_cfg.get("process_activity_prefixes", []))
    milestone_prefixes = tuple(prep_cfg.get("milestone_prefixes", []))

    activities = cfg.get("target_activities", [])
    if not activities:
        if proc_prefixes:
            activities = sorted([a for a in df['concept:name'].dropna().unique() if a.startswith(proc_prefixes)])
        else:
            activities = sorted(df['concept:name'].dropna().unique().tolist())
        print(f"Target activities: [AUTO-DISCOVERED {len(activities)} activities from log]: {activities}")
    else:
        print(f"Target activities ({len(activities)} configured): {activities}")

    # Discover milestone activities for dynamic business state context
    if milestone_prefixes:
        all_milestones = sorted([m for m in df['concept:name'].dropna().unique() if m.startswith(milestone_prefixes)])
        print(f"Discovered {len(all_milestones)} business milestone activities for context: {all_milestones}")
    else:
        all_milestones = []

    # Feature extraction settings
    feat_cfg = cfg.get("features", {})
    prefix_window = feat_cfg.get("prefix_window_size", 10)
    case_attrs = feat_cfg.get("case_attributes", [])
    event_attrs = feat_cfg.get("event_attributes", [])

    # Precompute empirical imputation statistics (median/mode) for case attributes
    attr_imputation = {}
    for attr in case_attrs:
        col = attr["column"]
        clean_col = col.replace("case:", "")
        series = df[col] if col in df.columns else (df[clean_col] if clean_col in df.columns else pd.Series(dtype=float))
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
    if attr_imputation:
        print(f"Empirical Imputation Defaults for Case Attributes: {attr_imputation}")

    feature_mgr = FeatureManager(cfg)
    calendar = CalendarManager.from_config(cfg)
    print(f"Prefix Window: {prefix_window}")
    print(f"Calendar Active: {calendar.enabled} ({calendar.hour_start}:00 - {calendar.hour_end}:00, Days: {calendar.work_days})")
    print(f"Selected Feature Extractors ({len(feature_mgr.selected_features)}): {feature_mgr.selected_features}")
    print(f"Domain Case Attributes (Tier 2): {[c['feature_name'] for c in case_attrs]}")
    print(f"Domain Event Attributes (Tier 2): {event_attrs}")

    # Pre-calculate original complete distributions for sample weighting
    if 'lifecycle:transition' in df.columns:
        trans_col = df['lifecycle:transition'].astype(str).str.upper()
        completes_df = df[(df['concept:name'].isin(activities)) & (trans_col == 'COMPLETE')]
        if 'is_synthetic' in completes_df.columns:
            completes_df = completes_df[completes_df['is_synthetic'].astype(str).str.lower() != 'true']
        if len(completes_df) == 0:
            completes_df = df[df['concept:name'].isin(activities)]
    else:
        completes_df = df[df['concept:name'].isin(activities)]

    orig_activity_counts = completes_df['concept:name'].value_counts().to_dict()
    total_orig_completes = max(1, len(completes_df))

    print("Sorting log chronologically (START before COMPLETE for equal timestamps)...")
    trans_order = {"START": 0, "COMPLETE": 1}
    df["_trans_rank"] = df["lifecycle:transition"].astype(str).str.upper().map(trans_order).fillna(2)
    df = df.sort_values(by=["time:timestamp", "_trans_rank"]).drop(columns=["_trans_rank"]).reset_index(drop=True)

    active_cases = set()
    models_dir_rel = cfg.get("paths", {}).get("models_dir", "models")
    role_file = os.path.join(PROJECT_ROOT, models_dir_rel, "role_mapping.json")
    if os.path.exists(role_file):
        with open(role_file, 'r') as f:
            role_data = json.load(f)
    else:
        role_data = {}

    if "activity_to_role" not in role_data or "roles" not in role_data:
        print(f"Role mapping in {role_file} is missing or legacy. Running Sperduti organizational role discovery...")
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("discover_roles", os.path.join(PROJECT_ROOT, "src", "01b_discover_roles.py"))
            disc_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(disc_mod)
            role_data = disc_mod.discover_roles(cfg)
        except Exception as e:
            print(f"Warning: Could not automatically run Sperduti role discovery: {e}")
            role_data = {}

    activity_to_role = role_data.get("activity_to_role", {act: f"Role_{i}" for i, act in enumerate(activities)})
    role_names = role_data.get("__role_names__") or list(role_data.get("roles", {}).keys()) or sorted(list(set(activity_to_role.values())))
    role_capacities = role_data.get("role_capacities", {r: 1 for r in role_names})
    print(f"Discovered Organizational Roles ({len(role_names)}): {role_names}")
    print(f"Role staffing capacities: {role_capacities}")
    role_active = defaultdict(int)

    ac_wip = defaultdict(int)
    case_event_counts = df['case:concept:name'].value_counts().to_dict()
    seen_events = {case_id: 0 for case_id in case_event_counts}

    case_prefixes = {}
    case_start_times = defaultdict(lambda: defaultdict(list))
    case_start_features = defaultdict(lambda: defaultdict(list))
    case_milestones_seen = defaultdict(set)
    case_offer_count = defaultdict(int)
    case_prev_time = {}
    case_last_complete = {}
    case_prev_end_time = {}

    all_rows = []
    synthetic_skipped = 0
    missing_start_skipped = 0
    print("Sweeping log chronologically to construct state vectors...")

    for _, row in df.iterrows():
        case_id = row["case:concept:name"]
        activity_name = row["concept:name"]
        transition = str(row.get("lifecycle:transition", "COMPLETE")).upper()
        timestamp = row["time:timestamp"]
        is_synthetic = str(row.get("is_synthetic", "false")).lower() == "true"

        # Dynamic extraction of Tier 2 case attributes
        extracted_case_attrs = {}
        for attr in case_attrs:
            col = attr["column"]
            clean_col = col.replace("case:", "")
            feat_name = attr["feature_name"]
            default_val = attr_imputation.get(feat_name, attr.get("fill_value", 0.0))
            val = row.get(col, row.get(clean_col, default_val))
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
            extracted_case_attrs[feat_name] = val

        # Dynamic extraction of Tier 2 event attributes
        extracted_event_attrs = {}
        for col in event_attrs:
            extracted_event_attrs[col] = row.get(col, "UNKNOWN")

        if seen_events[case_id] == 0:
            active_cases.add(case_id)
            case_prev_time[case_id] = 0.0
            case_last_complete[case_id] = timestamp

        seen_events[case_id] += 1

        wip = len(active_cases)
        weekday = timestamp.weekday()
        daytime = (timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second) / 86400.0
        rp_oc_array = [min(1.0, role_active[r] / max(1, role_capacities.get(r, 1))) for r in role_names]

        if case_id not in case_prefixes:
            case_prefixes[case_id] = []

        # -------------------------------------------------------------
        # A) Milestone Event: update dynamic case business state
        # -------------------------------------------------------------
        if milestone_prefixes and activity_name.startswith(milestone_prefixes):
            case_milestones_seen[case_id].add(activity_name)
            # count discrete offer instances (O_CREATED)
            if any(p.startswith("O_") for p in milestone_prefixes) and activity_name == "O_CREATED":
                case_offer_count[case_id] += 1

        # -------------------------------------------------------------
        # B) Process Work Item Execution
        # -------------------------------------------------------------
        elif activity_name in activities or (proc_prefixes and activity_name.startswith(proc_prefixes)):
            if is_synthetic:
                synthetic_skipped += 1
                case_prefixes[case_id].append(activity_name)

            elif transition == "START":
                ac_wip[activity_name] += 1
                act_role = activity_to_role.get(activity_name)
                if act_role:
                    role_active[act_role] += 1
                case_start_times[case_id][activity_name].append(timestamp)

                # Universal Net Queue Wait Calculation:
                # Effective business working waiting time between previous task completion and current task start
                if case_id in case_prev_end_time:
                    current_wait = calendar.calculate_net_working_seconds(
                        case_prev_end_time[case_id], timestamp, role_or_act=activity_name
                    )
                else:
                    current_wait = 0.0

                current_wait = max(0.0, float(current_wait))

                ctx = FeatureContext(
                    target_act=activity_name,
                    wip=wip,
                    ac_wip=ac_wip,
                    dt=timestamp,
                    role_oc=rp_oc_array,
                    role_names=role_names,
                    prev_dur=case_prev_time[case_id],
                    offer_count=case_offer_count[case_id],
                    has_offer=1.0 if case_offer_count[case_id] > 0 else 0.0,
                    milestones_seen=case_milestones_seen[case_id],
                    all_milestones=all_milestones,
                    activities=activities,
                    case_attrs=extracted_case_attrs,
                    event_attrs=extracted_event_attrs,
                    worker_id=row.get("org:resource", "UNKNOWN")
                )
                feat_dict = feature_mgr.build_features(ctx)
                feat_dict["Wait_Time_Seconds"] = current_wait
                feat_dict["Prev_Proc_Time"] = case_prev_time[case_id]

                case_start_features[case_id][activity_name].append(feat_dict)

            elif transition == "COMPLETE":
                if ac_wip[activity_name] > 0:
                    ac_wip[activity_name] -= 1
                act_role = activity_to_role.get(activity_name)
                if act_role and role_active[act_role] > 0:
                    role_active[act_role] -= 1

                if case_start_times[case_id][activity_name]:
                    start_ts = case_start_times[case_id][activity_name].pop(0)
                    duration = calendar.calculate_net_working_seconds(start_ts, timestamp, role_or_act=activity_name)
                    has_valid_start = True
                else:
                    duration = 0.0
                    has_valid_start = False

                case_prev_time[case_id] = duration
                case_last_complete[case_id] = timestamp
                case_prev_end_time[case_id] = timestamp

                # Only emit valid human training samples
                if not has_valid_start:
                    missing_start_skipped += 1
                    if case_start_features[case_id][activity_name]:
                        case_start_features[case_id][activity_name].pop(0)
                else:
                    feat = case_start_features[case_id][activity_name].pop(0) if case_start_features[case_id][activity_name] else None
                    if feat is not None:
                        row_dict = {
                            "Case_ID": case_id,
                            "Target_Activity": activity_name,
                            "Prefix": json.dumps(case_prefixes[case_id][-prefix_window:]),
                            "Duration_Seconds": duration,
                            "Wait_Time_Seconds": feat.get("Wait_Time_Seconds", 0.0)
                        }

                        # Merge all dynamically extracted features
                        for k, v in feat.items():
                            if k not in row_dict:
                                row_dict[k] = v

                        all_rows.append(row_dict)

                case_prefixes[case_id].append(activity_name)

        if seen_events[case_id] == case_event_counts[case_id]:
            active_cases.discard(case_id)
            if case_id in case_prefixes: del case_prefixes[case_id]
            if case_id in case_start_times: del case_start_times[case_id]
            if case_id in case_start_features: del case_start_features[case_id]
            if case_id in case_milestones_seen: del case_milestones_seen[case_id]
            if case_id in case_offer_count: del case_offer_count[case_id]
            if case_id in case_prev_time: del case_prev_time[case_id]
            if case_id in case_last_complete: del case_last_complete[case_id]
            if case_id in case_prev_end_time: del case_prev_end_time[case_id]

    print(f"\nExtraction Summary:")
    print(f"  - Extracted valid rows: {len(all_rows)}")
    print(f"  - Synthetic model moves skipped: {synthetic_skipped}")
    print(f"  - Incomplete work items (missing START) skipped: {missing_start_skipped}")

    task_df = pd.DataFrame(all_rows)
    total_before = len(task_df)

    # -------------------------------------------------------------
    # Config-Driven Zero Handling
    # -------------------------------------------------------------
    target_col = task_cfg["target_column"]
    filter_zeros = task_cfg.get("filter_zero_duration", False)

    if filter_zeros:
        # Task specifies filtering zeros (e.g. human duration prediction)
        task_df = task_df[task_df[target_col] > 0].reset_index(drop=True)
        total_after = len(task_df)
        print(f"\n[Zero-Filtering ENABLED] Filtered zero {target_col} records:")
        print(f"  {total_before} -> {total_after} samples ({total_before - total_after} dropped)")
    else:
        # Task retains zeros (e.g. waiting time immediate pickup, routing)
        print(f"\n[Zero-Filtering DISABLED] Retained all samples including zeros ({total_before} samples).")

    # -------------------------------------------------------------
    # Config-Driven Sample Weighting
    # -------------------------------------------------------------
    weight_cfg = task_cfg.get("sample_weighting", {})
    if weight_cfg.get("enabled", False):
        smoothing = weight_cfg.get("smoothing", "sqrt")
        max_cap = weight_cfg.get("max_weight_cap", 8.0)
        train_counts = task_df['Target_Activity'].value_counts().to_dict()
        total_train = len(task_df)

        print(f"\n--- Activity Sample Weights Calculation (Smoothing={smoothing}, Cap={max_cap}) ---")
        weights_dict = {}
        for act in sorted(train_counts.keys()):
            p_orig = orig_activity_counts.get(act, 1) / total_orig_completes
            p_train = train_counts.get(act, 1) / total_train
            ratio = p_orig / p_train

            if smoothing == "sqrt":
                smoothed_w = min(np.sqrt(ratio), max_cap)
            elif smoothing == "linear":
                smoothed_w = min(ratio, max_cap)
            else:
                smoothed_w = 1.0

            weights_dict[act] = smoothed_w
            print(f"  {act:32s}: Train={train_counts[act]:5d} | P_orig={p_orig:.4f} | P_train={p_train:.4f} | Ratio={ratio:6.2f} -> Sqrt_Weight={smoothed_w:5.2f}")

        task_df['Sample_Weight'] = task_df['Target_Activity'].map(weights_dict)
        mean_w = task_df['Sample_Weight'].mean()
        task_df['Sample_Weight'] = task_df['Sample_Weight'] / mean_w
        normalized_weights = {k: v / mean_w for k, v in weights_dict.items()}

        print("\nNormalized Weights (Mean = 1.0):")
        for act, w in normalized_weights.items():
            print(f"  {act:32s}: {w:.3f}")

        weights_out = weight_cfg.get("output_weights_file")
        if weights_out:
            weights_out_path = os.path.join(cfg["paths"]["output_base_dir"], weights_out)
            os.makedirs(os.path.dirname(weights_out_path), exist_ok=True)
            with open(weights_out_path, "w") as f:
                json.dump(normalized_weights, f, indent=2)
            print(f"Saved activity weights to: {weights_out_path}")
    else:
        task_df['Sample_Weight'] = 1.0
        print("\n[Sample Weighting DISABLED] Set Sample_Weight = 1.0 for all samples.")

    # Save final dataset
    out_rel = task_cfg["output_file"]
    out_path = os.path.join(cfg["paths"]["output_base_dir"], out_rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    task_df.to_csv(out_path, index=False)
    print(f"\nSaved [{task_name}] dataset to: {out_path} ({len(task_df)} rows)")
    print(f"Dataset creation for task '{task_name}' complete.\n")

if __name__ == "__main__":
    main()
