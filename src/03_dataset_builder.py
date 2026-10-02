import os
import json
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
import pm4py
from config_loader import load_config, PROJECT_ROOT

def main():
    parser = argparse.ArgumentParser(description="Config-Driven Dataset Creation for RIMS+")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--task", type=str, default="duration", choices=["duration", "waiting_time", "routing"],
                        help="Task-specific dataset to generate (duration, waiting_time, routing)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task_name = args.task
    
    if task_name not in cfg.get("tasks", {}):
        raise ValueError(f"Task '{task_name}' is not configured in config.yaml under 'tasks'")
        
    task_cfg = cfg["tasks"][task_name]
    print(f"============================================================")
    print(f"  RIMS+ Dataset Builder — Task: [{task_name.upper()}]")
    print(f"  Description: {task_cfg.get('description', '')}")
    print(f"============================================================")

    log_path = cfg["paths"]["aligned_log"]
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Aligned log not found at: {log_path}")

    print(f"Loading aligned log: {log_path}")
    log = pm4py.read_xes(log_path)
    df = pm4py.convert_to_dataframe(log)

    prep_cfg = cfg.get("preprocessing", {})
    proc_prefixes = tuple(prep_cfg.get("process_activity_prefixes", ["W_"]))
    milestone_prefixes = tuple(prep_cfg.get("milestone_prefixes", ["A_", "O_"]))

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
    univ_cfg = feat_cfg.get("universal", {})
    case_attrs = feat_cfg.get("case_attributes", [])
    event_attrs = feat_cfg.get("event_attributes", [])

    print(f"Prefix Window: {prefix_window}")
    print(f"Universal Features: WIP={univ_cfg.get('use_wip')}, Roles={univ_cfg.get('use_role_occupancy')}, Calendar={univ_cfg.get('use_calendar_time')}")
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

    print("Sorting log chronologically...")
    df = df.sort_values(by="time:timestamp").reset_index(drop=True)

    active_cases = set()
    role_file = os.path.join(PROJECT_ROOT, "models/role_mapping.json")
    if os.path.exists(role_file):
        with open(role_file, 'r') as f:
            role_data = json.load(f)
    else:
        print("models/role_mapping.json not found. Automatically discovering resource pools from log...")
        role_data = {}
        for act in activities:
            act_df = df[(df["concept:name"] == act) & (df["org:resource"].notna()) & (df["org:resource"].astype(str).str.upper() != "SYNTHETIC")]
            workers = sorted([str(r) for r in act_df["org:resource"].unique().tolist()])
            role_data[act] = workers if workers else ["DEFAULT_WORKER"]
        os.makedirs(os.path.dirname(role_file), exist_ok=True)
        with open(role_file, 'w') as f:
            json.dump(role_data, f, indent=2)
        print(f"Generated {role_file} with resource pools for {len(role_data)} activities.")
    activity_capacity = {act: max(1, len(role_data.get(act, [1]))) for act in activities}
    print(f"Activity resource capacities (Role Pools): {activity_capacity}")

    ac_wip = defaultdict(int)
    case_event_counts = df['case:concept:name'].value_counts().to_dict()
    seen_events = {case_id: 0 for case_id in case_event_counts}

    case_prefixes = {}
    case_start_times = defaultdict(dict)
    case_schedule_times = defaultdict(dict)
    case_start_features = defaultdict(dict)
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
            val = row.get(col, attr.get("fill_value", 0.0))
            if pd.isna(val) and col.replace("case:", "") in row:
                val = row.get(col.replace("case:", ""), attr.get("fill_value", 0.0))
            try:
                val = float(val) if attr.get("type") == "continuous" else val
            except Exception:
                val = attr.get("fill_value", 0.0)
            extracted_case_attrs[attr["feature_name"]] = val

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
        rp_oc_array = [ac_wip[act] / activity_capacity.get(act, 1) for act in activities]

        if case_id not in case_prefixes:
            case_prefixes[case_id] = []

        # -------------------------------------------------------------
        # A) Milestone Event: update dynamic case business state
        # -------------------------------------------------------------
        if milestone_prefixes and activity_name.startswith(milestone_prefixes):
            case_milestones_seen[case_id].add(activity_name)
            if activity_name.startswith("O_"):
                case_offer_count[case_id] += 1

        # -------------------------------------------------------------
        # B) Process Work Item Execution
        # -------------------------------------------------------------
        elif activity_name in activities or (proc_prefixes and activity_name.startswith(proc_prefixes)):
            if is_synthetic:
                synthetic_skipped += 1
                case_prefixes[case_id].append(activity_name)

            elif transition == "SCHEDULE":
                case_schedule_times[case_id][activity_name] = timestamp

            elif transition == "START":
                ac_wip[activity_name] += 1
                case_start_times[case_id][activity_name] = timestamp

                # True Queue Wait Calculation:
                # 1. Scheduled task: START - SCHEDULE (worklist queue wait)
                # 2. Unscheduled follow-up/loop: START - prev_end_time (inter-activity idle wait)
                if activity_name in case_schedule_times[case_id]:
                    current_wait = (timestamp - case_schedule_times[case_id][activity_name]).total_seconds()
                    del case_schedule_times[case_id][activity_name]
                elif case_id in case_prev_end_time:
                    current_wait = (timestamp - case_prev_end_time[case_id]).total_seconds()
                else:
                    current_wait = 0.0

                current_wait = max(0.0, current_wait)

                feat_dict = {
                    "Weekday": weekday,
                    "Daytime": daytime,
                    "WIP": wip,
                    "AC_WIP": ac_wip[activity_name],
                    "RP_OC_Array": rp_oc_array,
                    "Prev_Proc_Time": case_prev_time[case_id],
                    "Wait_Time_Seconds": current_wait,
                    "Offer_Count": float(case_offer_count[case_id]),
                    "Has_Offer": 1.0 if case_offer_count[case_id] > 0 else 0.0,
                    **extracted_case_attrs,
                    **extracted_event_attrs
                }
                for m in all_milestones:
                    feat_dict[f"Milestone_{m}"] = 1.0 if m in case_milestones_seen[case_id] else 0.0

                case_start_features[case_id][activity_name] = feat_dict

            elif transition == "COMPLETE":
                if ac_wip[activity_name] > 0:
                    ac_wip[activity_name] -= 1

                if activity_name in case_start_times[case_id]:
                    duration = (timestamp - case_start_times[case_id][activity_name]).total_seconds()
                    has_valid_start = True
                    del case_start_times[case_id][activity_name]
                else:
                    duration = 0.0
                    has_valid_start = False

                case_prev_time[case_id] = duration
                case_last_complete[case_id] = timestamp
                case_prev_end_time[case_id] = timestamp

                # Only emit valid human training samples
                if not has_valid_start:
                    missing_start_skipped += 1
                    case_start_features[case_id].pop(activity_name, None)
                else:
                    feat = case_start_features[case_id].pop(activity_name, None)
                    if feat is not None:
                        row_dict = {
                            "Case_ID": case_id,
                            "Target_Activity": activity_name,
                            "Prefix": json.dumps(case_prefixes[case_id][-prefix_window:]),
                            "Duration_Seconds": duration,
                            "Wait_Time_Seconds": feat.get("Wait_Time_Seconds", 0.0)
                        }

                        if univ_cfg.get("use_prev_duration", True):
                            row_dict["Prev_Proc_Time"] = feat.get("Prev_Proc_Time", 0.0)
                        if univ_cfg.get("use_wip", True):
                            row_dict["WIP"] = feat.get("WIP", wip)
                        if univ_cfg.get("use_ac_wip", True):
                            row_dict["AC_WIP"] = feat.get("AC_WIP", 0)
                        if univ_cfg.get("use_calendar_time", True):
                            row_dict["Daytime"] = feat.get("Daytime", daytime)
                            for i in range(7):
                                row_dict[f"Weekday_{i}"] = 1.0 if feat["Weekday"] == i else 0.0
                        if univ_cfg.get("use_role_occupancy", True):
                            for i, act in enumerate(activities):
                                row_dict[f"Role_{i}_OC"] = float(feat["RP_OC_Array"][i])

                        # Append dynamic milestone context
                        row_dict["Offer_Count"] = feat.get("Offer_Count", 0.0)
                        row_dict["Has_Offer"] = feat.get("Has_Offer", 0.0)
                        for m in all_milestones:
                            row_dict[f"Milestone_{m}"] = feat.get(f"Milestone_{m}", 0.0)

                        # Append domain-specific Tier 2 features
                        for attr in case_attrs:
                            fname = attr["feature_name"]
                            row_dict[fname] = feat.get(fname, attr.get("fill_value", 0.0))
                        for col in event_attrs:
                            row_dict[col] = feat.get(col, "UNKNOWN")

                        all_rows.append(row_dict)

                case_prefixes[case_id].append(activity_name)

        if seen_events[case_id] == case_event_counts[case_id]:
            active_cases.discard(case_id)
            if case_id in case_prefixes: del case_prefixes[case_id]
            if case_id in case_start_times: del case_start_times[case_id]
            if case_id in case_schedule_times: del case_schedule_times[case_id]
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
