import os
import json
import argparse
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

    activities = cfg.get("target_activities", [])
    print(f"Target activities ({len(activities)}): {activities}")

    # Feature extraction settings
    feat_cfg = cfg.get("features", {})
    prefix_window = feat_cfg.get("prefix_window_size", 10
    univ_cfg = feat_cfg.get("universal", {})
    case_attrs = feat_cfg.get("case_attributes", [])
    event_attrs = feat_cfg.get("event_attributes", [])

    print(f"Prefix Window: {prefix_window}")
    print(f"Universal Features: WIP={univ_cfg.get('use_wip')}, Roles={univ_cfg.get('use_role_occupancy')}, Calendar={univ_cfg.get('use_calendar_time')}")
    print(f"Domain Case Attributes (Tier 2): {[c['feature_name'] for c in case_attrs] if case_attrs else 'None (Strictly Agnostic)'}")
    print(f"Domain Event Attributes (Tier 2): {event_attrs if event_attrs else 'None (Resource-Agnostic)'}")

    # Pre-calculate original complete distributions for sample weighting
    completes_df = df[(df['concept:name'].isin(activities)) & (df['lifecycle:transition'].str.upper() == 'COMPLETE')]
    orig_activity_counts = completes_df['concept:name'].value_counts().to_dict()
    total_orig_completes = len(completes_df)

    print("Sorting log chronologically...")
    df = df.sort_values(by="time:timestamp").reset_index(drop=True)

    active_cases = set()
    role_file = os.path.join(PROJECT_ROOT, "models/role_mapping.json")
    if os.path.exists(role_file):
        with open(role_file, 'r') as f:
            role_data = json.load(f)
            activity_capacity = {act: len(role_data.get(act, [1])) for act in activities}
    else:
        activity_capacity = {}
    activity_capacity = {k: max(1, v) for k, v in activity_capacity.items()}

    ac_wip = {act: 0 for act in activities}
    case_event_counts = df['case:concept:name'].value_counts().to_dict()
    seen_events = {case_id: 0 for case_id in case_event_counts}

    case_prefixes = {}
    case_start_times = {}
    case_start_features = {}
    case_prev_time = {}
    case_last_complete = {}
    case_prev_end_time = {}

    all_rows = []
    print("Sweeping log chronologically to construct state vectors...")

    for _, row in df.iterrows():
        case_id = row["case:concept:name"]
        activity_name = row["concept:name"]
        transition = str(row.get("lifecycle:transition", "COMPLETE")).upper()
        timestamp = row["time:timestamp"]

        # Dynamic extraction of Tier 2 case attributes
        extracted_case_attrs = {}
        for attr in case_attrs:
            col = attr["column"]
            val = row.get(col, attr.get("fill_value", 0.0))
            try:
                val = float(val) if attr.get("type") == "continuous" else val
            except:
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
        if case_id not in case_start_times:
            case_start_times[case_id] = {}
        if case_id not in case_start_features:
            case_start_features[case_id] = {}

        if transition == "START":
            ac_wip[activity_name] += 1
            case_start_times[case_id][activity_name] = timestamp

            if case_id in case_prev_end_time:
                current_wait = (timestamp - case_prev_end_time[case_id]).total_seconds()
            else:
                current_wait = 0.0

            case_start_features[case_id][activity_name] = {
                "Weekday": weekday,
                "Daytime": daytime,
                "WIP": wip,
                "AC_WIP": ac_wip[activity_name],
                "RP_OC_Array": rp_oc_array,
                "Prev_Proc_Time": case_prev_time[case_id],
                "Wait_Time_Seconds": max(0.0, current_wait),
                **extracted_case_attrs,
                **extracted_event_attrs
            }

        elif transition == "COMPLETE":
            if ac_wip[activity_name] > 0:
                ac_wip[activity_name] -= 1

            if activity_name in case_start_times[case_id]:
                duration = (timestamp - case_start_times[case_id][activity_name]).total_seconds()
                del case_start_times[case_id][activity_name]
            else:
                duration = 0.0

            case_prev_time[case_id] = duration
            case_last_complete[case_id] = timestamp
            case_prev_end_time[case_id] = timestamp

            if activity_name in case_start_features[case_id]:
                feat = case_start_features[case_id].pop(activity_name)
            else:
                feat = {
                    "Weekday": weekday,
                    "Daytime": daytime,
                    "WIP": wip,
                    "AC_WIP": 0,
                    "RP_OC_Array": rp_oc_array,
                    "Prev_Proc_Time": 0.0,
                    "Wait_Time_Seconds": 0.0,
                    **extracted_case_attrs,
                    **extracted_event_attrs
                }

            row_dict = {
                "Target_Activity": activity_name,
                "Prefix": ",".join(case_prefixes[case_id][-prefix_window:]),
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
            del case_prefixes[case_id]
            del case_start_times[case_id]
            del case_prev_time[case_id]
            if case_id in case_last_complete: del case_last_complete[case_id]
            if case_id in case_prev_end_time: del case_prev_end_time[case_id]
            if case_id in case_start_features: del case_start_features[case_id]

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

