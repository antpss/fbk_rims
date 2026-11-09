import os
import json
import argparse
import pandas as pd
import numpy as np
import pm4py
from config_loader import load_config, PROJECT_ROOT
try:
    from features import FeatureManager
except ImportError:
    from src.features import FeatureManager
try:
    from simulator.calendar import CalendarManager
except ImportError:
    from src.simulator.calendar import CalendarManager
try:
    from activity_profiler import load_or_discover_activity_profiles
except ImportError:
    from src.activity_profiler import load_or_discover_activity_profiles

def build_dataset(cfg, task_name="duration"):
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

    from temporal_split import require_artifact
    require_artifact(cfg, log_path)
    print(f"Loading aligned log: {log_path}")
    log = pm4py.read_xes(log_path)
    df = pm4py.convert_to_dataframe(log)

    from temporal_split import training_rows, split_manifest
    fit_df = training_rows(df, cfg)
    manifest = split_manifest(cfg) if "model_strategy" in cfg else None

    activities = cfg.get("target_activities", [])
    if not activities:
        activities = sorted(fit_df['concept:name'].dropna().unique().tolist())
        print(f"Target activities: [AUTO-DISCOVERED {len(activities)} activities from log]: {activities}")
    else:
        print(f"Target activities ({len(activities)} configured): {activities}")

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
        series = fit_df[col] if col in fit_df.columns else (fit_df[clean_col] if clean_col in fit_df.columns else pd.Series(dtype=float))
        is_cont = (attr.get("type") == "continuous")
        treat_zero = attr.get("treat_zero_as_missing", False)
        if is_cont:
            clean_s = pd.to_numeric(series, errors="coerce")
            if treat_zero:
                clean_s = clean_s[clean_s != 0]
            clean_s = clean_s.dropna()
            strat = str(attr.get("fill_strategy", "median")).lower()
            if clean_s.empty:
                val_stat = float(attr.get("fill_value", 0.0))
            elif strat == "mean":
                val_stat = float(clean_s.mean())
            elif strat == "mode":
                val_stat = float(clean_s.mode().iloc[0])
            else:
                val_stat = float(clean_s.median())
        else:
            clean_s = series.dropna()
            val_stat = str(clean_s.mode().iloc[0]) if not clean_s.empty else str(attr.get("fill_value", "UNKNOWN"))
        attr_imputation[attr["feature_name"]] = val_stat
    if attr_imputation:
        print(f"Empirical Imputation Defaults for Case Attributes: {attr_imputation}")

    feature_mgr = FeatureManager(cfg)
    print(f"Prefix Window: {prefix_window}")
    print(f"Selected Feature Extractors ({len(feature_mgr.selected_features)}): {feature_mgr.selected_features}")
    print(f"Domain Case Attributes (Tier 2): {[c['feature_name'] for c in case_attrs]}")
    print(f"Domain Event Attributes (Tier 2): {event_attrs}")

    # Pre-calculate original complete distributions for sample weighting
    if 'lifecycle:transition' in df.columns:
        trans_col = fit_df['lifecycle:transition'].astype(str).str.upper()
        completes_df = fit_df[(fit_df['concept:name'].isin(activities)) & (trans_col == 'COMPLETE')]
        if 'is_synthetic' in completes_df.columns:
            completes_df = completes_df[completes_df['is_synthetic'].astype(str).str.lower() != 'true']
        if len(completes_df) == 0:
            completes_df = fit_df[fit_df['concept:name'].isin(activities)]
    else:
        completes_df = fit_df[fit_df['concept:name'].isin(activities)]

    orig_activity_counts = completes_df['concept:name'].value_counts().to_dict()
    total_orig_completes = max(1, len(completes_df))

    print("Sorting log chronologically (START before COMPLETE for equal timestamps)...")
    trans_order = {"START": 0, "COMPLETE": 1}
    df["_trans_rank"] = df["lifecycle:transition"].astype(str).str.upper().map(trans_order).fillna(2)
    df = df.sort_values(by=["time:timestamp", "_trans_rank"], kind="stable").drop(columns=["_trans_rank"]).reset_index(drop=True)

    models_dir_rel = cfg.get("paths", {}).get("models_dir", "models")
    role_file = os.path.join(PROJECT_ROOT, cfg.get("resources", {}).get("discovery", {}).get("output_file", os.path.join(models_dir_rel, "role_mapping.json")))
    require_artifact(cfg, role_file)
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

    calendar = CalendarManager.from_config(cfg, activity_to_role=activity_to_role, role_mapping_path=role_file)
    print(f"Calendar Active: {calendar.enabled} ({calendar.hour_start}:00 - {calendar.hour_end}:00, Days: {calendar.work_days})")
    if calendar.role_calendars:
        print(f"Role-specific calendars active: {list(calendar.role_calendars.keys())}")
    profiles_data = load_or_discover_activity_profiles(cfg)
    activity_types = profiles_data.get("activity_types", {})
    print(f"Loaded activity profile classifications for {len(activity_types)} activities.")

    # Determine tracked activities for trace_history
    th_cfg = feat_cfg.get("trace_history", {})
    th_mode = th_cfg.get("mode", "profile_aware")
    if th_mode == "whitelist":
        tracked_activities = th_cfg.get("whitelist", [])
    elif th_mode == "all_activities":
        tracked_activities = sorted(fit_df['concept:name'].dropna().unique().tolist())
    else:  # "profile_aware"
        tracked_activities = sorted([
            act for act, p_type in activity_types.items()
            if p_type in ("INSTANT_GATEWAY", "AUTOMATED_TASK")
        ])
    if feature_mgr.is_feature_active("trace_history"):
        print(f"Trace History ({th_mode}): tracking {len(tracked_activities)} activities: {tracked_activities}")

    df_inst = None
    if task_name == "waiting_time":
        if feature_mgr.is_feature_active("resource_id") or feature_mgr.is_feature_active("event_attributes"):
            raise ValueError("Waiting prediction cannot use the future worker or execution-event attributes; disable them for this task")
        out_base = cfg["paths"]["output_base_dir"]
        parquet = os.path.join(out_base, "causal_delays_instance_level.parquet")
        csv = os.path.join(out_base, "causal_delays_instance_level.csv.gz")
        if os.path.exists(parquet):
            require_artifact(cfg, parquet)
            df_inst = pd.read_parquet(parquet)
        elif os.path.exists(csv):
            require_artifact(cfg, csv)
            df_inst = pd.read_csv(csv)
        else:
            raise FileNotFoundError("Run Step 02b before waiting dataset creation: eligibility timestamps are required")
        if "predecessor_end_timestamp" not in df_inst:
            raise ValueError("Delay records lack eligibility timestamps; rerun Step 02b")
    from historical_features import snapshot_rows
    extraction_summary = {}
    all_rows = snapshot_rows(df, cfg, task_name, feature_mgr, calendar, role_data,
                             activity_types, activities, tracked_activities,
                             attr_imputation, df_inst, manifest, diagnostics=extraction_summary)
    if not all_rows:
        raise ValueError(f"No usable {task_name} training instances")

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
        fit_rows = task_df[task_df.Partition == 'train'] if manifest else task_df
        train_counts = fit_rows['Target_Activity'].value_counts().to_dict()
        total_train = len(fit_rows)

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

        task_df['Sample_Weight'] = task_df['Target_Activity'].map(weights_dict).fillna(1.0)
        mean_w = float(fit_rows['Target_Activity'].map(weights_dict).mean())
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
    extraction_summary["rows_after_target_filter"] = len(task_df)
    extraction_summary["partition_rows"] = task_df.Partition.value_counts().to_dict()
    with open(out_path + ".summary.json", "w") as f:
        json.dump(extraction_summary, f, indent=2)
    from temporal_split import stamp_artifact
    stamp_artifact(cfg, out_path)
    print(f"\nSaved [{task_name}] dataset to: {out_path} ({len(task_df)} rows)")
    print(f"Dataset creation for task '{task_name}' complete.\n")
    return task_df


def main():
    parser = argparse.ArgumentParser(description="Config-Driven Dataset Creation for RIMS++")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--task", type=str, default="duration", choices=["duration", "waiting_time"],
                        help="Task-specific dataset to generate ('duration' or 'waiting_time')")
    args = parser.parse_args()

    cfg = load_config(args.config)
    build_dataset(cfg, task_name=args.task)


if __name__ == "__main__":
    main()
