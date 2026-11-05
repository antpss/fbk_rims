import os
import argparse
import warnings
import pm4py
import pandas as pd
from config_loader import load_config, PROJECT_ROOT

#suppress PM4Py warnings about r4pm backend
warnings.filterwarnings("ignore", category=UserWarning)


def filter_log(config_path=None):
    cfg = load_config(config_path)
    prep_cfg = cfg.get("preprocessing", {})

    input_rel = prep_cfg.get("input_log", "data/raw/BPI_Challenge_2012.xes")
    input_log = os.path.join(PROJECT_ROOT, input_rel)

    output_rel = cfg["paths"].get("raw_log", "data/processed/BPI_2012_filtered.xes")
    output_log = os.path.join(PROJECT_ROOT, output_rel)
    os.makedirs(os.path.dirname(output_log), exist_ok=True)

    # Prefixes / activities to keep from config (empty list = keep all)
    prefixes_to_keep = list(dict.fromkeys(prep_cfg.get("prefixes_to_keep", [])))

    # Translation / renaming map from config (empty dict = no translation)
    translation_map = prep_cfg.get("activity_mapping", {})

    print(f"Loading log: {input_log}")
    if not os.path.exists(input_log):
        raise FileNotFoundError(f"Input event log not found at: {input_log}")

    if input_log.lower().endswith(".csv"):
        df = pd.read_csv(input_log)
    else:
        df = pm4py.read_xes(input_log)
    print(f"Original events: {len(df)}")

    # Standardize column names if configured
    col_cfg = prep_cfg.get("columns", {})
    rename_map = {}
    for std_col, cfg_col in [
        ("case:concept:name", col_cfg.get("case_id")),
        ("concept:name", col_cfg.get("activity")),
        ("time:timestamp", col_cfg.get("timestamp")),
        ("org:resource", col_cfg.get("resource")),
        ("lifecycle:transition", col_cfg.get("lifecycle")),
    ]:
        if cfg_col and cfg_col in df.columns and cfg_col != std_col:
            rename_map[cfg_col] = std_col
    if rename_map:
        print(f"Applying custom column mapping: {rename_map}")
        df = df.rename(columns=rename_map)

    # Ensure timestamp column is parsed to UTC datetime
    if "time:timestamp" in df.columns:
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)

    # Standardize lifecycle transition values to standard START / COMPLETE
    lc_cfg = prep_cfg.get("lifecycle_mapping", {})
    if lc_cfg and "lifecycle:transition" in df.columns:
        start_aliases = [s.upper() for s in lc_cfg.get("start", ["START", "start", "begin", "open"])]
        comp_aliases = [c.upper() for c in lc_cfg.get("complete", ["COMPLETE", "complete", "finish", "end", "close"])]
        trans_upper = df["lifecycle:transition"].astype(str).str.strip().str.upper()
        df.loc[trans_upper.isin(start_aliases), "lifecycle:transition"] = "START"
        df.loc[trans_upper.isin(comp_aliases), "lifecycle:transition"] = "COMPLETE"

    # Filtering by prefix if specified
    if prefixes_to_keep:
        prefixes_tuple = tuple(prefixes_to_keep)
        print(f"Filtering to keep activities starting with: {prefixes_tuple}")
        df_filtered = df[df['concept:name'].str.startswith(prefixes_tuple)].copy()
    else:
        print("No prefix filtering specified; keeping all activities.")
        df_filtered = df.copy()

    # Remove empty traces (cases that have 0 events after filtering)
    case_counts = df_filtered.groupby('case:concept:name').size()
    valid_cases = case_counts[case_counts > 0].index
    df_filtered = df_filtered[df_filtered['case:concept:name'].isin(valid_cases)].copy()

    # Apply activity renaming/translation if configured
    if translation_map:
        print(f"Applying activity mapping ({len(translation_map)} replacements)...")
        df_filtered['concept:name'] = df_filtered['concept:name'].replace(translation_map)

    print(f"Filtered process events: {len(df_filtered)}")
    print(f"Exporting process log to: {output_log}")
    pm4py.write_xes(df_filtered, output_log)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Filter and standardize input event log for RIMS++.")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()
    filter_log(config_path=args.config)


if __name__ == "__main__":
    main()
