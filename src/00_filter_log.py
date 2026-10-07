import os
import warnings
import pm4py
import pandas as pd
from config_loader import load_config, PROJECT_ROOT

#suppress PM4Py warnings about r4pm backend
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# CONFIGURATION
cfg = load_config()
prep_cfg = cfg.get("preprocessing", {})

input_rel = prep_cfg.get("input_log", "data/raw/BPI_Challenge_2012.xes")
INPUT_LOG = os.path.join(PROJECT_ROOT, input_rel)

output_rel = cfg["paths"].get("raw_log", "data/processed/BPI_2012_W_only.xes")
OUTPUT_LOG = os.path.join(PROJECT_ROOT, output_rel)
os.makedirs(os.path.dirname(OUTPUT_LOG), exist_ok=True)

# Prefixes to keep from config (empty list = keep all)
process_prefixes = prep_cfg.get("process_activity_prefixes", prep_cfg.get("prefixes_to_keep", []))
milestone_prefixes = prep_cfg.get("milestone_prefixes", [])
PREFIXES_TO_KEEP = list(dict.fromkeys(process_prefixes + milestone_prefixes))

# Translation / renaming map from config (empty dict = no translation)
TRANSLATION_MAP = prep_cfg.get("activity_mapping", {})
# ==========================================

def filter_log():
    print(f"Loading log: {INPUT_LOG}")
    if not os.path.exists(INPUT_LOG):
        raise FileNotFoundError(f"Input event log not found at: {INPUT_LOG}")

    # pm4py reads the XES and returns a Pandas DataFrame
    df = pm4py.read_xes(INPUT_LOG)
    print(f"Original events: {len(df)}")

    # Filtering by prefix if specified
    if PREFIXES_TO_KEEP:
        prefixes_tuple = tuple(PREFIXES_TO_KEEP)
        print(f"Filtering to keep activities starting with: {prefixes_tuple}")
        if process_prefixes:
            print(f"  - Active Process Prefixes:    {process_prefixes}")
        if milestone_prefixes:
            print(f"  - Milestone Routing Prefixes: {milestone_prefixes}")
        df_filtered = df[df['concept:name'].str.startswith(prefixes_tuple)].copy()
    else:
        print("No prefix filtering specified; keeping all activities.")
        df_filtered = df.copy()
    
    # Remove empty traces (cases that have 0 events after filtering)
    case_counts = df_filtered.groupby('case:concept:name').size()
    valid_cases = case_counts[case_counts > 0].index
    df_filtered = df_filtered[df_filtered['case:concept:name'].isin(valid_cases)].copy()

    # Apply activity renaming/translation if configured
    if TRANSLATION_MAP:
        print(f"Applying activity mapping ({len(TRANSLATION_MAP)} replacements)...")
        df_filtered['concept:name'] = df_filtered['concept:name'].replace(TRANSLATION_MAP)

    print(f"Filtered process events: {len(df_filtered)}")
    print(f"Exporting process log to: {OUTPUT_LOG}")
    pm4py.write_xes(df_filtered, OUTPUT_LOG)

    # Optional: If separate routing lifecycle log is configured (e.g. for dual-stream logs like BPI 2012)
    routing_raw_rel = cfg.get("paths", {}).get("routing_raw_log")
    if routing_raw_rel and milestone_prefixes:
        routing_out_log = os.path.join(PROJECT_ROOT, routing_raw_rel)
        if routing_out_log != OUTPUT_LOG:
            os.makedirs(os.path.dirname(routing_out_log), exist_ok=True)
            df_routing = df[df['concept:name'].str.startswith(tuple(milestone_prefixes))].copy()
            r_counts = df_routing.groupby('case:concept:name').size()
            df_routing = df_routing[df_routing['case:concept:name'].isin(r_counts[r_counts > 0].index)].copy()
            print(f"Exporting routing lifecycle log ({len(df_routing)} events) to: {routing_out_log}")
            pm4py.write_xes(df_routing, routing_out_log)

    print("Done.")

if __name__ == "__main__":
    filter_log()
