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
PREFIXES_TO_KEEP = prep_cfg.get("prefixes_to_keep", [])
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
        print(f"Filtering to keep only activities starting with: {prefixes_tuple}")
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

    print(f"Filtered events: {len(df_filtered)}")

    print(f"Exporting to {OUTPUT_LOG}")
    # Exporting the Pandas df back to XES format
    pm4py.write_xes(df_filtered, OUTPUT_LOG)
    print("Done.")

if __name__ == "__main__":
    filter_log()
