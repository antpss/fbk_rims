import pm4py
import pandas as pd
import warnings

#suppress PM4Py warnings about r4pm backend
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# CONFIGURATION
INPUT_LOG = "../data/raw/BPI_Challenge_2012.xes"
OUTPUT_LOG = "../data/processed/BPI_2012_W_only.xes"

# Choose prefixes to keep different event types
# For example: ("W_", "A_") or ("A_",)
PREFIXES_TO_KEEP = ("W_",) 
# ==========================================

def filter_log():
    print(f"Loading log: {INPUT_LOG}")
    #pm4py reads the XES and returns a Pandas DataFrame
    df = pm4py.read_xes(INPUT_LOG)
    print(f"Original events: {len(df)}")

    #filtering using Pandas (faster)
    print(f"Filtering to keep only activities starting with: {PREFIXES_TO_KEEP}")
    df_filtered = df[df['concept:name'].str.startswith(PREFIXES_TO_KEEP)]
    
    #remove empty traces (cases that have 0 events after filtering)
    #group by case ID, count events, and filter
    case_counts = df_filtered.groupby('case:concept:name').size()
    valid_cases = case_counts[case_counts > 0].index
    df_filtered = df_filtered[df_filtered['case:concept:name'].isin(valid_cases)]

    print(f"Filtered events: {len(df_filtered)}")

    print(f"Exporting to {OUTPUT_LOG}")
    #exporting the Pandas df back to XES format
    pm4py.write_xes(df_filtered, OUTPUT_LOG)
    print("Done.")

if __name__ == "__main__":
    filter_log()
