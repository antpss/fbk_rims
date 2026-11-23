"""Conventions for optional XES lifecycle and resource fields."""
from pathlib import Path
import warnings

import pandas as pd


def normalize_optional_columns(df):
    """Treat unspecified lifecycle as COMPLETE and unspecified resource as UNKNOWN.

    This does not create START events or infer human execution durations. Other
    lifecycle states (SCHEDULE, ASSIGN, etc.) retain their distinct meanings.
    """
    df = df.copy()
    if "lifecycle:transition" not in df:
        df["lifecycle:transition"] = "COMPLETE"
    else:
        df["lifecycle:transition"] = (df["lifecycle:transition"].fillna("COMPLETE")
                                      .astype(str).str.strip().str.upper().replace("", "COMPLETE"))
    if "org:resource" not in df:
        df["org:resource"] = "UNKNOWN"
    else:
        df["org:resource"] = df["org:resource"].fillna("UNKNOWN").replace("", "UNKNOWN")
    return df


def read_delay_records(cfg):
    """Prefer Parquet, but support CSV when no Parquet engine is installed."""
    from temporal_split import require_artifact
    base = Path(cfg["paths"]["output_base_dir"])
    parquet = base / "causal_delays_instance_level.parquet"
    csv = base / "causal_delays_instance_level.csv.gz"
    if parquet.exists():
        require_artifact(cfg, parquet)
        try:
            return pd.read_parquet(parquet)
        except ImportError:
            if not csv.exists():
                raise ImportError("Parquet support is unavailable and no CSV delay table exists; rerun Step 02b")
            warnings.warn("Parquet support is unavailable; using the CSV delay table.", UserWarning)
    if csv.exists():
        require_artifact(cfg, csv)
        return pd.read_csv(csv)
    raise FileNotFoundError("Run Step 02b before waiting dataset creation: eligibility timestamps are required")

