"""One case split shared by discovery, feature fitting and predictor evaluation."""
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

METADATA_COLUMNS = {"Case_Arrival", "Case_End", "Partition", "Instance_ID", "Eligibility_Time"}


def make_case_split(df, strategy="temporal", seed=42):
    events = df.assign(**{"case:concept:name": df["case:concept:name"].astype(str)})
    bounds = events.groupby("case:concept:name")["time:timestamp"].agg(["min", "max"])
    bounds["min"] = pd.to_datetime(bounds["min"], utc=True)
    bounds["max"] = pd.to_datetime(bounds["max"], utc=True)
    bounds = bounds.sort_values(["min"], kind="stable")
    if len(bounds) < 3:
        raise ValueError("At least three cases are required for train/validation/test splitting")
    n = len(bounds)
    if strategy == "temporal":
        cut1 = bounds.iloc[min(n-2, max(1, int(n*.7)))]["min"]
        cut2 = bounds.iloc[min(n-1, max(2, int(n*.8)))]["min"]
        labels = pd.Series("excluded", index=bounds.index)
        labels[bounds["max"] < cut1] = "train"
        labels[(bounds["min"] >= cut1) & (bounds["max"] < cut2)] = "validation"
        labels[bounds["min"] >= cut2] = "test"
        cutoffs = [cut1.isoformat(), cut2.isoformat()]
    elif strategy == "random":
        ids = bounds.sample(frac=1, random_state=seed).index
        labels = pd.Series("test", index=bounds.index)
        labels.loc[ids[:max(1, int(n*.7))]] = "train"
        labels.loc[ids[max(1, int(n*.7)):max(2, int(n*.8))]] = "validation"
        cutoffs = []
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
    cases = {str(cid): {"arrival": row["min"].isoformat(), "end": row["max"].isoformat(),
                       "partition": labels.loc[cid]} for cid, row in bounds.iterrows()}
    return {"schema_version": 1, "strategy": strategy, "seed": seed, "cutoffs": cutoffs, "counts": labels.value_counts().to_dict(), "cases": cases}


def split_manifest(cfg):
    """Cache by source contents and split policy, never by aligned row ordering."""
    if "_split_manifest" in cfg:
        return cfg["_split_manifest"]
    import pm4py
    raw = cfg["paths"].get("raw_log", cfg["paths"]["aligned_log"])
    digest = hashlib.sha256(Path(raw).read_bytes()).hexdigest()
    strategy = cfg.get("model_strategy", {}).get("split_strategy", "temporal")
    seed = cfg.get("project", {}).get("random_seed", 42)
    path = Path(cfg["paths"]["output_base_dir"]) / "case_split.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    if (manifest.get("source_sha256"), manifest.get("strategy"), manifest.get("seed")) != (digest, strategy, seed):
        manifest = make_case_split(pm4py.convert_to_dataframe(pm4py.read_xes(raw)), strategy, seed)
        manifest["source_sha256"] = digest
        manifest["split_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2))
        os.replace(temporary, path)
    cfg["_split_manifest"] = manifest
    return manifest


def training_rows(df, cfg, case_column="case:concept:name"):
    # Hand-built legacy test configs without an evaluation policy retain their fixture.
    if "model_strategy" not in cfg:
        return df.copy()
    manifest = split_manifest(cfg)
    ids = {cid for cid, info in manifest["cases"].items() if info["partition"] == "train"}
    return df[df[case_column].astype(str).isin(ids)].copy()


def stamp_artifact(cfg, path):
    """Mark learned/repaired artifacts so pre-fix full-log artifacts cannot be reused."""
    if "model_strategy" not in cfg:
        return
    path = Path(path)
    metadata = {"split_id": split_manifest(cfg)["split_id"],
                "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    Path(str(path) + ".provenance.json").write_text(json.dumps(metadata, indent=2))


def require_artifact(cfg, path):
    if "model_strategy" not in cfg:
        return
    path = Path(path)
    meta_path = Path(str(path) + ".provenance.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if (not path.exists() or meta.get("split_id") != split_manifest(cfg)["split_id"] or
            meta.get("artifact_sha256") != hashlib.sha256(path.read_bytes()).hexdigest()):
        raise ValueError(f"Missing/stale training-only provenance for {path}; rerun its producing pipeline stage")
