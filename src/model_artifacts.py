"""Small artifact contracts for reproducible candidate selection and deployment."""
import hashlib
import json
from pathlib import Path
import numpy as np
import warnings


def learning_settings(cfg):
    """Record modeling settings, excluding run limits and output locations."""
    def without_paths(value):
        if isinstance(value, dict):
            return {k: without_paths(v) for k, v in value.items()
                    if k not in {"output_file", "report_file", "output_dir", "input_log"}}
        if isinstance(value, list):
            return [without_paths(v) for v in value]
        return value
    snapshot = {key: without_paths(cfg.get(key, {})) for key in
                ("preprocessing", "discovery", "activity_profiling", "resources", "features", "tasks", "model_strategy")}
    snapshot["target_activities"] = sorted(cfg.get("target_activities", []))
    sim = cfg.get("simulation", {})
    snapshot["calendar"] = sim.get("calendar", {})
    snapshot["delay_discovery"] = {key: sim.get("extraneous_delays", {}).get(key)
                                    for key in ("estimator", "placement")}
    return snapshot


def warn_learning_settings(cfg, metadata, artifact):
    """Configuration drift is advisory; existing content/schema checks stay strict."""
    saved = metadata.get("learning_settings")
    current = learning_settings(cfg)
    if saved is None:
        warnings.warn(f"Cannot check modeling settings for {artifact}: no saved settings. "
                      "Rebuild to record them; continuing.", UserWarning, stacklevel=2)
    elif saved != current:
        changed = sorted(k for k in set(saved) | set(current)
                         if saved.get(k) != current.get(k))
        warnings.warn(f"Modeling settings changed for {artifact}: {', '.join(changed)}. "
                      "Rebuild the affected discovery, datasets and models; continuing.",
                      UserWarning, stacklevel=2)


def experiment_id(cfg, dataset):
    from temporal_split import split_manifest
    identity = {"dataset_sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
                "split_id": split_manifest(cfg)["split_id"],
                "features": cfg.get("features", {}), "model_strategy": cfg.get("model_strategy", {}),
                "schema_version": 2}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def prediction_metrics(predictions):
    valid = predictions[np.isfinite(predictions.Predicted) & predictions.Source_Model.ne("")]
    def scores(frame):
        actual, pred = frame.Actual.to_numpy(), frame.Predicted.to_numpy()
        return {"count": len(frame), "mae": float(np.mean(np.abs(actual-pred))),
                "smape": float(np.mean(np.abs(actual-pred)/((np.abs(actual)+np.abs(pred))/2+1e-8))*100)}
    per_act = {str(act): {**scores(sub), "model_source": sub.Source_Model.iloc[0]} for act, sub in valid.groupby("Target_Activity")}
    overall = scores(valid) if len(valid) else {"mae": None, "smape": None}
    return {"weighted_mae": overall["mae"], "weighted_smape": overall["smape"],
            "macro_mae": float(np.mean([s["mae"] for s in per_act.values()])) if per_act else None,
            "activity_metrics": per_act, "available_count": len(valid), "total_count": len(predictions)}
