"""Train configured candidates, select on validation, package and evaluate frozen dispatch."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

import numpy as np
import pandas as pd
from config_loader import load_config, PROJECT_ROOT
from model_artifacts import prediction_metrics, experiment_id


def discover_trained_metrics(models_dir, task="duration", candidates=None, identity=None):
    results = []
    root = Path(models_dir)
    for folder in sorted(root.iterdir()) if root.exists() else []:
        if candidates is not None and folder.name not in candidates:
            continue
        path = folder / "metrics.json"
        if folder.name.endswith("champion") or not path.exists():
            continue
        data = json.loads(path.read_text())
        if data.get("task") != task or data.get("selection_partition") != "validation":
            continue
        if identity is not None and data.get("experiment_id") != identity:
            continue
        data["folder_name"] = folder.name
        results.append(data)
    return results


def valid_entry(root, candidate, act):
    path = root / candidate["folder_name"] / "dispatch_config.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    if manifest.get("experiment_id") != candidate.get("experiment_id"):
        return None
    entry = manifest["dispatch"].get(act, manifest["dispatch"].get("__default__"))
    if not isinstance(entry, dict) or not entry.get("file"):
        return None
    if not (path.parent / entry["file"]).is_file():
        return None
    return dict(entry)


def synthesize_champion(models_dir, task="duration", mode="hybrid", candidates=None, identity=None):
    if mode == "single_global":
        return None
    root = Path(models_dir)
    choices = discover_trained_metrics(root, task, candidates, identity)
    identities = {c.get("experiment_id") for c in choices}
    if not choices or len(identities) != 1 or None in identities:
        raise ValueError("No compatible current-experiment candidates; rerun training")
    globals_ = [c for c in choices if c.get("strategy") == "global"
                and c.get("weighted_mae") is not None
                and c.get("available_count") == c.get("total_count")
                and valid_entry(root, c, "__default__")]
    if not globals_:
        raise ValueError("A fully evaluated global fallback is required")
    fallback = min(globals_, key=lambda c: (c["weighted_mae"], c["folder_name"]))
    assignments = {"__default__": fallback}
    acts = set().union(*(c.get("activity_metrics", {}) for c in choices))
    for act in sorted(acts):
        eligible = [c for c in choices if act in c.get("activity_metrics", {})
                    and np.isfinite(c["activity_metrics"][act]["mae"]) and valid_entry(root, c, act)]
        assignments[act] = (min(eligible, key=lambda c: (c["activity_metrics"][act]["mae"], c["folder_name"]))
                            if mode == "hybrid" and eligible else fallback)
    name = "hybrid_champion" if mode == "hybrid" else "global_champion"
    manifest = {"schema_version": 2, "task": task, "strategy": name,
                "experiment_id": next(iter(identities)), "dispatch": {}}
    # Decide and freeze routing before consulting held-out predictions.
    for act, candidate in assignments.items():
        manifest["dispatch"][act] = {**valid_entry(root, candidate, act), "bundle": candidate["folder_name"]}
    selected = {c["folder_name"] for c in assignments.values()}
    stage = Path(tempfile.mkdtemp(prefix=".champion-", dir=root))
    try:
        for folder in selected:
            source = root / folder
            child = json.loads((source / "dispatch_config.json").read_text())
            files = {entry["file"] for entry in child["dispatch"].values() if entry.get("file")}
            files.update({"dispatch_config.json", "training_metadata.json"})
            files.update(path.name for path in source.glob("scaler_*.pkl"))
            files.update(name for name in ("vocab.json", "categories.json") if (source / name).exists())
            (stage / folder).mkdir()
            for filename in files:
                shutil.copyfile(source / filename, stage / folder / filename)
        shutil.copyfile(root / fallback["folder_name"] / "training_metadata.json", stage / "training_metadata.json")
        (stage / "dispatch_config.json").write_text(json.dumps(manifest, indent=2))
        # Validation and test rows must have identical identities/targets across candidates.
        tables = {folder: pd.read_csv(root / folder / "predictions.csv", dtype={"Case_ID": str, "Instance_ID": str}) for folder in selected}
        base = tables[fallback["folder_name"]].copy()
        keys = ["Case_ID", "Instance_ID", "Partition", "Target_Activity"]
        for table in tables.values():
            if table.duplicated(keys).any():
                raise ValueError("Duplicate evaluation instance identity")
            check = base[keys + ["Actual"]].merge(table[keys + ["Actual"]], on=keys, how="outer", suffixes=("_base", "_candidate"))
            if len(check) != len(base) or not np.array_equal(check.Actual_base, check.Actual_candidate):
                raise ValueError("Candidate evaluation populations differ")
        for act in base.Target_Activity.unique():
            folder = assignments.get(act, fallback)["folder_name"]
            mask = base.Target_Activity == act
            rows = base.loc[mask, keys].merge(tables[folder], on=keys, validate="one_to_one", how="left")
            base.loc[mask, "Predicted"] = rows.Predicted.to_numpy()
            base.loc[mask, "Source_Model"] = folder
        if not np.isfinite(base.Predicted).all():
            raise ValueError("Frozen champion has unavailable predictions")
        result = {"task": task, "strategy": mode, "model_type": "champion", "folder_name": name,
                  "experiment_id": manifest["experiment_id"], "selection_partition": "validation",
                  **prediction_metrics(base[base.Partition == "validation"]),
                  "test_metrics": prediction_metrics(base[base.Partition == "test"])}
        (stage / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        base.to_csv(stage / "predictions.csv", index=False)
        # Loading the staged suite checks architecture, weights, vocab and scaler together.
        from simulator.predictor import ChampionPredictor
        loader = ChampionPredictor.__new__(ChampionPredictor)
        loader._load_model_suite(str(stage), manifest["dispatch"])
        destination = root / name
        backup = root / ("." + name + "-" + uuid.uuid4().hex)
        if destination.exists():
            destination.rename(backup)
        try:
            stage.rename(destination)
        except BaseException:
            if backup.exists():
                backup.rename(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def synthesize_champion_hybrid(models_dir, task="duration"):
    return synthesize_champion(models_dir, task, "hybrid")


def synthesize_champion_global(models_dir, task="duration"):
    return synthesize_champion(models_dir, task, "global_tournament")


def run_training_experiment(strategy, model_type, task="duration", config_path=None):
    cmd = [sys.executable, str(Path(PROJECT_ROOT)/"src/04_train.py"), "--task", task,
           "--strategy", strategy, "--model_type", model_type]
    if config_path:
        cmd += ["--config", config_path]
    subprocess.run(cmd, check=True)


def print_comparison_table(metrics_list, models_dir, task="duration"):
    def fmt(v):
        return "N/A" if v is None else f"{v:.3f}"
    lines = [f"# {task} candidate validation results", "",
             "Selection uses validation MAE. Partial-coverage candidates are eligible only for activities they actually predict.", "",
             "| Candidate | Validation MAE (s) | Coverage | Frozen champion test MAE (s) |",
             "|---|---:|---:|---:|"]
    for m in metrics_list:
        lines.append(f"| {m['folder_name']} | {fmt(m.get('weighted_mae'))} | {m.get('available_count')}/{m.get('total_count')} | {fmt(m.get('test_metrics', {}).get('weighted_mae'))} |")
    text = "\n".join(lines) + "\n"
    print(text)
    (Path(models_dir)/"benchmark_report.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--task", choices=["duration", "waiting_time"], default="duration")
    parser.add_argument("--mode", choices=["hybrid", "global_tournament", "single_global"])
    parser.add_argument("--compare", "--report", action="store_true")
    parser.add_argument("--run_all", "--tournament", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    settings = cfg.get("model_strategy", {})
    mode = args.mode or settings.get("mode", "hybrid")
    candidates = ([{"strategy": "global", "engine": settings.get("single_global_engine", "xgboost")}]
                  if mode == "single_global" else settings.get("candidates", {}).get(mode, []))
    if not candidates:
        raise ValueError(f"Configure model_strategy.candidates.{mode}")
    root = Path(cfg["paths"]["models_dir"]) / args.task
    root.mkdir(parents=True, exist_ok=True)
    if args.run_all:
        for c in candidates:
            run_training_experiment(c["strategy"], c["engine"], args.task, args.config)
    names = [c["strategy"] + "_" + c["engine"] for c in candidates]
    identity = experiment_id(cfg, Path(cfg["paths"]["output_base_dir"]) / cfg["tasks"][args.task]["output_file"])
    champion = synthesize_champion(root, args.task, mode, names, identity)
    results = discover_trained_metrics(root, args.task, names, identity)
    if mode == "single_global":
        if len(results) != 1:
            raise ValueError("Requested current-experiment single model is unavailable")
        table = pd.read_csv(root / names[0] / "predictions.csv")
        results[0]["test_metrics"] = prediction_metrics(table[table.Partition == "test"])
    print_comparison_table(results + ([champion] if champion else []), root, args.task)


if __name__ == "__main__":
    main()
