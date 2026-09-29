import os
import sys
import json
import argparse
import subprocess
import shutil
import pandas as pd
import numpy as np

from config_loader import load_config, PROJECT_ROOT

def discover_trained_metrics(models_dir, task="duration"):
    """
    Scans the models directory for any subdirectories containing metrics.json.
    Filters by task if specified.
    """
    results = []
    if not os.path.exists(models_dir):
        return results

    for entry in sorted(os.listdir(models_dir)):
        if entry.startswith(".") or entry.startswith("_") or entry in ["petri_nets", "petrinets"]:
            continue
        sub_dir = os.path.join(models_dir, entry)
        metrics_file = os.path.join(sub_dir, "metrics.json")
        if os.path.isdir(sub_dir) and os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r") as f:
                    data = json.load(f)
                    if task is None or data.get("task") == task:
                        data["folder_name"] = entry
                        results.append(data)
            except Exception as e:
                print(f"[Warning] Could not read {metrics_file}: {e}")
    return results

def synthesize_champion_hybrid(models_dir, task="duration"):
    """
    Analyzes all candidate models, identifies the best performer for each activity,
    and synthesizes an optimal 'hybrid_champion' dispatch manifest and metrics summary.
    """
    candidates = discover_trained_metrics(models_dir, task=task)
    # Exclude any existing champions so we do not self-reference
    candidates = [c for c in candidates if c.get("folder_name") not in ["hybrid_champion", "global_champion"]]
    if len(candidates) < 2:
        return None

    # Collect all activities evaluated across models
    all_acts = set()
    for c in candidates:
        all_acts.update(c.get("activity_metrics", {}).keys())
    all_acts = sorted(list(all_acts))

    if not all_acts:
        return None

    champion_dir = os.path.join(models_dir, "hybrid_champion")
    os.makedirs(champion_dir, exist_ok=True)

    champion_dispatch = {
        "strategy": "hybrid_champion",
        "task": task,
        "description": "Synthesized Best-of-All-Worlds Hybrid (dynamically selects champion model per activity)",
        "source_candidates": [c.get("folder_name") for c in candidates],
        "dispatch": {}
    }

    champion_activity_metrics = {}
    total_samples = 0
    sum_mae_weighted = 0.0
    sum_smape_weighted = 0.0

    # Determine global fallback (__default__)
    global_candidates = [c for c in candidates if "global" in c.get("folder_name", "").lower()]
    default_cand = min(global_candidates, key=lambda x: x.get("weighted_mae", float("inf"))) if global_candidates else min(candidates, key=lambda x: x.get("weighted_mae", float("inf")))

    default_dispatch_file = os.path.join(models_dir, default_cand["folder_name"], "dispatch_config.json")
    if os.path.exists(default_dispatch_file):
        try:
            with open(default_dispatch_file) as f:
                d_table = json.load(f)
                def_entry = d_table.get("dispatch", {}).get("__default__", {})
                if isinstance(def_entry, dict):
                    champion_dispatch["dispatch"]["__default__"] = {
                        "model_type": def_entry.get("model_type", default_cand.get("model_type")),
                        "source_dir": default_cand["folder_name"],
                        "file": def_entry.get("file")
                    }
                else:
                    champion_dispatch["dispatch"]["__default__"] = {
                        "model_type": default_cand.get("model_type"),
                        "source_dir": default_cand["folder_name"],
                        "file": str(def_entry)
                    }
        except Exception:
            pass

    # For each activity, pick the champion
    for act in all_acts:
        best_cand = None
        best_mae = float("inf")
        best_act_data = None

        for c in candidates:
            act_data = c.get("activity_metrics", {}).get(act)
            if act_data and act_data.get("mae") is not None:
                if act_data["mae"] < best_mae:
                    best_mae = act_data["mae"]
                    best_cand = c
                    best_act_data = act_data

        if best_cand and best_act_data:
            cnt = best_act_data.get("count", 0)
            mae = best_act_data.get("mae", 0.0)
            smape = best_act_data.get("smape", 0.0)
            src_folder = best_cand["folder_name"]

            # Read artifact filename from best_cand's dispatch_config.json
            cand_dispatch_file = os.path.join(models_dir, src_folder, "dispatch_config.json")
            model_file = None
            model_type = best_cand.get("model_type", "unknown")
            if os.path.exists(cand_dispatch_file):
                try:
                    with open(cand_dispatch_file) as f:
                        c_table = json.load(f)
                        entry = c_table.get("dispatch", {}).get(act)
                        if entry is None:
                            entry = c_table.get("dispatch", {}).get("__default__")
                        if isinstance(entry, dict):
                            model_file = entry.get("file")
                            model_type = entry.get("model_type", model_type)
                        elif isinstance(entry, str):
                            model_file = entry
                except Exception:
                    pass

            champion_dispatch["dispatch"][act] = {
                "model_type": model_type,
                "source_dir": src_folder,
                "file": model_file,
                "best_test_mae": round(mae, 2),
                "best_test_smape": round(smape, 2)
            }

            champion_activity_metrics[act] = {
                "count": cnt,
                "model_source": f"{src_folder} ({model_type.upper()})",
                "mae": round(mae, 2),
                "smape": round(smape, 2)
            }

            total_samples += cnt
            sum_mae_weighted += mae * cnt
            sum_smape_weighted += smape * cnt

    weighted_mae = (sum_mae_weighted / total_samples) if total_samples > 0 else 0.0
    weighted_smape = (sum_smape_weighted / total_samples) if total_samples > 0 else 0.0
    macro_mae = np.mean([v["mae"] for v in champion_activity_metrics.values()]) if champion_activity_metrics else 0.0

    # Copy auxiliary assets (vocab.json, scalers) into champion_dir for standalone deployment
    for act, info in champion_dispatch["dispatch"].items():
        s_folder = info.get("source_dir")
        if not s_folder:
            continue
        src_dir = os.path.join(models_dir, s_folder)
        if not os.path.exists(src_dir):
            continue
        v_src = os.path.join(src_dir, "vocab.json")
        if os.path.exists(v_src) and not os.path.exists(os.path.join(champion_dir, "vocab.json")):
            shutil.copy(v_src, os.path.join(champion_dir, "vocab.json"))
        for fname in os.listdir(src_dir):
            if fname.startswith("scaler_") and fname.endswith(".pkl"):
                dest = os.path.join(champion_dir, fname)
                if not os.path.exists(dest):
                    shutil.copy(os.path.join(src_dir, fname), dest)

    # Save champion dispatch_config.json
    with open(os.path.join(champion_dir, "dispatch_config.json"), "w") as f:
        json.dump(champion_dispatch, f, indent=2)

    # Save champion metrics.json
    champion_metrics = {
        "task": task,
        "strategy": "hybrid",
        "model_type": "champion",
        "folder_name": "hybrid_champion",
        "training_time_seconds": round(sum(c.get("training_time_seconds", 0.0) for c in candidates), 2),
        "weighted_mae": round(weighted_mae, 2),
        "weighted_smape": round(weighted_smape, 2),
        "macro_mae": round(macro_mae, 2),
        "activity_metrics": champion_activity_metrics
    }
    with open(os.path.join(champion_dir, "metrics.json"), "w") as f:
        json.dump(champion_metrics, f, indent=2)

    return champion_metrics

def synthesize_champion_global(models_dir, task="duration"):
    """
    Analyzes global candidate models, crowns the single best performing global engine,
    and synthesizes a 'global_champion' dispatch manifest and metrics summary.
    """
    candidates = discover_trained_metrics(models_dir, task=task)
    candidates = [c for c in candidates if c.get("folder_name") not in ["hybrid_champion", "global_champion"]]
    global_cands = [c for c in candidates if c.get("strategy") == "global" or "global" in c.get("folder_name", "").lower()]
    if not global_cands:
        return None

    best_cand = min(global_cands, key=lambda x: x.get("weighted_mae", float("inf")))
    champion_dir = os.path.join(models_dir, "global_champion")
    os.makedirs(champion_dir, exist_ok=True)

    src_folder = best_cand["folder_name"]
    cand_dispatch_file = os.path.join(models_dir, src_folder, "dispatch_config.json")
    model_file = None
    model_type = best_cand.get("model_type", "unknown")
    if os.path.exists(cand_dispatch_file):
        try:
            with open(cand_dispatch_file) as f:
                c_table = json.load(f)
                entry = c_table.get("dispatch", {}).get("__default__")
                if isinstance(entry, dict):
                    model_file = entry.get("file")
                    model_type = entry.get("model_type", model_type)
                elif isinstance(entry, str):
                    model_file = entry
        except Exception:
            pass

    champion_dispatch = {
        "strategy": "global_champion",
        "task": task,
        "description": "Winning Global Engine (selected across competing global architectures)",
        "winning_engine": model_type,
        "source_candidate": src_folder,
        "dispatch": {
            "__default__": {
                "model_type": model_type,
                "source_dir": src_folder,
                "file": model_file,
                "best_test_mae": best_cand.get("weighted_mae"),
                "best_test_smape": best_cand.get("weighted_smape")
            }
        }
    }
    for act, act_m in best_cand.get("activity_metrics", {}).items():
        champion_dispatch["dispatch"][act] = {
            "model_type": model_type,
            "source_dir": src_folder,
            "file": model_file,
            "best_test_mae": act_m.get("mae"),
            "best_test_smape": act_m.get("smape")
        }

    # Copy auxiliary assets (vocab.json, scalers) into global_champion for standalone deployment
    src_dir = os.path.join(models_dir, src_folder)
    v_src = os.path.join(src_dir, "vocab.json")
    if os.path.exists(v_src) and not os.path.exists(os.path.join(champion_dir, "vocab.json")):
        shutil.copy(v_src, os.path.join(champion_dir, "vocab.json"))
    for fname in os.listdir(src_dir):
        if fname.startswith("scaler_") and fname.endswith(".pkl"):
            dest = os.path.join(champion_dir, fname)
            if not os.path.exists(dest):
                shutil.copy(os.path.join(src_dir, fname), dest)

    with open(os.path.join(champion_dir, "dispatch_config.json"), "w") as f:
        json.dump(champion_dispatch, f, indent=2)

    champion_metrics = {
        "task": task,
        "strategy": "global",
        "model_type": "champion",
        "folder_name": "global_champion",
        "training_time_seconds": best_cand.get("training_time_seconds", 0.0),
        "weighted_mae": best_cand.get("weighted_mae", 0.0),
        "weighted_smape": best_cand.get("weighted_smape", 0.0),
        "macro_mae": best_cand.get("macro_mae", 0.0),
        "activity_metrics": best_cand.get("activity_metrics", {})
    }
    with open(os.path.join(champion_dir, "metrics.json"), "w") as f:
        json.dump(champion_metrics, f, indent=2)

    return champion_metrics

def synthesize_champion(models_dir, task="duration", mode="hybrid"):
    """
    Synthesizes the champion deployment according to active mode:
    - 'global_tournament': crowns the best global architecture
    - 'hybrid': synthesizes the per-activity hybrid champion
    - 'single_global': standalone single model, no tournament synthesis needed
    """
    if mode == "global_tournament":
        return synthesize_champion_global(models_dir, task=task)
    elif mode == "single_global":
        return None
    else:
        return synthesize_champion_hybrid(models_dir, task=task)

def print_comparison_table(metrics_list, models_dir, task="duration"):
    """
    Renders an ASCII and Markdown comparison table across all evaluated models.
    """
    if not metrics_list:
        print("\n[!] No trained model metrics found. Train models first using src/04_train.py.")
        return

    task_title = task.replace("_", " ").title()

    # Sort by Weighted MAE (ascending, best first)
    metrics_list = sorted(metrics_list, key=lambda x: x.get("weighted_mae", float("inf")))

    print("\n" + "=" * 95)
    print(f"                     RIMS+ {task_title.upper()} MODEL BENCHMARK RESULTS")
    print("=" * 95)
    header = f"{'Configuration':<28} | {'Strategy':<8} | {'Engine':<14} | {'MAE (s)':>8} | {'SMAPE (%)':>9} | {'Train (s)':>9}"
    print(header)
    print("-" * 95)

    md_lines = [
        f"# RIMS+ {task_title} Model Benchmark Report\n",
        "| Configuration | Strategy | Engine | Test MAE (s) | Test SMAPE (%) | Training Time (s) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: |"
    ]

    for i, m in enumerate(metrics_list):
        cfg_name = m.get("folder_name", "unknown")
        strat = m.get("strategy", "-")
        mtype = m.get("model_type", "-")
        mae = m.get("weighted_mae", 0.0)
        smape = m.get("weighted_smape", 0.0)
        ttime = m.get("training_time_seconds", 0.0)

        is_champion = (i == 0)
        star = " ★ (Champion)" if is_champion else ""
        print(f"{cfg_name:<28} | {strat:<8} | {mtype:<14} | {mae:>8.2f} | {smape:>8.2f}% | {ttime:>8.2f}s{star}")

        cfg_label = f"**{cfg_name}**" if is_champion else cfg_name
        md_lines.append(f"| {cfg_label} | `{strat}` | `{mtype}` | **{mae:.2f}** | {smape:.2f}% | {ttime:.2f}s |")

    print("=" * 95)

    # Activity-level breakdown
    all_acts = set()
    for m in metrics_list:
        all_acts.update(m.get("activity_metrics", {}).keys())
    all_acts = sorted(list(all_acts))

    if all_acts:
        print("\n" + "=" * 115)
        print("                        ACTIVITY-LEVEL MAE COMPARISON (seconds)")
        print("=" * 115)
        act_header = f"{'Activity':<32} | " + " | ".join([f"{m.get('folder_name')[:14]:>14}" for m in metrics_list])
        print(act_header)
        print("-" * 115)

        md_lines.append("\n## Activity-Level MAE Breakdown (seconds)\n")
        md_act_header = "| Activity | " + " | ".join([f"`{m.get('folder_name')}`" for m in metrics_list]) + " |"
        md_act_sep = "| :--- | " + " | ".join([":---:" for _ in metrics_list]) + " |"
        md_lines.append(md_act_header)
        md_lines.append(md_act_sep)

        for act in all_acts:
            row_raw_maes = []
            for m in metrics_list:
                act_data = m.get("activity_metrics", {}).get(act)
                if act_data:
                    row_raw_maes.append(act_data.get("mae", float("inf")))
                else:
                    row_raw_maes.append(float("inf"))

            min_act_mae = min(row_raw_maes) if row_raw_maes else 0.0

            row_strs = []
            md_row_strs = []
            for val in row_raw_maes:
                if val != float("inf"):
                    best_mark = "*" if val == min_act_mae else " "
                    row_strs.append(f"{val:>13.2f}{best_mark}")
                    md_val = f"**{val:.2f}**" if val == min_act_mae else f"{val:.2f}"
                    md_row_strs.append(md_val)
                else:
                    row_strs.append(f"{'N/A':>14}")
                    md_row_strs.append("N/A")

            print(f"{act:<32} | " + " | ".join(row_strs))
            md_lines.append(f"| {act} | " + " | ".join(md_row_strs) + " |")

        print("=" * 115)
        print("(* indicates best performer for that specific activity)\n")

    # Add Champion Hybrid Assignment table to Markdown
    champ_dispatch_file = os.path.join(models_dir, "hybrid_champion", "dispatch_config.json")
    if os.path.exists(champ_dispatch_file):
        try:
            with open(champ_dispatch_file) as f:
                c_data = json.load(f)
            md_lines.append("\n## 🏆 Hybrid Champion Architecture Assignment\n")
            md_lines.append("| Activity | Winning Model | Model Engine | Test MAE (s) | Best SMAPE (%) |")
            md_lines.append("| :--- | :--- | :---: | :---: | :---: |")
            for act, info in sorted(c_data.get("dispatch", {}).items()):
                if act == "__default__":
                    continue
                s_dir = info.get("source_dir", "-")
                m_type = info.get("model_type", "-").upper()
                mae_val = info.get("best_test_mae", "-")
                smape_val = info.get("best_test_smape", "-")
                md_lines.append(f"| **{act}** | `{s_dir}` | **{m_type}** | **{mae_val}s** | {smape_val}% |")
            def_info = c_data.get("dispatch", {}).get("__default__", {})
            if def_info:
                md_lines.append(f"| *Fallback (`__default__`)* | `{def_info.get('source_dir')}` | `{def_info.get('model_type', '').upper()}` | — | — |")
        except Exception:
            pass

    glob_champ_file = os.path.join(models_dir, "global_champion", "dispatch_config.json")
    if os.path.exists(glob_champ_file):
        try:
            with open(glob_champ_file) as f:
                g_data = json.load(f)
            def_info = g_data.get("dispatch", {}).get("__default__", {})
            md_lines.append("\n## 🏆 Global Champion Architecture Assignment\n")
            md_lines.append("| Winning Global Model | Engine | Test MAE (s) | Best SMAPE (%) |")
            md_lines.append("| :--- | :---: | :---: | :---: |")
            s_dir = def_info.get("source_dir", "-")
            m_type = def_info.get("model_type", "-").upper()
            mae_val = def_info.get("best_test_mae", "-")
            smape_val = def_info.get("best_test_smape", "-")
            md_lines.append(f"| `{s_dir}` | **{m_type}** | **{mae_val}s** | {smape_val}% |")
        except Exception:
            pass

    # Save Markdown report
    report_path = os.path.join(models_dir, "benchmark_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[✓] Saved comprehensive benchmark report to: {report_path}\n")

def run_training_experiment(strategy, model_type, task="duration"):
    """
    Executes a single training run via python src/04_train.py.
    """
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "src", "04_train.py"),
        "--task", task,
        "--strategy", strategy,
        "--model_type", model_type
    ]
    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="RIMS+ Model Benchmarking & Comparison Suite")
    parser.add_argument("--task", type=str, default="duration", choices=["duration", "waiting_time"], help="Task to benchmark")
    parser.add_argument("--mode", type=str, default=None, choices=["hybrid", "global_tournament", "single_global"],
                        help="Exploration mode (hybrid, global_tournament, single_global). Defaults to config.yaml")
    parser.add_argument("--compare", action="store_true", help="Compare already trained models in models/ and synthesize champion")
    parser.add_argument("--run_all", action="store_true", help="Train candidate models and synthesize champion")
    args = parser.parse_args()

    cfg = load_config()
    strat_cfg = cfg.get("model_strategy", {})
    mode = args.mode or strat_cfg.get("mode", "hybrid")

    models_dir = os.path.join(PROJECT_ROOT, cfg["paths"]["models_dir"], args.task)
    os.makedirs(models_dir, exist_ok=True)

    if args.run_all:
        print("\n" + "=" * 80)
        print(f"      LAUNCHING RIMS+ [{args.task.upper()}] BENCHMARK SUITE")
        print(f"      Mode: [{mode.upper()}]")
        print("=" * 80)

        if mode == "single_global":
            engine = strat_cfg.get("single_global_engine", "xgboost")
            run_training_experiment("global", engine, task=args.task)
        else:
            candidates_list = strat_cfg.get("candidates", {}).get(mode, [])
            if not candidates_list:
                if mode == "global_tournament":
                    candidates_list = [
                        {"strategy": "global", "engine": "xgboost"},
                        {"strategy": "global", "engine": "tcn"}
                    ]
                else:
                    candidates_list = [
                        {"strategy": "global", "engine": "xgboost"},
                        {"strategy": "local", "engine": "xgboost"},
                        {"strategy": "global", "engine": "tcn"},
                        {"strategy": "local", "engine": "tcn"}
                    ]
            for cand in candidates_list:
                run_training_experiment(cand["strategy"], cand["engine"], task=args.task)

    # Synthesize the Champion from available candidate models
    champion = synthesize_champion(models_dir, task=args.task, mode=mode)
    if champion:
        label = "Hybrid" if mode == "hybrid" else "Global"
        folder = "hybrid_champion" if mode == "hybrid" else "global_champion"
        print(f"[✓] Synthesized Best-of-All-Worlds {label} Champion: {champion['weighted_mae']}s MAE ({champion['weighted_smape']}% SMAPE)")
        print(f"    Saved champion dispatch table to: {os.path.join(models_dir, folder, 'dispatch_config.json')}")

    # Discover and display the full comparison leaderboard
    metrics_list = discover_trained_metrics(models_dir, task=args.task)
    print_comparison_table(metrics_list, models_dir, task=args.task)

if __name__ == "__main__":
    main()
