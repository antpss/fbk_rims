#!/usr/bin/env python3
"""
src/06_simulate.py

CLI Entrypoint for RIMS++ Discrete-Event Simulation Engine.
Orchestrates discrete-event simulation via the modular OOP package `src.simulator`:
  - SimulationProcess & SharedWorkerPool (Multi-skilled resource engine)
  - Token (PM4Py Petri net token semantics & XOR Hyperopt decision routing)
  - ChampionPredictor (Runtime inference for XGBoost, TCN, LSTM champions)
  - CalendarManager (Global & per-role shift schedules)
  - ArrivalGenerator (Historical replay & generative arrivals)
  - SimulationEvaluator (IEEE XES/CSV export & benchmark validation)
"""

import os
import sys
import argparse
import random
import numpy as np

# Ensure src is on sys.path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from config_loader import load_config, PROJECT_ROOT
except ImportError:
    from src.config_loader import load_config, PROJECT_ROOT

from simulator import (
    SimulationProcess,
    DiscreteEventSimulation,
    SharedWorkerPool,
    WorkerManager,
    ChampionPredictor,
    CalendarManager,
    ArrivalGenerator,
    Token,
    SimulationEvaluator,
    format_dur,
)


def main():
    parser = argparse.ArgumentParser(description="RIMS++ Modern Discrete-Event Simulator")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--cases", type=str, default=None,
                        help="Number of cases to simulate (e.g. 1000, 2500, or 'all')")
    parser.add_argument("--all", action="store_true",
                        help="Simulate all cases available in the event log")
    parser.add_argument("--mode", type=str, default=None, choices=["hybrid_residual", "pure_physics"],
                        help="Simulation waiting time mode")
    parser.add_argument("--arrival_mode", type=str, default=None, choices=["replay", "generative", "prophet"],
                        help="Arrival generation mode: 'replay' from aligned log, 'generative' distribution, or 'prophet' time-series")
    parser.add_argument("--export_xes", action="store_true", default=None,
                        help="Export standard IEEE XES log (.xes)")
    parser.add_argument("--pnml", type=str, default=None,
                        help="Path to override Petri net PNML file")
    parser.add_argument("--duration", type=float, default=None,
                        help="Simulation horizon duration in days")
    parser.add_argument("--compare_modes", action="store_true",
                        help="Run 2-way comparison study between pure_physics and hybrid_residual")
    parser.add_argument("--force_retrain_prophet", action="store_true",
                        help="Force retraining of Prophet arrival model ignoring disk cache")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("project", {}).get("random_seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

    if args.all or (args.cases and str(args.cases).lower() == "all"):
        num_cases = None
    elif args.cases is not None:
        num_cases = int(args.cases)
    else:
        num_cases = cfg.get("simulation", {}).get("num_cases", 1000)

    arrival_mode = args.arrival_mode or cfg.get("simulation", {}).get("arrival", {}).get("mode", "replay")
    export_xes = args.export_xes if args.export_xes is not None else cfg.get("simulation", {}).get("export_xes", True)
    pnml_path = args.pnml or os.path.join(PROJECT_ROOT, cfg.get("paths", {}).get("petri_net", "models/petri_nets/discovered_model_split.pnml"))

    if args.compare_modes:
        modes = ["pure_physics", "hybrid_residual"]
        results = []
        case_label = f"{num_cases:,}" if num_cases else "ALL"
        print("\n" + "=" * 105)
        print("                        RIMS++ 2-WAY SIMULATION BENCHMARK SUITE")
        print(f"                        Evaluating {case_label} cases (Pure Physics vs. Hybrid Residual)")
        print("=" * 105)

        for m in modes:
            print(f"\n>>> Running Mode: [{m.upper()}] ...")
            sim = SimulationProcess(
                cfg,
                num_cases=num_cases,
                mode=m,
                arrival_mode=arrival_mode,
                export_xes=export_xes,
                pnml_path=pnml_path,
                duration_days=args.duration,
                force_retrain_prophet=args.force_retrain_prophet,
            )
            res = sim.run()
            results.append(res)

        print("\n" + "=" * 140)
        print("                                RIMS++ MULTI-PERSPECTIVE BENCHMARK LEADERBOARD")
        print("=" * 140)
        print(f"{'Simulation Mode':<22} | {'CFLD':>8} | {'L_traces':>8} | {'2-Gram':>7} | {'AED (h)':>8} | {'CED (h)':>8} | {'RED (h)':>8} | {'CAR (h)':>8} | {'CTD (WD)':>14} | {'MAE Cycle':>14} | {'Run Time':>9}")
        print("-" * 140)
        real_m = results[0]["real_mean"]
        real_med = results[0]["real_median"]
        print(f"{'REAL HISTORICAL':<22} | {'---':>8} | {'---':>8} | {'---':>7} | {'---':>8} | {'---':>8} | {'---':>8} | {'---':>8} | {'---':>14} | {'---':>14} | {'---':>9}")
        print("-" * 140)

        # Sort by lowest Wasserstein distance (cycle time distribution fidelity)
        sorted_res = sorted(results, key=lambda x: x["wd"])
        best_mode = sorted_res[0]["mode"]
        for r in sorted_res:
            mode_display = f"{r['mode']} ★" if r["mode"] == best_mode else r["mode"]
            cfld_str = f"{r.get('cfld', 0.0):.4f}"
            len_str = f"{r.get('l_traces', 0.0):.2f}"
            ng2_str = f"{r.get('ngram_2', 0.0):.4f}"
            aed_str = f"{r.get('aed', 0.0):.1f}"
            ced_str = f"{r.get('ced', 0.0):.1f}"
            red_str = f"{r.get('red_dist', 0.0):.1f}"
            car_str = f"{r.get('car', 0.0):.1f}"
            wd_str = f"{r['wd']/86400:.2f} days" if r["wd"] >= 86400 else format_dur(r["wd"])
            mae_str = f"{r['mae_cycle']/86400:.2f} days" if r["mae_cycle"] >= 86400 else format_dur(r["mae_cycle"])
            print(f"{mode_display:<22} | {cfld_str:>8} | {len_str:>8} | {ng2_str:>7} | {aed_str:>8} | {ced_str:>8} | {red_str:>8} | {car_str:>8} | {wd_str:>14} | {mae_str:>14} | {r['elapsed']:>8.1f}s")
        print("-" * 140)
        print("  ★ Note: Evaluated across Control-Flow, Temporal, and Congestion perspectives (Meneghello et al. 2025).")
        print("=" * 140 + "\n")

        # Save Ablation Report
        for rep_name in ["ablation_report.md", f"ablation_report_{arrival_mode}.md"]:
            abl_path = os.path.join(PROJECT_ROOT, "data/processed", rep_name)
            with open(abl_path, "w") as f:
                f.write(f"# RIMS++ Multi-Perspective Simulation Benchmark Report ({arrival_mode.upper()} ARRIVAL)\n\n")
                f.write(f"- **Arrival Mode:** `{arrival_mode}`\n")
                f.write(f"- **Evaluated Cases per Mode:** {case_label}\n")
                f.write(f"- **Real Historical Mean Cycle Time:** {format_dur(real_m)}\n")
                f.write(f"- **Evaluation Suite:** Meneghello et al. (Information Systems 2025) / Chapela-Campa et al. (BPM 2023)\n\n")
                f.write("## Comparative Benchmark Leaderboard\n\n")
                f.write("| Simulation Mode | CFLD | $L^{traces}$ | NGD 2-Gram | AED (h) | CED (h) | RED (h) | CAR (h) | CTD (WD) | Cycle MAE | Run Time |\n")
                f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
                f.write(f"| **REAL HISTORICAL** | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
                for r in sorted_res:
                    tag = " ★ (Best Fit)" if r["mode"] == best_mode else ""
                    cfld_str = f"{r.get('cfld', 0.0):.4f}"
                    len_str = f"{r.get('l_traces', 0.0):.2f}"
                    ng2_str = f"{r.get('ngram_2', 0.0):.4f}"
                    aed_str = f"{r.get('aed', 0.0):.2f}"
                    ced_str = f"{r.get('ced', 0.0):.2f}"
                    red_str = f"{r.get('red_dist', 0.0):.2f}"
                    car_str = f"{r.get('car', 0.0):.2f}"
                    wd_str = f"{r['wd']/86400:.2f} days" if r["wd"] >= 86400 else format_dur(r["wd"])
                    mae_str = f"{r['mae_cycle']/86400:.2f} days" if r["mae_cycle"] >= 86400 else format_dur(r["mae_cycle"])
                    f.write(f"| `{r['mode']}`{tag} | {cfld_str} | {len_str} | {ng2_str} | {aed_str} | {ced_str} | {red_str} | {car_str} | {wd_str} | {mae_str} | {r['elapsed']:.1f}s |\n")
            print(f"[✓] Saved comprehensive ablation report to: {abl_path}\n")

    else:
        mode = args.mode or cfg.get("simulation", {}).get("mode", "hybrid_residual")
        sim = SimulationProcess(
            cfg,
            num_cases=num_cases,
            mode=mode,
            arrival_mode=arrival_mode,
            export_xes=export_xes,
            pnml_path=pnml_path,
            duration_days=args.duration,
            force_retrain_prophet=args.force_retrain_prophet,
        )
        sim.run()


if __name__ == "__main__":
    main()
