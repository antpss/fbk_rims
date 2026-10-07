import os
import argparse
import pandas as pd
import pm4py
from config_loader import load_config, PROJECT_ROOT

def main():
    cfg = load_config()
    # CLI setup to pick which model to evaluate
    parser = argparse.ArgumentParser()
    parser.add_argument("--miner", choices=["split", "inductive"], default="split")
    parser.add_argument("--use-aligned", action="store_true", help="Evaluate conformance on the repaired aligned log")
    args = parser.parse_args()

    if args.use_aligned:
        log_rel = cfg["paths"].get("aligned_log", "data/processed/aligned_BPI_2012.xes")
    else:
        log_rel = cfg["paths"].get("raw_log", "data/processed/BPI_2012_filtered.xes")

    input_log_path = log_rel if os.path.isabs(log_rel) else os.path.join(PROJECT_ROOT, log_rel)
    if not os.path.exists(input_log_path):
        fallback_path = os.path.join(PROJECT_ROOT, "data/processed/BPI_2012_W_only.xes")
        if os.path.exists(fallback_path):
            input_log_path = fallback_path
    petri_nets_dir = os.path.join(PROJECT_ROOT, cfg["paths"].get("petri_nets_dir", "models/petri_nets"))
    input_pnml_path = os.path.join(petri_nets_dir, f"discovered_model_{args.miner}.pnml")

    # data loading
    print(f"Loading log: {input_log_path}")
    log = pm4py.read_xes(input_log_path)
    
    print(f"Loading model: {input_pnml_path}")
    net, initial_marking, final_marking = pm4py.read_pnml(input_pnml_path)

    # Filter evaluation log to match the Petri net's abstraction level:
    # Process activities (W_) with COMPLETE lifecycle transitions.
    prep_cfg = cfg.get("preprocessing", {})
    proc_prefixes = tuple(prep_cfg.get("process_activity_prefixes", ["W_"]))
    
    df = pm4py.convert_to_dataframe(log) if not isinstance(log, pd.DataFrame) else log.copy()
    if proc_prefixes:
        df = df[df["concept:name"].str.startswith(proc_prefixes)].copy()
    if "lifecycle:transition" in df.columns:
        df = df[df["lifecycle:transition"].str.upper() == "COMPLETE"].copy()

    case_counts = df.groupby("case:concept:name").size()
    valid_cases = case_counts[case_counts > 0].index
    df = df[df["case:concept:name"].isin(valid_cases)].copy()

    print(f"Evaluation log prepared: {len(df)} COMPLETE events across {len(valid_cases)} cases.")
    eval_log = pm4py.convert_to_event_log(df)

    # --- FULL ALIGNMENTS (SLOW) ---
    # print("Calculating fitness (hang tight, computing alignments)...")
    # fitness_dict = pm4py.fitness_alignments(eval_log, net, initial_marking, final_marking)
    # fitness = fitness_dict.get('log_fitness', 0.0)

    # --- TOKEN-BASED REPLAY (FAST) ---
    print("Calculating fitness (Fast Token-Based Replay)...")
    fitness_dict = pm4py.fitness_token_based_replay(eval_log, net, initial_marking, final_marking)
    fitness = fitness_dict.get('log_fitness', 0.0)

    # --- FULL ALIGNMENTS PRECISION (SLOW) ---
    # print("Calculating precision...")
    # precision = pm4py.precision_alignments(eval_log, net, initial_marking, final_marking)

    # --- TOKEN-BASED PRECISION (FAST) ---
    print("Calculating precision (Fast Token-Based Replay)...")
    precision = pm4py.precision_token_based_replay(eval_log, net, initial_marking, final_marking)

    # Calculate F1-score
    if fitness + precision > 0:
        f1_score = 2 * (fitness * precision) / (fitness + precision)
    else:
        f1_score = 0.0

    #final metrics
    print("\n--- Conformance Results ---")
    print(f"Model: {args.miner.upper()} MINER")
    print(f"Fitness:   {fitness:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")
    print("---------------------------\n")

if __name__ == "__main__":
    main()

