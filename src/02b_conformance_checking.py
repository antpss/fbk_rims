import os
import argparse
import pm4py
from config_loader import load_config, PROJECT_ROOT

def main():
    cfg = load_config()
    # CLI setup to pick which model to evaluate
    parser = argparse.ArgumentParser()
    parser.add_argument("--miner", choices=["split", "inductive"], default="split")
    args = parser.parse_args()

    input_log_path = cfg["paths"].get("raw_log", os.path.join(PROJECT_ROOT, "data/processed/BPI_2012_W_only.xes"))
    petri_nets_dir = os.path.join(PROJECT_ROOT, cfg["paths"].get("petri_nets_dir", "models/petri_nets"))
    input_pnml_path = os.path.join(petri_nets_dir, f"discovered_model_{args.miner}.pnml")

    # data loading
    print(f"Loading log: {input_log_path}")
    log = pm4py.read_xes(input_log_path)
    
    print(f"Loading model: {input_pnml_path}")
    net, initial_marking, final_marking = pm4py.read_pnml(input_pnml_path)

    # --- FULL ALIGNMENTS (SLOW) ---
    # print("Calculating fitness (hang tight, computing alignments)...")
    # fitness_dict = pm4py.fitness_alignments(log, net, initial_marking, final_marking)
    # fitness = fitness_dict.get('log_fitness', 0.0)

    # --- TOKEN-BASED REPLAY (FAST) ---
    print("Calculating fitness (Fast Token-Based Replay)...")
    fitness_dict = pm4py.fitness_token_based_replay(log, net, initial_marking, final_marking)
    fitness = fitness_dict.get('log_fitness', 0.0)

    # --- FULL ALIGNMENTS PRECISION (SLOW) ---
    # print("Calculating precision...")
    # precision = pm4py.precision_alignments(log, net, initial_marking, final_marking)

    # --- TOKEN-BASED PRECISION (FAST) ---
    print("Calculating precision (Fast Token-Based Replay)...")
    precision = pm4py.precision_token_based_replay(log, net, initial_marking, final_marking)

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

