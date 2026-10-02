import os
import argparse
import pm4py
from pm4py.algo.analysis.woflan import algorithm as woflan
from config_loader import load_config, PROJECT_ROOT

def main():
    cfg = load_config()
    default_miner = cfg.get("discovery", {}).get("miner", "split")
    default_noise = cfg.get("discovery", {}).get("inductive_noise_threshold", 0.2)

    parser = argparse.ArgumentParser(description="Petri net discovery using Split Miner or Inductive Miner.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--miner", choices=["split", "inductive"], default="split",
                        help=f"Miner choice (default: 'split')")
    parser.add_argument("--noise_threshold", type=float, default=None,
                        help=f"Noise threshold for inductive miner (default from config: {default_noise})")
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        default_noise = cfg.get("discovery", {}).get("inductive_noise_threshold", 0.2)

    chosen_miner = args.miner
    noise_threshold = args.noise_threshold if args.noise_threshold is not None else default_noise

    # Paths resolved from config
    raw_rel = cfg["paths"].get("raw_log", "data/processed/BPI_2012_filtered.xes")
    input_log_path = raw_rel if os.path.isabs(raw_rel) else os.path.join(PROJECT_ROOT, raw_rel)
    if not os.path.exists(input_log_path):
        fallback_path = os.path.join(PROJECT_ROOT, "data/processed/BPI_2012_W_only.xes")
        if os.path.exists(fallback_path):
            print(f"[Notice] '{input_log_path}' not found, falling back to existing '{fallback_path}'.")
            input_log_path = fallback_path
    petri_nets_dir = os.path.join(PROJECT_ROOT, cfg["paths"].get("petri_nets_dir", "models/petri_nets"))
    os.makedirs(petri_nets_dir, exist_ok=True)
    output_pnml_path = os.path.join(petri_nets_dir, f"discovered_model_{chosen_miner}.pnml")

    print(f"Loading event log from: {input_log_path}...")
    log = pm4py.read_xes(input_log_path)
    print(f"Log loaded successfully. Number of traces: {len(log)}")

    #filter strictly to process activities and COMPLETE events
    prep_cfg = cfg.get("preprocessing", {})
    proc_prefixes = prep_cfg.get("process_activity_prefixes", [])
    
    import pandas as pd
    df_disc = pm4py.convert_to_dataframe(log) if not isinstance(log, pd.DataFrame) else log.copy()
    if proc_prefixes:
        df_disc = df_disc[df_disc["concept:name"].str.startswith(tuple(proc_prefixes))].copy()
    if "lifecycle:transition" in df_disc.columns:
        df_disc = df_disc[df_disc["lifecycle:transition"].str.upper() == "COMPLETE"].copy()

    case_counts = df_disc.groupby("case:concept:name").size()
    valid_cases = case_counts[case_counts > 0].index
    df_disc = df_disc[df_disc["case:concept:name"].isin(valid_cases)].copy()

    print(f"Discovery log filtered to COMPLETE events: {len(df_disc)} events across {df_disc['concept:name'].nunique()} process activities.")
    discovery_log = pm4py.convert_to_event_log(df_disc)

    if chosen_miner == "split":
        # Discover BPMN model using Split Miner and convert to Petri Net
        print("Discovering BPMN model using Split Miner...")
        bpmn_model = pm4py.discover_bpmn_split_miner(discovery_log)
        print("Converting BPMN to Petri Net...")
        net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn_model)

    elif chosen_miner == "inductive":
        # Discover Petri Net directly using Inductive Miner (with noise filtering)
        print(f"Discovering Petri Net using Inductive Miner (noise_threshold={noise_threshold})...")
        net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(discovery_log, noise_threshold=noise_threshold)

    print(f"Petri Net created with {len(net.places)} places and {len(net.transitions)} transitions.")

    # Validate the structural soundness of the discovered net
    print("Running soundness check...")
    is_sound = woflan.apply(net, initial_marking, final_marking)
    
    if is_sound:
        print("SUCCESS: The discovered Petri net is perfectly SOUND.")
    else:
        print("WARNING: The discovered Petri net is NOT sound. It may contain deadlocks.")

    # Export the final model
    print(f"Exporting Petri net to {output_pnml_path}...")
    pm4py.write_pnml(net, initial_marking, final_marking, output_pnml_path)
    print("Done!")

if __name__ == "__main__":
    main()
