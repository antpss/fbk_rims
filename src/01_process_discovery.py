import argparse
import pm4py
from pm4py.algo.analysis.woflan import algorithm as woflan

def main():
    # CHOICE OF MINER
    parser = argparse.ArgumentParser(description="Petri net discovery using Split Miner or Inductive Miner.")
    parser.add_argument("--miner", choices=["split", "inductive"], default="split")
    args = parser.parse_args()

    #paths
    #input_log_path = "../RIMS_tool/core/example/example_decision_mining/BPIChallenge2012A.xes"
    input_log_path = "../data/raw/BPI_Challenge_2012.xes"
    output_pnml_path = f"../models/discovered_model_{args.miner}.pnml"

    print(f"Loading event log from: {input_log_path}...")
    log = pm4py.read_xes(input_log_path)
    print(f"Log loaded successfully. Number of traces: {len(log)}")

    if args.miner == "split":
        # Discover BPMN model using Split Miner and convert to Petri Net
        print("Discovering BPMN model using Split Miner...")
        bpmn_model = pm4py.discover_bpmn_split_miner(log)
        print("Converting BPMN to Petri Net...")
        net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn_model)
    elif args.miner == "inductive":
        # Discover Petri Net directly using Inductive Miner (with noise filtering)
        print("Discovering Petri Net using Inductive Miner...")
        net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log, noise_threshold=0.2)

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
