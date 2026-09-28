import os
import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from config_loader import load_config, PROJECT_ROOT

def main():
    cfg = load_config()
    input_log_path = cfg["paths"].get("raw_log", os.path.join(PROJECT_ROOT, "data/processed/BPI_2012_W_only.xes"))
    input_pnml_path = cfg["paths"].get("petri_net", os.path.join(PROJECT_ROOT, "models/petri_nets/discovered_model_split.pnml"))
    output_log_path = cfg["paths"].get("aligned_log", os.path.join(PROJECT_ROOT, "data/processed/aligned_BPI_2012.xes"))

    os.makedirs(os.path.dirname(output_log_path), exist_ok=True)

    #data loading
    print(f"Loading log: {input_log_path}")
    log = pm4py.read_xes(input_log_path, return_legacy_log_object=True)

    print(f"Loading Split Miner model: {input_pnml_path}")
    net, initial_marking, final_marking = pm4py.read_pnml(input_pnml_path)

    #aAlignments calculation
    print("Calculating alignments...")
    alignments = pm4py.conformance_diagnostics_alignments(log, net, initial_marking, final_marking)

    #log repair
    print("Repairing log based on alignments...")
    repaired_log = EventLog()
    
    for trace_idx, trace in enumerate(log):
        alignment = alignments[trace_idx]['alignment']
        repaired_trace = Trace(attributes=trace.attributes)
        
        event_idx = 0
        last_timestamp = None
        
        for step in alignment:
            log_move = step[0]
            model_move = step[1]
            
            # Sync move: Keep original event
            if log_move != ">>" and model_move != ">>":
                real_event = trace[event_idx]
                repaired_trace.append(real_event)
                if "time:timestamp" in real_event:
                    last_timestamp = real_event["time:timestamp"]
                event_idx += 1
            
            # Log move: Event not allowed by model, skip it
            elif model_move == ">>":
                event_idx += 1
            
            # Model move: Model requires event, insert it
            elif log_move == ">>" and model_move is not None:
                if model_move != "tau" and not model_move.startswith("tau"):
                    fake_event = Event()
                    fake_event["concept:name"] = model_move
                    fake_event["lifecycle:transition"] = "COMPLETE"
                    if last_timestamp:
                        fake_event["time:timestamp"] = last_timestamp
                    repaired_trace.append(fake_event)
        
        repaired_log.append(repaired_trace)

    #export
    print(f"Exporting aligned log to {output_log_path}...")
    pm4py.write_xes(repaired_log, output_log_path)
    print("Done.")

if __name__ == "__main__":
    main()

