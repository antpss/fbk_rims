import os
import pm4py
import pandas as pd
from pm4py.objects.log.obj import EventLog, Trace, Event
from config_loader import load_config, PROJECT_ROOT

def main():
    cfg = load_config()
    raw_rel = cfg["paths"].get("raw_log", "data/processed/BPI_2012_filtered.xes")
    input_log_path = raw_rel if os.path.isabs(raw_rel) else os.path.join(PROJECT_ROOT, raw_rel)
    if not os.path.exists(input_log_path):
        fallback_path = os.path.join(PROJECT_ROOT, "data/processed/BPI_2012_W_only.xes")
        if os.path.exists(fallback_path):
            input_log_path = fallback_path
    input_pnml_path = cfg["paths"].get("petri_net", os.path.join(PROJECT_ROOT, "models/petri_nets/discovered_model_split.pnml"))
    output_log_path = cfg["paths"].get("aligned_log", os.path.join(PROJECT_ROOT, "data/processed/aligned_BPI_2012.xes"))

    os.makedirs(os.path.dirname(output_log_path), exist_ok=True)

    prep_cfg = cfg.get("preprocessing", {})
    proc_prefixes = tuple(prep_cfg.get("process_activity_prefixes", ["W_"]))
    milestone_prefixes = tuple(prep_cfg.get("milestone_prefixes", ["A_", "O_"]))

    print(f"Loading log: {input_log_path}")
    df = pm4py.read_xes(input_log_path)
    print(f"Loaded {len(df)} events across {df['case:concept:name'].nunique()} cases.")

    print(f"Loading Petri net model: {input_pnml_path}")
    net, initial_marking, final_marking = pm4py.read_pnml(input_pnml_path)

    #isolate process activity COMPLETE events for alignment
    #PM4Py Petri net transitions correspond to process activity completions.
    print(f"Preparing control-flow log on COMPLETE events of process activities ({proc_prefixes})...")
    df_proc_comp = df[(df["concept:name"].str.startswith(proc_prefixes)) & (df["lifecycle:transition"] == "COMPLETE")].copy()
    
    # Build alignment event log with case:concept:name as trace concept:name
    log_for_align = EventLog()
    for cid, grp in df_proc_comp.groupby("case:concept:name", sort=False):
        grp_sorted = grp.sort_values(by="time:timestamp")
        events_list = [Event({"concept:name": act, "time:timestamp": ts}) for act, ts in zip(grp_sorted["concept:name"], grp_sorted["time:timestamp"])]
        log_for_align.append(Trace(events_list, attributes={"concept:name": str(cid)}))

    print(f"Calculating alignments for {len(log_for_align)} process traces...")
    raw_alignments = pm4py.conformance_diagnostics_alignments(log_for_align, net, initial_marking, final_marking)
    
    alignments_by_case = {}
    for i, trace in enumerate(log_for_align):
        cid = trace.attributes["concept:name"]
        alignments_by_case[cid] = raw_alignments[i]["alignment"]
    print("Alignments calculated successfully.")

    #reconstruct repaired log
    print("Reconstructing repaired event log preserving SCHEDULE/START lifecycles and business milestones...")
    repaired_log = EventLog()
    
    total_model_moves = 0
    total_log_moves = 0
    total_sync_moves = 0

    cases_grouped = df.groupby("case:concept:name", sort=False)
    for cid, grp in cases_grouped:
        cid_str = str(cid)
        grp_sorted = grp.sort_values(by="time:timestamp").reset_index(drop=True)

        # Extract trace-level case attributes (without 'case:' prefix)
        trace_attrs = {"concept:name": cid_str}
        for col in grp_sorted.columns:
            if col.startswith("case:") and col != "case:concept:name":
                clean_key = col[5:]
                trace_attrs[clean_key] = grp_sorted[col].iloc[0]

        # Separate milestones vs process activity executions
        milestones = []
        work_items = []
        pending_sched = {}
        pending_start = {}

        for _, row in grp_sorted.iterrows():
            act = row["concept:name"]
            trans = str(row.get("lifecycle:transition", "COMPLETE")).upper()
            row_dict = {k: v for k, v in row.dropna().to_dict().items() if not k.startswith("case:")}
            evt = Event(row_dict)

            if act.startswith(milestone_prefixes):
                milestones.append(evt)
            elif act.startswith(proc_prefixes):
                if trans == "SCHEDULE":
                    pending_sched[act] = evt
                elif trans == "START":
                    pending_start[act] = evt
                elif trans == "COMPLETE":
                    wi = {
                        "act": act,
                        "schedule": pending_sched.pop(act, None),
                        "start": pending_start.pop(act, None),
                        "complete": evt
                    }
                    work_items.append(wi)

        repaired_events = []
        alignment = alignments_by_case.get(cid_str)

        if alignment and len(work_items) > 0:
            wi_idx = 0
            curr_time = grp_sorted["time:timestamp"].iloc[0]

            for step in alignment:
                log_move = step[0]
                model_move = step[1]

                # Skip silent tau transitions
                if model_move is None or str(model_move).startswith("tau") or model_move == "tau":
                    continue

                if log_move != ">>" and model_move != ">>":
                    # Sync move: keep real work item with full lifecycle
                    total_sync_moves += 1
                    if wi_idx < len(work_items):
                        wi = work_items[wi_idx]
                        wi_idx += 1
                        if wi["schedule"] is not None:
                            repaired_events.append(wi["schedule"])
                        if wi["start"] is not None:
                            repaired_events.append(wi["start"])
                        repaired_events.append(wi["complete"])
                        curr_time = wi["complete"]["time:timestamp"]

                elif model_move == ">>":
                    # Log move: activity violation, skip
                    total_log_moves += 1
                    if wi_idx < len(work_items):
                        wi_idx += 1

                elif log_move == ">>":
                    # Model move: required by Petri net, insert synthetic event
                    total_model_moves += 1
                    fake_evt = Event({
                        "concept:name": model_move,
                        "lifecycle:transition": "COMPLETE",
                        "time:timestamp": curr_time,
                        "org:resource": "SYNTHETIC",
                        "is_synthetic": "true"
                    })
                    repaired_events.append(fake_evt)
        else:
            # For cases without alignment or non-process traces, retain all original work items
            for wi in work_items:
                if wi["schedule"] is not None:
                    repaired_events.append(wi["schedule"])
                if wi["start"] is not None:
                    repaired_events.append(wi["start"])
                repaired_events.append(wi["complete"])

        # Preserve all business milestone events (A_, O_)
        repaired_events.extend(milestones)

        # Sort all events chronologically (stable sort)
        repaired_events.sort(key=lambda e: e["time:timestamp"])

        repaired_trace = Trace(repaired_events, attributes=trace_attrs)
        repaired_log.append(repaired_trace)

    print(f"Alignment Summary:")
    print(f"  - Synchronous moves: {total_sync_moves}")
    print(f"  - Log moves dropped: {total_log_moves}")
    print(f"  - Synthetic model moves: {total_model_moves}")

    print(f"Exporting repaired aligned log to {output_log_path}...")
    pm4py.write_xes(repaired_log, output_log_path)
    print("Alignment and log repair completed successfully.")

if __name__ == "__main__":
    main()
