import os
import pandas as pd
import pm4py

def main():
    input_log_path = "../data/processed/aligned_BPI_2012.xes"
    output_dir = "../data/processed/datasets"
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading aligned log: {input_log_path}")
    df = pm4py.read_xes(input_log_path)
    
    # Sort everything chronologically to simulate passage of time
    print("Sorting log chronologically...")
    df = df.sort_values(by=["time:timestamp"])
    
    # Global state variables for WIP and RP_OC
    active_cases = set()
    busy_resources = set()
    
    # Calculate total unique resources to compute the RP_OC percentage
    # (Avoid division by zero if resource column is missing)
    total_resources = df['org:resource'].nunique() if 'org:resource' in df.columns else 1
    if total_resources == 0: total_resources = 1
    
    # Track how many events each case has, so we know when to remove them from active_cases
    case_event_counts = df['case:concept:name'].value_counts().to_dict()
    seen_events = {case_id: 0 for case_id in case_event_counts}
    
    # Track per-case state
    case_prefixes = {}
    case_start_times = {} 
    
    activity_datasets = {}

    print("Sweeping log chronologically to extract WIP and RP_OC...")
    
    # Global Sweep Loop
    for _, row in df.iterrows():
        case_id = row["case:concept:name"]
        activity_name = row["concept:name"]
        transition = str(row.get("lifecycle:transition", "COMPLETE")).upper()
        timestamp = row["time:timestamp"]
        resource_id = row.get("org:resource", None)
        
        # A. Manage Active Cases (WIP)
        if seen_events[case_id] == 0:
            active_cases.add(case_id)
        seen_events[case_id] += 1
        
        # B. Calculate Features BEFORE state changes
        wip = len(active_cases)
        rp_oc = len(busy_resources) / total_resources
        weekday = timestamp.weekday()
        daytime = (timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second) / 86400.0
        
        #case tracking
        if case_id not in case_prefixes:
            case_prefixes[case_id] = []
        if case_id not in case_start_times:
            case_start_times[case_id] = {}
            
        # C. Process Transition
        if transition == "START":
            if pd.notna(resource_id):
                busy_resources.add(resource_id)
            case_start_times[case_id][activity_name] = timestamp
            
        elif transition == "COMPLETE":
            if pd.notna(resource_id):
                busy_resources.discard(resource_id)
                
            # Calculate duration if there's a START
            if activity_name in case_start_times[case_id]:
                duration = (timestamp - case_start_times[case_id][activity_name]).total_seconds()
                del case_start_times[case_id][activity_name]
            else:
                duration = 0.0
                
            if activity_name not in activity_datasets:
                activity_datasets[activity_name] = []
                
            activity_datasets[activity_name].append({
                "Prefix": list(case_prefixes[case_id]),
                "Weekday": weekday,
                "Daytime": daytime,
                "WIP": wip,
                "RP_OC": rp_oc,
                "Duration_Seconds": duration
            })
            
            # Update prefix AFTER the event completes
            case_prefixes[case_id].append(activity_name)
            
        # D. Remove case from active_cases if this was its last event
        if seen_events[case_id] == case_event_counts[case_id]:
            active_cases.discard(case_id)
            del case_prefixes[case_id]
            del case_start_times[case_id]

    #save the datasets
    print(f"Saving datasets to {output_dir}/...")
    for activity_name, rows in activity_datasets.items():
        out_df = pd.DataFrame(rows)
        safe_name = activity_name.replace(" ", "_").replace("/", "_")
        file_path = os.path.join(output_dir, f"dataset_duration_{safe_name}.csv")
        
        out_df["Prefix"] = out_df["Prefix"].apply(lambda x: ",".join(x))
        out_df.to_csv(file_path, index=False)
        print(f"  -> Saved {file_path} ({len(out_df)} samples)")

    print("\nDataset creation complete.")

if __name__ == "__main__":
    main()
