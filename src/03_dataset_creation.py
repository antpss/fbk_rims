import os
import pandas as pd
import pm4py

def main():
    input_log_path = "../data/raw/BPI_Challenge_2012.xes"
    output_dir = "../data/processed"
    
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load the log
    print(f"Loading log: {input_log_path}")
    # Note: Because we installed the fast `r4pm` library, this returns a Pandas DataFrame!
    df = pm4py.read_xes(input_log_path)
    
    # Sort the data by trace ID and time so everything is in perfect chronological order
    df = df.sort_values(by=["case:concept:name", "time:timestamp"])
    
    # We will store a list of rows for each activity. 
    activity_datasets = {}

    print("Extracting prefixes and durations...")
    
    # 2. Group the data by case (trace)
    for case_id, group in df.groupby("case:concept:name"):
        prefix = []           # Keeps track of the sequence of activities that have happened so far
        start_times = {}      # Remembers when an activity was started (for duration math)
        
        # 3. Loop over every event inside this specific trace
        for _, row in group.iterrows():
            activity_name = row["concept:name"]
            
            # Get the lifecycle transition (e.g. START, COMPLETE). If missing, assume COMPLETE.
            transition = str(row.get("lifecycle:transition", "COMPLETE")).upper()
            timestamp = row["time:timestamp"]
            
            if transition == "START":
                # Remember when this specific activity started
                start_times[activity_name] = timestamp
                
            elif transition == "COMPLETE":
                # Calculate duration (Complete Time - Start Time)
                if activity_name in start_times:
                    duration = (timestamp - start_times[activity_name]).total_seconds()
                    del start_times[activity_name] 
                else:
                    duration = 0.0

                if activity_name not in activity_datasets:
                    activity_datasets[activity_name] = []
                
                # Append the row: Input = the prefix, Target = the duration
                activity_datasets[activity_name].append({
                    "Prefix": list(prefix),
                    "Duration_Seconds": duration
                })
                
                # Finally, add this activity to the prefix so the NEXT event knows this one happened!
                prefix.append(activity_name)

    # 4. Save the datasets to CSV files
    print(f"Saving datasets to {output_dir}/...")
    for activity_name, rows in activity_datasets.items():
        out_df = pd.DataFrame(rows)
        
        # Clean the activity name so it's safe to use as a file name
        safe_name = activity_name.replace(" ", "_").replace("/", "_")
        file_path = os.path.join(output_dir, f"dataset_duration_{safe_name}.csv")
        
        # Since prefixes are lists (e.g. ["A", "B"]), we convert them to a simple string "A,B"
        out_df["Prefix"] = out_df["Prefix"].apply(lambda x: ",".join(x))
        
        out_df.to_csv(file_path, index=False)
        print(f"  -> Saved {file_path} ({len(out_df)} samples)")

    print("\nDataset creation complete! We now have one training file per activity.")

if __name__ == "__main__":
    main()

