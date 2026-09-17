import os
import pandas as pd
import pm4py

def main():
    input_log_path = "../data/processed/BPI_2012_W_only.xes"
    output_dir = "../data/processed"
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading log: {input_log_path}")
    df = pm4py.read_xes(input_log_path)
    
    #sort data by trace ID and time
    df = df.sort_values(by=["case:concept:name", "time:timestamp"])
    
    #list of rows for each activity
    activity_datasets = {}

    print("Extracting prefixes and durations...")
    
    #group the data by case (trace)
    for case_id, group in df.groupby("case:concept:name"):
        prefix = []           # Keeps track of the sequence of activities that have happened so far
        start_times = {}      # Remembers when an activity was started (for duration math)
        
        #loop over every event inside this specific trace
        for _, row in group.iterrows():
            activity_name = row["concept:name"]
            
            #get the lifecycle transition (START, COMPLETE). If missing, assume COMPLETE.
            transition = str(row.get("lifecycle:transition", "COMPLETE")).upper()
            timestamp = row["time:timestamp"]
            
            if transition == "START":
                #remember when this specific activity started
                start_times[activity_name] = timestamp
                
            elif transition == "COMPLETE":
                #calculate duration (Complete Time - Start Time)
                if activity_name in start_times:
                    duration = (timestamp - start_times[activity_name]).total_seconds()
                    del start_times[activity_name] 
                else:
                    duration = 0.0

                # Calculate specific RIMS engine features
                weekday = timestamp.weekday()
                daytime = (timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second) / 86400.0

                if activity_name not in activity_datasets:
                    activity_datasets[activity_name] = []
                
                # Append the row: Input = the prefix, Target = the duration
                activity_datasets[activity_name].append({
                    "Prefix": list(prefix),
                    "Weekday": weekday,
                    "Daytime": daytime,
                    "Duration_Seconds": duration
                })
                
                # Finally, add this activity to the prefix so the NEXT event knows this one happened
                prefix.append(activity_name)

    #save the datasets to csv
    print(f"Saving datasets to {output_dir}/...")
    for activity_name, rows in activity_datasets.items():
        out_df = pd.DataFrame(rows)
        
        #clean the activity name so it's safe to use as a file name
        safe_name = activity_name.replace(" ", "_").replace("/", "_")
        file_path = os.path.join(output_dir, f"dataset_duration_{safe_name}.csv")
        
        #since prefixes are lists (e.g. ["A", "B"]), we convert them to a simple string "A,B"
        out_df["Prefix"] = out_df["Prefix"].apply(lambda x: ",".join(x))
        
        out_df.to_csv(file_path, index=False)
        print(f"  -> Saved {file_path} ({len(out_df)} samples)")

    print("\nDataset creation complete. We now have one training file per activity.")

if __name__ == "__main__":
    main()

