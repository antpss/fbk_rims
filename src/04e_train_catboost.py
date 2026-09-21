import os
import glob
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

def calculate_smape(actual, predicted):
    numerator = np.abs(predicted - actual)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    smape = np.mean(numerator / (denominator + 1e-8)) * 100.0
    return smape

def main():
    data_dir = "../data/processed/datasets"
    model_dir = "../models/catboost"
    os.makedirs(model_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print("No datasets found in data/processed/datasets/")
        return
        
    MAX_SEQ_LEN = 10
    
    for file in csv_files:
        activity_name = os.path.basename(file).replace("dataset_duration_", "").replace(".csv", "")
        print(f"\n--- Training CatBoost for Activity: {activity_name} ---")
        
        df = pd.read_csv(file)
        
        # Flatten Prefix using pure text strings
        prefix_features = []
        for p_str in df['Prefix']:
            if pd.isna(p_str) or p_str == "":
                seq = []
            else:
                seq = p_str.split(',')
            
            # Truncate or Pad with the word "NONE" instead of 0
            if len(seq) > MAX_SEQ_LEN:
                seq = seq[-MAX_SEQ_LEN:]
            else:
                seq = seq + ["NONE"] * (MAX_SEQ_LEN - len(seq))
            prefix_features.append(seq)
            
        col_names = [f"Act_Minus_{MAX_SEQ_LEN - i}" for i in range(MAX_SEQ_LEN)]
        prefix_df = pd.DataFrame(prefix_features, columns=col_names)
        
        # Continuous features
        df[['Weekday', 'Daytime', 'WIP', 'RP_OC']] = df[['Weekday', 'Daytime', 'WIP', 'RP_OC']].fillna(0.0)
        continuous_df = df[['Weekday', 'Daytime', 'WIP', 'RP_OC']]
        
        X = pd.concat([prefix_df, continuous_df], axis=1)
        y = np.log1p(df['Duration_Seconds'].values)
        
        #telling CatBoost which columns are categorical text
        cat_features = list(range(MAX_SEQ_LEN)) 
        
        model = CatBoostRegressor(
            iterations=200,      
            learning_rate=0.1,
            depth=6,
            cat_features=cat_features,
            verbose=0,           # Silent training
            random_seed=42
        )
        
        model.fit(X, y)
        
        preds_log = model.predict(X)
        
        actual_seconds = np.expm1(y)
        predicted_seconds = np.expm1(preds_log)
        
        mae = mean_absolute_error(actual_seconds, predicted_seconds)
        smape = calculate_smape(actual_seconds, predicted_seconds)
        
        print(f"Final Evaluation | MAE: {mae:.2f}s | SMAPE: {smape:.2f}%")
        
        model_path = os.path.join(model_dir, f"catboost_{activity_name}.cbm")
        model.save_model(model_path)
        print(f"Finished. Model saved to {model_path}")

if __name__ == "__main__":
    main()

