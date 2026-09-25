import os
import glob
import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import argparse
import time

def calculate_smape(actual, predicted):
    numerator = np.abs(predicted - actual)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    smape = np.mean(numerator / (denominator + 1e-8)) * 100.0
    return smape

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_zeros", action="store_true", help="Use dataset without zero durations")
    args = parser.parse_args()

    if args.no_zeros:
        data_dir = "../data/processed/datasets_no_zeros"
        model_dir = "../models_no_zeros/xgboost"
    else:
        data_dir = "../data/processed/datasets"
        model_dir = "../models/xgboost"
        
    os.makedirs(model_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(data_dir, "dataset_duration_*.csv"))
    if not csv_files:
        print("No datasets found in data/processed/datasets/")
        return
        
    MAX_SEQ_LEN = 10
    
    for file in csv_files:
        activity_name = os.path.basename(file).replace("dataset_duration_", "").replace(".csv", "")
        print(f"\n--- Training XGBoost for Activity: {activity_name} ---")
        
        df = pd.read_csv(file)
        
        # Flatten Prefix into 10 separate columns (Act_Minus_10 to Act_Minus_1)
        prefix_features = []
        for p_str in df['Prefix']:
            if pd.isna(p_str) or p_str == "":
                seq = []
            else:
                seq = p_str.split(',')
            
            # Truncate or Pad with string "NONE"
            if len(seq) > MAX_SEQ_LEN:
                seq = seq[-MAX_SEQ_LEN:]
            else:
                seq = seq + ["NONE"] * (MAX_SEQ_LEN - len(seq))
            prefix_features.append(seq)
            
        # Create columns: Act_Minus_10 is the oldest, Act_Minus_1 is the most recent activity
        prefix_df = pd.DataFrame(prefix_features, columns=[f"Act_Minus_{MAX_SEQ_LEN - i}" for i in range(MAX_SEQ_LEN)])
        
        # Convert prefix columns to pandas categorical dtype
        for col in prefix_df.columns:
            prefix_df[col] = prefix_df[col].astype('category')
        
        # Continuous and other features
        df['Resource_ID'] = df['Resource_ID'].astype('category')
        feature_cols = [c for c in df.columns if c not in ['Prefix', 'Duration_Seconds', 'Wait_Time_Seconds']]
        feature_cols = [c for c in df.columns if c not in ['Prefix', 'Duration_Seconds', 'Wait_Time_Seconds', 'Resource_ID']]
        df[feature_cols] = df[feature_cols].fillna(0.0)
        X_features = df[feature_cols]
        
        # Combine flattened sequence and continuous features
        X = pd.concat([prefix_df, X_features], axis=1)
        
        # Target (LOG TRANSFORMED)
        y = np.log1p(df['Duration_Seconds'].values)
        
        # 80/20 train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
        
        # Train XGBoost
        model = xgb.XGBRegressor(
            n_estimators=300,      # Increased to 300 Trees
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            enable_categorical=True,
            tree_method='hist',
            early_stopping_rounds=10
        )
        
        start_t = time.time()
        
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        end_t = time.time()
        print(f"Training time: {end_t - start_t:.2f} seconds")
        
        # Predict & calculate metrics
        preds_log = model.predict(X_test)
        
        # Convert back to actual seconds for human-readable metrics
        actual_seconds = np.expm1(y_test)
        predicted_seconds = np.expm1(preds_log)
        
        mae = mean_absolute_error(actual_seconds, predicted_seconds)
        smape = calculate_smape(actual_seconds, predicted_seconds)
        
        print(f"Final Evaluation | MAE: {mae:.2f}s | SMAPE: {smape:.2f}%")
        
        # Save model
        model_path = os.path.join(model_dir, f"xgboost_{activity_name}.json")
        model.save_model(model_path)
        print(f"Finished. Model saved to {model_path}")

if __name__ == "__main__":
    main()

