import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

def calculate_smape(actual, predicted):
    numerator = np.abs(predicted - actual)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    smape = np.mean(numerator / (denominator + 1e-8)) * 100.0
    return smape

def main():
    data_dir = "../data/processed/datasets"
    model_dir = "../models/xgboost"
    os.makedirs(model_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print("No datasets found in data/processed/datasets/")
        return
        
    print("Building global vocabulary...")
    all_activities = set()
    for file in csv_files:
        df = pd.read_csv(file)
        for p_str in df['Prefix']:
            if pd.notna(p_str) and p_str != "":
                all_activities.update(p_str.split(','))
                
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for idx, act in enumerate(sorted(all_activities), start=2):
        vocab[act] = idx
        
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
                seq = [vocab.get(act, vocab['<UNK>']) for act in p_str.split(',')]
            
            # Truncate or Pad
            if len(seq) > MAX_SEQ_LEN:
                seq = seq[-MAX_SEQ_LEN:]
            else:
                seq = seq + [vocab['<PAD>']] * (MAX_SEQ_LEN - len(seq))
            prefix_features.append(seq)
            
        # Create columns: Act_Minus_10 is the oldest, Act_Minus_1 is the most recent activity
        prefix_df = pd.DataFrame(prefix_features, columns=[f"Act_Minus_{MAX_SEQ_LEN - i}" for i in range(MAX_SEQ_LEN)])
        
        # Continuous features
        df[['Weekday', 'Daytime', 'WIP', 'RP_OC']] = df[['Weekday', 'Daytime', 'WIP', 'RP_OC']].fillna(0.0)
        continuous_df = df[['Weekday', 'Daytime', 'WIP', 'RP_OC']]
        
        # Combine flattened sequence and continuous features
        X = pd.concat([prefix_df, continuous_df], axis=1)
        
        # Target (LOG TRANSFORMED)
        y = np.log1p(df['Duration_Seconds'].values)
        
        # Train XGBoost
        model = xgb.XGBRegressor(
            n_estimators=100,      # 100 Trees
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        model.fit(X, y)
        
        # Predict & calculate metrics
        preds_log = model.predict(X)
        
        # Convert back to actual seconds for human-readable metrics
        actual_seconds = np.expm1(y)
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

