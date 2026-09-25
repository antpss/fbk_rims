import os
import time
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def calculate_smape(actual, predicted):
    numerator = np.abs(predicted - actual)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    smape = np.mean(numerator / (denominator + 1e-8)) * 100.0
    return smape

from config_loader import load_config, PROJECT_ROOT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--unweighted", action="store_true", help="Train without sample weights")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task_cfg = cfg["tasks"]["duration"]
    data_path = os.path.join(cfg["paths"]["output_base_dir"], task_cfg["output_file"])
    model_dir = os.path.join(cfg["paths"]["models_dir"], "global")
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}. Run 03c_dataset_creation_global.py first.")
        return

    print(f"Loading global dataset: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Total samples: {len(df)}")

    MAX_SEQ_LEN = 10
    MAX_SEQ_LEN = cfg.get("features", {}).get("prefix_window_size", 10)
    
    # Flatten Prefix into 10 lag columns
    prefix_features = []
    for p_str in df['Prefix']:
        if pd.isna(p_str) or p_str == "":
            seq = []
        else:
            seq = p_str.split(',')
        if len(seq) > MAX_SEQ_LEN:
            seq = seq[-MAX_SEQ_LEN:]
        else:
            seq = seq + ["NONE"] * (MAX_SEQ_LEN - len(seq))
        prefix_features.append(seq)
        
    prefix_df = pd.DataFrame(prefix_features, columns=[f"Act_Minus_{MAX_SEQ_LEN - i}" for i in range(MAX_SEQ_LEN)])
    for col in prefix_df.columns:
        prefix_df[col] = prefix_df[col].astype('category')
        
    df['Target_Activity'] = df['Target_Activity'].astype('category')
    
    feature_cols = [c for c in df.columns if c not in ['Prefix', 'Duration_Seconds', 'Wait_Time_Seconds', 'Resource_ID', 'Sample_Weight', 'Target_Activity']]
    df[feature_cols] = df[feature_cols].fillna(0.0)
    
    X = pd.concat([prefix_df, df[['Target_Activity']], df[feature_cols]], axis=1)
    y = np.log1p(df['Duration_Seconds'].values)
    weights = df['Sample_Weight'].values if not args.unweighted else np.ones(len(df))

    # Stratified 80/20 train/test split by Target_Activity
    X_train, X_test, y_train, y_test, w_train, w_test, acts_train, acts_test = train_test_split(
        X, y, weights, df['Target_Activity'], test_size=0.2, random_state=42, stratify=df['Target_Activity']
    )
    print(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")
    print(f"Sample weighting: {'ENABLED (Importance + Sqrt Capped)' if not args.unweighted else 'DISABLED'}")

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        enable_categorical=True,
        tree_method='hist',
        early_stopping_rounds=15
    )

    print("\nTraining Global XGBoost model...")
    start_t = time.time()
    model.fit(
        X_train, y_train, 
        sample_weight=w_train if not args.unweighted else None,
        eval_set=[(X_test, y_test)], 
        verbose=False
    )
    end_t = time.time()
    print(f"Training time: {end_t - start_t:.2f} seconds")

    preds_log = model.predict(X_test)
    actual_sec = np.expm1(y_test)
    predicted_sec = np.expm1(preds_log)

    overall_mae = mean_absolute_error(actual_sec, predicted_sec)
    overall_smape = calculate_smape(actual_sec, predicted_sec)
    
    print("\n" + "=" * 75)
    print("GLOBAL XGBOOST TEST EVALUATION")
    print("=" * 75)
    print(f"Overall Test Set (All Activities) | MAE: {overall_mae:.2f}s | SMAPE: {overall_smape:.2f}%")
    print("-" * 75)

    per_act_results = []
    test_df_eval = pd.DataFrame({
        'Target_Activity': acts_test,
        'Actual': actual_sec,
        'Predicted': predicted_sec
    })

    print(f"{'Activity':32s} | {'Count':6s} | {'MAE (s)':10s} | {'SMAPE (%)':10s}")
    print("-" * 75)
    for act, grp in test_df_eval.groupby('Target_Activity'):
        mae_act = mean_absolute_error(grp['Actual'], grp['Predicted'])
        smape_act = calculate_smape(grp['Actual'].values, grp['Predicted'].values)
        per_act_results.append((act, len(grp), mae_act, smape_act))
        print(f"{act:32s} | {len(grp):6d} | {mae_act:10.2f} | {smape_act:9.2f}%")

    weighted_mae = sum(c * m for _, c, m, _ in per_act_results) / len(test_df_eval)
    macro_mae = sum(m for _, _, m, _ in per_act_results) / len(per_act_results)
    print("-" * 75)
    print(f"{'WEIGHTED AVERAGE MAE':32s} | {len(test_df_eval):6d} | {weighted_mae:10.2f} |")
    print(f"{'MACRO AVERAGE MAE':32s} | {len(test_df_eval):6d} | {macro_mae:10.2f} |")
    print("=" * 75)

    model_path = os.path.join(model_dir, "xgboost_global.json")
    model.save_model(model_path)
    print(f"\nModel saved to: {model_path}")

if __name__ == "__main__":
    main()
