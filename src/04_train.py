import os
import sys
import json
import time
import argparse
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config_loader import load_config, PROJECT_ROOT

# =====================================================================
# Prefix Parsing Helper (supports both JSON arrays and legacy comma strings)
# =====================================================================
def parse_prefix(p):
    if pd.isna(p) or p == "" or p == "[]":
        return []
    if isinstance(p, list):
        return p
    p_str = str(p).strip()
    if p_str.startswith("[") and p_str.endswith("]"):
        try:
            return json.loads(p_str)
        except Exception:
            pass
    return [x.strip() for x in p_str.split(",") if x.strip()]

# =====================================================================
# Evaluation Metrics
# =====================================================================
def calculate_smape(actual, predicted):
    numerator = np.abs(predicted - actual)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    return np.mean(numerator / (denominator + 1e-8)) * 100.0

def format_time_duration(seconds: float) -> str:
    
    #e.g. 81,252s (22.6h), 440s (7.3m), or 25.65s.
    if seconds is None or np.isnan(seconds) or np.isinf(seconds):
        return "N/A"
    sec_int = int(round(seconds))
    if seconds >= 86400:
        days = seconds / 86400.0
        return f"{sec_int:,}s ({days:.1f}d)"
    elif seconds >= 3600:
        hours = seconds / 3600.0
        return f"{sec_int:,}s ({hours:.1f}h)"
    elif seconds >= 60:
        mins = seconds / 60.0
        return f"{sec_int:,}s ({mins:.1f}m)"
    else:
        return f"{seconds:.2f}s"

# =====================================================================
# PyTorch Deep Learning Architectures
# =====================================================================
class EventLogDataset(Dataset):
    def __init__(self, df, vocab, act_to_idx, max_seq_len, feature_cols, weights):
        self.max_seq_len = max_seq_len
        self.features = df[feature_cols].values.astype(np.float32)
        self.weights = weights.astype(np.float32)
        self.durations = df['Target_Duration'].values.astype(np.float32)
        
        if 'Target_Activity' in df.columns and act_to_idx is not None:
            self.target_acts = np.array([act_to_idx.get(act, 0) for act in df['Target_Activity']], dtype=np.int64)
        else:
            self.target_acts = np.zeros(len(df), dtype=np.int64)
            
        prefixes = []
        for p_str in df['Prefix']:
            act_list = parse_prefix(p_str)
            tokens = [vocab.get(a, vocab.get('<UNK>', 1)) for a in act_list]
            if len(tokens) > max_seq_len:
                tokens = tokens[-max_seq_len:]
            else:
                tokens = [vocab.get('<PAD>', 0)] * (max_seq_len - len(tokens)) + tokens
            prefixes.append(tokens)
        self.prefixes = np.array(prefixes, dtype=np.int64)

    def __len__(self):
        return len(self.durations)

    def __getitem__(self, idx):
        return (torch.tensor(self.prefixes[idx], dtype=torch.long),
                torch.tensor(self.target_acts[idx], dtype=torch.long),
                torch.tensor(self.features[idx], dtype=torch.float32),
                torch.tensor(self.durations[idx], dtype=torch.float32),
                torch.tensor(self.weights[idx], dtype=torch.float32))

class DurationLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_acts, num_features, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.act_embedding = nn.Embedding(num_acts, embed_dim) if num_acts > 1 else None
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        in_dim = hidden_dim + num_features + (embed_dim if num_acts > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, prefix_seq, target_act, cont_features):
        emb = self.embedding(prefix_seq)
        _, (h_n, _) = self.lstm(emb)
        last_hidden = h_n[-1]
        tensors = [last_hidden, cont_features]
        if self.act_embedding is not None:
            tensors.append(self.act_embedding(target_act))
        x = torch.cat(tensors, dim=1)
        return F.softplus(self.fc(x).squeeze(-1))

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.dropout1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.dropout2(self.relu2(self.chomp2(self.conv2(out))))
        return self.relu(out + res)

class DurationTCN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_acts, num_features, hidden_dim, num_layers=3, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.act_embedding = nn.Embedding(num_acts, embed_dim) if num_acts > 1 else None
        layers = []
        in_c = embed_dim
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_c, hidden_dim, kernel_size=3, stride=1, dilation=dilation, padding=(3-1)*dilation, dropout=dropout))
            in_c = hidden_dim
        self.tcn = nn.Sequential(*layers)
        in_dim = hidden_dim + num_features + (embed_dim if num_acts > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, prefix_seq, target_act, cont_features):
        emb = self.embedding(prefix_seq).transpose(1, 2)
        tcn_out = self.tcn(emb)[:, :, -1]
        tensors = [tcn_out, cont_features]
        if self.act_embedding is not None:
            tensors.append(self.act_embedding(target_act))
        x = torch.cat(tensors, dim=1)
        return F.softplus(self.fc(x).squeeze(-1))

class DurationTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_acts, num_features, hidden_dim, max_seq_len, num_layers=2, nhead=4, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.act_embedding = nn.Embedding(num_acts, embed_dim) if num_acts > 1 else None
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        in_dim = embed_dim + num_features + (embed_dim if num_acts > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, prefix_seq, target_act, cont_features):
        b_size, seq_len = prefix_seq.shape
        pos = torch.arange(seq_len, device=prefix_seq.device).unsqueeze(0).expand(b_size, -1)
        x = self.embedding(prefix_seq) + self.pos_embedding(pos)
        mask = (prefix_seq == 0)
        trans_out = self.transformer(x, src_key_padding_mask=mask)
        pooled = trans_out[:, -1, :]
        tensors = [pooled, cont_features]
        if self.act_embedding is not None:
            tensors.append(self.act_embedding(target_act))
        x = torch.cat(tensors, dim=1)
        return F.softplus(self.fc(x).squeeze(-1))

# =====================================================================
# Main Unified Training Engine
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified RIMS+ Model Training Engine")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--task", type=str, default="duration", choices=["duration", "waiting_time"], help="Target task to train")
    parser.add_argument("--strategy", type=str, default=None, choices=["global", "local", "hybrid"], help="Override strategy (global, local, hybrid)")
    parser.add_argument("--model_type", "--engine", dest="model_type", type=str, default=None, choices=["xgboost", "lstm", "tcn", "transformer"], help="Override model type")
    parser.add_argument("--unweighted", action="store_true", help="Disable importance sample weights")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task_name = args.task
    task_cfg = cfg["tasks"][task_name]
    strat_cfg = cfg.get("model_strategy", {})

    mode = strat_cfg.get("mode", "hybrid")
    strategy = args.strategy or ("global" if mode in ["global_tournament", "single_global"] else "global")
    model_type = args.model_type or strat_cfg.get("single_global_engine", "xgboost")
    
    # Read DL training parameters from config.yaml
    dl_cfg = strat_cfg.get("dl", {})
    epochs = dl_cfg.get("epochs", 50)
    patience = dl_cfg.get("patience", 5)
    batch_size = dl_cfg.get("batch_size", 256)
    learning_rate = dl_cfg.get("learning_rate", 0.001)
    min_delta = dl_cfg.get("min_delta", 0.01)
    min_samples_local_dl = dl_cfg.get("min_samples_for_local_dl", 3000)

    data_path = os.path.join(cfg["paths"]["output_base_dir"], task_cfg["output_file"])
    target_col = task_cfg["target_column"]
    max_seq_len = cfg.get("features", {}).get("prefix_window_size", 10)
    target_acts = cfg.get("target_activities", [])
    
    models_base_dir = os.path.join(PROJECT_ROOT, cfg["paths"]["models_dir"], task_name, f"{strategy}_{model_type}")
    os.makedirs(models_base_dir, exist_ok=True)

    print(f"============================================================")
    print(f"  RIMS+ UNIFIED TRAINING ENGINE")
    print(f"  Task:        [{task_name.upper()}] (Target: {target_col})")
    print(f"  Strategy:    [{strategy.upper()}]")
    print(f"  Model:       [{model_type.upper()}]")
    print(f"  Output Dir:  {models_base_dir}")
    print(f"  Dataset:     {data_path}")
    print(f"============================================================")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run 03_dataset_builder.py first.")

    df = pd.read_csv(data_path)
    df['Target_Duration'] = df[target_col]
    print(f"Loaded master dataset: {len(df)} samples across {df['Target_Activity'].nunique()} activities.")

    feature_cols = [c for c in df.columns if c not in [
        'Case_ID', 'case_id', 'Prefix', 'Duration_Seconds', 'Wait_Time_Seconds', 'Target_Duration',
        'Resource_ID', 'Sample_Weight', 'Target_Activity'
    ]]
    df[feature_cols] = df[feature_cols].fillna(0.0)

    # -----------------------------------------------------------------
    # STRATEGY DEFINITION: Determine Local vs Global Allocation
    # -----------------------------------------------------------------
    counts = df['Target_Activity'].value_counts().to_dict()

    local_targets = []
    global_targets = []

    if strategy == "global":
        global_targets = sorted(counts.keys())
        print(f"\n--- Strategy Allocation: GLOBAL (All {len(global_targets)} activities pooled) ---")
        print(f"  Target Activities: {global_targets}")
    elif strategy == "local":
        local_targets = sorted(counts.keys())
        print(f"\n--- Strategy Allocation: LOCAL ({len(local_targets)} individual activity models) ---")
        print(f"  Target Activities: {local_targets}")
    elif strategy == "hybrid":
        threshold = strat_cfg.get("hybrid", {}).get("local_threshold_count", 3000)
        for act, cnt in counts.items():
            if cnt >= threshold:
                local_targets.append(act)
            else:
                global_targets.append(act)
        local_targets = sorted(local_targets)
        global_targets = sorted(global_targets)
        print(f"\n--- Strategy Allocation: HEURISTIC HYBRID (Threshold = {threshold} samples) ---")
        print(f"  Local Models  ({len(local_targets)}): {local_targets}")
        print(f"  Global Models ({len(global_targets)}): {global_targets}")

    # Case-level 70/10/20 train/val/test split to prevent cross-event data leakage
    seed = cfg["project"]["random_seed"]
    case_col = 'Case_ID' if 'Case_ID' in df.columns else ('case_id' if 'case_id' in df.columns else None)

    if case_col is not None:
        unique_cases = df[case_col].unique()
        train_val_cases, test_cases = train_test_split(unique_cases, test_size=0.20, random_state=seed)
        train_cases, val_cases = train_test_split(train_val_cases, test_size=0.125, random_state=seed) # 0.125 * 0.8 = 0.10

        train_df = df[df[case_col].isin(train_cases)].reset_index(drop=True)
        val_df = df[df[case_col].isin(val_cases)].reset_index(drop=True)
        test_df = df[df[case_col].isin(test_cases)].reset_index(drop=True)
        print(f"\n[Case-Level 70/10/20 Split] Unique cases: {len(unique_cases)}")
        print(f"  Train: {len(train_cases)} cases ({len(train_df)} events, {len(train_df)/len(df)*100:.1f}%)")
        print(f"  Val:   {len(val_cases)} cases ({len(val_df)} events, {len(val_df)/len(df)*100:.1f}%)")
        print(f"  Test:  {len(test_cases)} cases ({len(test_df)} events, {len(test_df)/len(df)*100:.1f}%)")
    else:
        train_val_df, test_df = train_test_split(df, test_size=0.20, random_state=seed, stratify=df['Target_Activity'])
        train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=seed, stratify=train_val_df['Target_Activity'])
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        print(f"\n[Stratified Event-Level 70/10/20 Split]")
        print(f"  Train: {len(train_df)} events ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  Val:   {len(val_df)} events ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  Test:  {len(test_df)} events ({len(test_df)/len(df)*100:.1f}%)")

    test_predictions = pd.DataFrame({
        'Target_Activity': test_df['Target_Activity'].values,
        'Actual': test_df['Target_Duration'].values,
        'Predicted': np.zeros(len(test_df)),
        'Source_Model': [''] * len(test_df)
    })

    dispatch_table = {
        "strategy": strategy,
        "model_type": model_type,
        "task": task_name,
        "target_column": target_col,
        "dispatch": {}
    }

    start_train_time = time.time()

    # =================================================================
    # EXECUTION: XGBOOST ENGINE
    # =================================================================
    if model_type == "xgboost":
        all_acts_in_data = set(df['Target_Activity'].unique())
        for p in df['Prefix']:
            all_acts_in_data.update(parse_prefix(p))
        act_categories = sorted(list(all_acts_in_data))
        prefix_cat_dtype = pd.CategoricalDtype(categories=act_categories + ["NONE"], ordered=False)
        target_act_dtype = pd.CategoricalDtype(categories=act_categories, ordered=False)

        def build_xgb_features(data_subset):
            p_features = []
            for p_str in data_subset['Prefix']:
                seq = parse_prefix(p_str)
                seq = seq[-max_seq_len:] if len(seq) > max_seq_len else seq + ["NONE"] * (max_seq_len - len(seq))
                p_features.append(seq)
            p_df = pd.DataFrame(p_features, columns=[f"Act_Minus_{max_seq_len - i}" for i in range(max_seq_len)], index=data_subset.index)
            for c in p_df.columns:
                p_df[c] = p_df[c].astype(prefix_cat_dtype)
            act_cat = data_subset[['Target_Activity']].astype(target_act_dtype)
            return pd.concat([p_df, act_cat, data_subset[feature_cols]], axis=1)

        # 1. Train Global Model
        if len(global_targets) > 0 or strategy in ["global", "hybrid"]:
            print(f"\nTraining Global XGBoost model (for {global_targets})...")
            X_train = build_xgb_features(train_df)
            y_train = np.log1p(train_df['Target_Duration'].values)
            w_train = train_df['Sample_Weight'].values if not args.unweighted else np.ones(len(train_df))

            X_val_all = build_xgb_features(val_df)
            y_val_all = np.log1p(val_df['Target_Duration'].values)

            X_test_all = build_xgb_features(test_df)
            y_test_all = np.log1p(test_df['Target_Duration'].values)

            global_xgb = xgb.XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                enable_categorical=True,
                early_stopping_rounds=20,
                eval_metric="mae",
                n_jobs=-1
            )
            eval_set = [(X_val_all, y_val_all)] if len(X_val_all) > 0 else [(X_train, y_train)]
            global_xgb.fit(X_train, y_train, sample_weight=w_train, eval_set=eval_set, verbose=False)
            
            global_model_path = os.path.join(models_base_dir, f"xgboost_global.json")
            global_xgb.save_model(global_model_path)
            dispatch_table["dispatch"]["__default__"] = {"model_type": "xgboost", "file": os.path.basename(global_model_path), "log1p_target": True}

            for act in global_targets:
                act_mask = (test_df['Target_Activity'] == act)
                if act_mask.sum() > 0:
                    X_act = X_test_all[act_mask]
                    preds = np.expm1(global_xgb.predict(X_act))
                    test_predictions.loc[act_mask, 'Predicted'] = preds
                    test_predictions.loc[act_mask, 'Source_Model'] = 'Global XGBoost'
                dispatch_table["dispatch"][act] = {"model_type": "xgboost", "file": os.path.basename(global_model_path), "log1p_target": True}

        # 2. Train Local Models
        for act in local_targets:
            safe_name = act.replace(" ", "_")
            print(f"\nTraining Local XGBoost model for: {act}...")
            train_sub = train_df[train_df['Target_Activity'] == act]
            val_sub = val_df[val_df['Target_Activity'] == act]
            test_sub = test_df[test_df['Target_Activity'] == act]

            X_tr = build_xgb_features(train_sub)
            y_tr = np.log1p(train_sub['Target_Duration'].values)
            X_va = build_xgb_features(val_sub) if len(val_sub) > 0 else None
            y_va = np.log1p(val_sub['Target_Duration'].values) if len(val_sub) > 0 else None
            X_te = build_xgb_features(test_sub)
            y_te = np.log1p(test_sub['Target_Duration'].values)

            loc_xgb = xgb.XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                enable_categorical=True,
                early_stopping_rounds=20,
                eval_metric="mae",
                n_jobs=-1
            )
            eval_set = [(X_va, y_va)] if (X_va is not None and len(X_va) > 0) else [(X_tr, y_tr)]
            loc_xgb.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)

            loc_model_path = os.path.join(models_base_dir, f"xgboost_local_{safe_name}.json")
            loc_xgb.save_model(loc_model_path)
            dispatch_table["dispatch"][act] = {"model_type": "xgboost", "file": os.path.basename(loc_model_path), "log1p_target": True}

            preds = np.expm1(loc_xgb.predict(X_te))
            test_predictions.loc[test_df['Target_Activity'] == act, 'Predicted'] = preds
            test_predictions.loc[test_df['Target_Activity'] == act, 'Source_Model'] = f'Local XGBoost'

    # =================================================================
    # EXECUTION: PYTORCH DEEP LEARNING (LSTM / TCN / Transformer)
    # =================================================================
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using compute device: {device}")

        all_prefix_acts = set()
        for p_str in df['Prefix']:
            all_prefix_acts.update(parse_prefix(p_str))
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for idx, act in enumerate(sorted(all_prefix_acts), start=2):
            vocab[act] = idx
        act_to_idx = {act: i for i, act in enumerate(sorted(counts.keys()))}

        scaler = StandardScaler()
        train_df_scaled = train_df.copy()
        val_df_scaled = val_df.copy()
        test_df_scaled = test_df.copy()
        train_df_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        val_df_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
        test_df_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
        joblib.dump(scaler, os.path.join(models_base_dir, f"scaler_{strategy}.pkl"))
        with open(os.path.join(models_base_dir, "vocab.json"), "w") as f:
            json.dump({"vocab": vocab, "act_to_idx": act_to_idx}, f, indent=2)

        def get_model(num_acts):
            if model_type == "lstm":
                return DurationLSTM(len(vocab), 32, num_acts, len(feature_cols), 128, 2, 0.2)
            elif model_type == "tcn":
                return DurationTCN(len(vocab), 32, num_acts, len(feature_cols), 128, 3, 0.2)
            elif model_type == "transformer":
                return DurationTransformer(len(vocab), 32, num_acts, len(feature_cols), 128, max_seq_len, 2, 4, 0.2)

        def train_nn(model, tr_data, va_data, epochs=50, patience=5, min_delta=0.01):
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.L1Loss(reduction='none')
            tr_loader = DataLoader(tr_data, batch_size=batch_size, shuffle=True)
            va_loader = DataLoader(va_data, batch_size=batch_size, shuffle=False)
            
            best_val = float('inf')
            best_weights = None
            patience_counter = 0

            for epoch in range(epochs):
                model.train()
                for seq, act, feat, dur, w in tr_loader:
                    seq, act, feat, dur, w = seq.to(device), act.to(device), feat.to(device), dur.to(device), w.to(device)
                    optimizer.zero_grad()
                    out = model(seq, act, feat)
                    dur_log = torch.log1p(torch.clamp(dur, min=0.0))
                    loss = (criterion(out, dur_log) * w).mean()
                    loss.backward()
                    optimizer.step()

                model.eval()
                val_losses = []
                with torch.no_grad():
                    for seq, act, feat, dur, _ in va_loader:
                        seq, act, feat, dur = seq.to(device), act.to(device), feat.to(device), dur.to(device)
                        out = model(seq, act, feat)
                        pred_s = torch.expm1(torch.clamp(out, min=0.0, max=25.0))
                        val_losses.append(torch.abs(pred_s - dur).cpu().numpy())
                mean_val = np.mean(np.concatenate(val_losses)) if len(val_losses) > 0 else 0.0

                print(f"  Epoch {epoch+1:3d}/{epochs:3d} - Val MAE: {format_time_duration(mean_val)}", end="")
                
                # Check relative improvement
                if best_val == float('inf'):
                    rel_improvement = 1.0
                elif best_val > 0:
                    rel_improvement = (best_val - mean_val) / best_val
                else:
                    rel_improvement = 0.0

                if rel_improvement >= min_delta:
                    best_val = mean_val
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                    print(f" [Best - improved by {rel_improvement*100:.1f}%]")
                else:
                    patience_counter += 1
                    print(f" [Patience: {patience_counter}/{patience}]")
                    if patience_counter >= patience:
                        print(f"  --> Early stopping triggered at epoch {epoch+1}.")
                        break

            if best_weights is not None:
                model.load_state_dict(best_weights)
            return model

        # 1. Train Global DL Model
        if len(global_targets) > 0 or strategy in ["global", "hybrid"]:
            print(f"\nTraining Global {model_type.upper()} model...")
            tr_ds = EventLogDataset(train_df_scaled, vocab, act_to_idx, max_seq_len, feature_cols, train_df['Sample_Weight'].values if not args.unweighted else np.ones(len(train_df)))
            va_ds = EventLogDataset(val_df_scaled, vocab, act_to_idx, max_seq_len, feature_cols, val_df['Sample_Weight'].values if not args.unweighted else np.ones(len(val_df)))
            te_ds = EventLogDataset(test_df_scaled, vocab, act_to_idx, max_seq_len, feature_cols, test_df['Sample_Weight'].values)
            nn_global = get_model(len(act_to_idx))
            nn_global = train_nn(nn_global, tr_ds, va_ds, epochs=epochs, patience=patience, min_delta=min_delta)
            
            glob_path = os.path.join(models_base_dir, f"{model_type}_global.pth")
            torch.save(nn_global.state_dict(), glob_path)
            dispatch_table["dispatch"]["__default__"] = {"model_type": model_type, "file": os.path.basename(glob_path), "log1p_target": True}

            nn_global.eval()
            with torch.no_grad():
                loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)
                all_preds = np.concatenate([np.expm1(np.clip(nn_global(s.to(device), a.to(device), f.to(device)).cpu().numpy(), 0.0, 25.0)) for s, a, f, _, _ in loader])
            for act in global_targets:
                act_mask = (test_df['Target_Activity'] == act)
                if act_mask.sum() > 0:
                    test_predictions.loc[act_mask, 'Predicted'] = all_preds[act_mask]
                    test_predictions.loc[act_mask, 'Source_Model'] = f'Global {model_type.upper()}'
                dispatch_table["dispatch"][act] = {"model_type": model_type, "file": os.path.basename(glob_path), "log1p_target": True}

        # 2. Train Local DL Models
        for act in local_targets:
            safe_name = act.replace(" ", "_")
            tr_sub = train_df_scaled[train_df_scaled['Target_Activity'] == act]
            va_sub = val_df_scaled[val_df_scaled['Target_Activity'] == act]
            te_sub = test_df_scaled[test_df_scaled['Target_Activity'] == act]

            if len(tr_sub) < min_samples_local_dl:
                print(f"\n[Skip Local {model_type.upper()} for {act}]: {len(tr_sub)} training samples < {min_samples_local_dl} threshold.")
                print(f"  --> Fallback to Global {model_type.upper()} and Local XGBoost.")
                act_mask = (test_df['Target_Activity'] == act)
                if 'all_preds' in locals() and act_mask.sum() > 0:
                    test_predictions.loc[act_mask, 'Predicted'] = all_preds[act_mask]
                    test_predictions.loc[act_mask, 'Source_Model'] = f'Global {model_type.upper()} (Fallback)'
                continue

            print(f"\nTraining Local {model_type.upper()} model for: {act}...")
            tr_ds = EventLogDataset(tr_sub, vocab, None, max_seq_len, feature_cols, np.ones(len(tr_sub)))
            va_ds = EventLogDataset(va_sub, vocab, None, max_seq_len, feature_cols, np.ones(len(va_sub))) if len(va_sub) > 0 else tr_ds
            te_ds = EventLogDataset(te_sub, vocab, None, max_seq_len, feature_cols, np.ones(len(te_sub)))

            nn_loc = get_model(1)
            nn_loc = train_nn(nn_loc, tr_ds, va_ds, epochs=epochs, patience=patience, min_delta=min_delta)
            
            loc_path = os.path.join(models_base_dir, f"{model_type}_local_{safe_name}.pth")
            torch.save(nn_loc.state_dict(), loc_path)
            dispatch_table["dispatch"][act] = {"model_type": model_type, "file": os.path.basename(loc_path), "log1p_target": True}

            nn_loc.eval()
            with torch.no_grad():
                loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)
                act_preds = np.concatenate([np.expm1(np.clip(nn_loc(s.to(device), a.to(device), f.to(device)).cpu().numpy(), 0.0, 25.0)) for s, a, f, _, _ in loader])
            test_predictions.loc[test_df['Target_Activity'] == act, 'Predicted'] = act_preds
            test_predictions.loc[test_df['Target_Activity'] == act, 'Source_Model'] = f'Local {model_type.upper()}'

    total_training_time = time.time() - start_train_time

    # Save Dispatch Table
    dispatch_file = os.path.join(models_base_dir, "dispatch_config.json")
    with open(dispatch_file, "w") as f:
        json.dump(dispatch_table, f, indent=2)

    # =================================================================
    # EVALUATION RESULTS & BENCHMARK TABLE
    # =================================================================
    print("\n" + "=" * 88)
    print(f"  RIMS+ [{strategy.upper()}] STRATEGY EVALUATION ({model_type.upper()})")
    print(f"  Total Training Time: {total_training_time:.2f} seconds")
    print("=" * 88)
    print(f"{'Activity':<32} | {'Count':>5} | {'Model Source':<15} | {'MAE':>18} | {'SMAPE (%)':>9}")
    print("-" * 88)

    act_maes = []
    act_metrics_summary = {}
    for act, sub in test_predictions.groupby('Target_Activity'):
        mae = float(mean_absolute_error(sub['Actual'], sub['Predicted']))
        smape = float(calculate_smape(sub['Actual'], sub['Predicted']))
        src = sub['Source_Model'].iloc[0] if len(sub) > 0 else 'Unknown'
        act_maes.append(mae)
        act_metrics_summary[act] = {
            "count": int(len(sub)),
            "model_source": src,
            "mae": round(mae, 2),
            "smape": round(smape, 2)
        }
        print(f"{act:<32} | {len(sub):>5} | {src:<15} | {format_time_duration(mae):>18} | {smape:>8.2f}%")

    weighted_mae = float(mean_absolute_error(test_predictions['Actual'], test_predictions['Predicted']))
    weighted_smape = float(calculate_smape(test_predictions['Actual'], test_predictions['Predicted']))
    macro_mae = float(np.mean(act_maes)) if len(act_maes) > 0 else 0.0

    print("-" * 88)
    print(f"{'WEIGHTED AVERAGE':<32} | {len(test_predictions):>5} | {'---':<15} | {format_time_duration(weighted_mae):>18} | {weighted_smape:>8.2f}%")
    print(f"{'MACRO AVERAGE':<32} | {len(test_predictions):>5} | {'---':<15} | {format_time_duration(macro_mae):>18} | {'---':>9}")
    print("=" * 88)

    # Save metrics.json for benchmarking
    metrics_summary = {
        "task": task_name,
        "strategy": strategy,
        "model_type": model_type,
        "output_dir": models_base_dir,
        "training_time_seconds": round(total_training_time, 2),
        "weighted_mae": round(weighted_mae, 2),
        "weighted_smape": round(weighted_smape, 2),
        "macro_mae": round(macro_mae, 2),
        "activity_metrics": act_metrics_summary
    }
    metrics_file = os.path.join(models_base_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"Saved deployment dispatch configuration to: {dispatch_file}")
    print(f"Saved evaluation metrics summary to:       {metrics_file}\n")

if __name__ == "__main__":
    main()
