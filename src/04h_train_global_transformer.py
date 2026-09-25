import os
import time
import math
import json
import joblib
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

class GlobalEventLogDataset(Dataset):
    def __init__(self, df, vocab, act_to_idx, max_seq_len, feature_cols, weights=None):
        self.max_seq_len = max_seq_len
        
        self.prefixes = []
        for p_str in df['Prefix']:
            if pd.isna(p_str) or p_str == "":
                seq = []
            else:
                seq = [vocab.get(act, vocab['<UNK>']) for act in p_str.split(',')]
            if len(seq) > max_seq_len:
                seq = seq[-max_seq_len:]
            else:
                seq = seq + [vocab['<PAD>']] * (max_seq_len - len(seq))
            self.prefixes.append(seq)
        self.prefixes = torch.tensor(self.prefixes, dtype=torch.long)
        
        self.target_acts = torch.tensor([act_to_idx[act] for act in df['Target_Activity']], dtype=torch.long)
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.targets = torch.tensor(df['Duration_Seconds'].values, dtype=torch.float32)
        if weights is not None:
            self.weights = torch.tensor(weights, dtype=torch.float32)
        else:
            self.weights = torch.ones(len(self.targets), dtype=torch.float32)
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return self.prefixes[idx], self.target_acts[idx], self.features[idx], self.targets[idx], self.weights[idx]

def calculate_smape(actual, predicted):
    numerator = np.abs(predicted - actual)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    smape = np.mean(numerator / (denominator + 1e-8)) * 100.0
    return smape

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class GlobalDurationTransformer(nn.Module):
    def __init__(self, vocab_size, num_target_acts, embed_size, num_continuous_features, hidden_dim, num_layers=2, nhead=4, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_size, max_len=20)
        self.target_act_embed = nn.Embedding(num_target_acts, embed_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, 
            nhead=nhead, 
            dim_feedforward=hidden_dim, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc1 = nn.Linear(embed_size * 2 + num_continuous_features, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, prefix, target_act, continuous_features):
        padding_mask = (prefix == 0)
        
        x = self.embed(prefix)
        x = self.pos_encoder(x)
        trans_out = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Mean pooling over non-padded elements
        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        sum_embeddings = torch.sum(trans_out * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        seq_repr = sum_embeddings / sum_mask
        
        act_emb = self.target_act_embed(target_act)
        
        combined = torch.cat((seq_repr, act_emb, continuous_features), dim=1)
        h = self.relu(self.fc1(combined))
        h = self.dropout(h)
        out = self.fc2(h)
        return out.squeeze(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unweighted", action="store_true", help="Train without sample weights")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.normpath(os.path.join(script_dir, "../data/processed/datasets_no_zeros/dataset_duration_global.csv"))
    model_dir = os.path.normpath(os.path.join(script_dir, "../models_no_zeros/global"))
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}. Run 03c_dataset_creation_global.py first.")
        return

    print(f"Loading global dataset: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Total samples: {len(df)}")

    # Vocabularies
    unique_target_acts = sorted(df['Target_Activity'].unique().tolist())
    act_to_idx = {act: idx for idx, act in enumerate(unique_target_acts)}
    
    all_prefix_acts = set(unique_target_acts)
    for p_str in df['Prefix']:
        if pd.notna(p_str) and p_str != "":
            all_prefix_acts.update(p_str.split(','))
            
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for idx, act in enumerate(sorted(all_prefix_acts), start=2):
        vocab[act] = idx
        
    with open(os.path.join(model_dir, "vocab_global.json"), "w") as f:
        json.dump({"vocab": vocab, "act_to_idx": act_to_idx}, f, indent=2)

    feature_cols = [c for c in df.columns if c not in ['Prefix', 'Duration_Seconds', 'Wait_Time_Seconds', 'Resource_ID', 'Sample_Weight', 'Target_Activity']]
    
    # Stratified 80/20 train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Target_Activity'])
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Scale continuous features
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols].fillna(0.0))
    test_df[feature_cols] = scaler.transform(test_df[feature_cols].fillna(0.0))
    joblib.dump(scaler, os.path.join(model_dir, "scaler_global.pkl"))

    weights_train = train_df['Sample_Weight'].values if not args.unweighted else np.ones(len(train_df))
    weights_test = test_df['Sample_Weight'].values

    MAX_SEQ_LEN = 10
    EMBED_SIZE = 32
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    NHEAD = 4
    DROPOUT = 0.2
    EPOCHS = 80
    BATCH_SIZE = 128
    LR = 0.001
    PATIENCE = 7

    train_dataset = GlobalEventLogDataset(train_df, vocab, act_to_idx, MAX_SEQ_LEN, feature_cols, weights_train)
    test_dataset = GlobalEventLogDataset(test_df, vocab, act_to_idx, MAX_SEQ_LEN, feature_cols, weights_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training set: {len(train_dataset)} | Test set: {len(test_dataset)}")
    print(f"Sample weighting: {'ENABLED (Importance + Sqrt Capped)' if not args.unweighted else 'DISABLED'}")

    model = GlobalDurationTransformer(len(vocab), len(act_to_idx), EMBED_SIZE, len(feature_cols), HIDDEN_DIM, NUM_LAYERS, NHEAD, DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_loss = float('inf')
    patience_counter = 0
    model_path = os.path.join(model_dir, "transformer_global.pt")

    print("\nTraining Global Transformer model...")
    start_t = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for prefixes, target_acts, features, targets, weights in train_loader:
            optimizer.zero_grad()
            outputs = model(prefixes, target_acts, features)
            unweighted_loss = torch.abs(outputs - targets)
            loss = (unweighted_loss * weights).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Weighted Train MAE: {avg_loss:.2f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}. Best loss: {best_loss:.2f}s")
            break

    end_t = time.time()
    print(f"Training time: {end_t - start_t:.2f} seconds")

    # Evaluation
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    all_preds, all_actuals = [], []
    with torch.no_grad():
        for prefixes, target_acts, features, targets, _ in test_loader:
            preds = model(prefixes, target_acts, features)
            all_preds.extend(preds.numpy())
            all_actuals.extend(targets.numpy())

    all_preds = np.maximum(0, np.array(all_preds))
    all_actuals = np.array(all_actuals)

    test_df_eval = pd.DataFrame({
        'Target_Activity': test_df['Target_Activity'],
        'Actual': all_actuals,
        'Predicted': all_preds
    })

    overall_mae = mean_absolute_error(all_actuals, all_preds)
    overall_smape = calculate_smape(all_actuals, all_preds)

    print("\n" + "=" * 75)
    print("GLOBAL TRANSFORMER TEST EVALUATION")
    print("=" * 75)
    print(f"Overall Test Set (All Activities) | MAE: {overall_mae:.2f}s | SMAPE: {overall_smape:.2f}%")
    print("-" * 75)
    print(f"{'Activity':32s} | {'Count':6s} | {'MAE (s)':10s} | {'SMAPE (%)':10s}")
    print("-" * 75)

    per_act_results = []
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
    print(f"\nModel saved to: {model_path}")

if __name__ == "__main__":
    main()
