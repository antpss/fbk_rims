import os
import glob
import time
import argparse
import json
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

class EventLogDataset(Dataset):
    def __init__(self, df, vocab, max_seq_len, feature_cols):
        self.max_seq_len = max_seq_len
        self.feature_cols = feature_cols
        
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
        
        df[feature_cols] = df[feature_cols].fillna(0.0)
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.targets = torch.tensor(df['Duration_Seconds'].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return self.prefixes[idx], self.features[idx], self.targets[idx]

def calculate_smape(actual, predicted):
    numerator = torch.abs(predicted - actual)
    denominator = (torch.abs(actual) + torch.abs(predicted)) / 2.0
    smape = torch.mean(numerator / (denominator + 1e-8)) * 100.0
    return smape.item()

# Causal Convolution Block for TCN
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        # Padding for causal convolution: (kernel_size - 1) * dilation
        padding = (kernel_size - 1) * dilation
        
        # 1D Convolution over the time dimension
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            padding=padding, dilation=dilation
        )
        
        self.relu = nn.ReLU()
        
        # 1x1 Convolution for residual connection if channel dimension changes
        if in_channels != out_channels:
            self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.res_conv = None

    def forward(self, x):
        # x is (batch, channels, time)
        out = self.conv(x)
        
        # Remove the extra padded output on the right to maintain causal property
        # out has size (batch, channels, time + padding) -> keep (batch, channels, time)
        out = out[:, :, :-self.conv.padding[0]]
        
        out = self.relu(out)
        
        # Residual connection
        res = x if self.res_conv is None else self.res_conv(x)
        return self.relu(out + res)

class DurationTCN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_continuous_features, hidden_dim, num_layers, max_seq_len, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        layers = []
        in_channels = embed_size
        for i in range(num_layers):
            dilation_size = 2 ** i
            out_channels = hidden_dim
            layers += [TCNBlock(in_channels, out_channels, kernel_size=2, dilation=dilation_size)]
            in_channels = out_channels
        self.tcn = nn.Sequential(*layers)
        
        self.fc1 = nn.Linear(hidden_dim + num_continuous_features, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1) 

    def forward(self, prefix, continuous_features):
        x = self.embed(prefix) 
        x = x.transpose(1, 2)  
        x = self.tcn(x) 
        tcn_out = x[:, :, -1]
        
        x = torch.cat((tcn_out, continuous_features), dim=1)
        x = self.relu3(self.fc1(x))
        out = self.fc2(x)
        return out.squeeze(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_zeros", action="store_true")
    args = parser.parse_args()

    if args.no_zeros:
        data_dir = "../data/processed/datasets_no_zeros"
        model_dir = "../models_no_zeros/tcns"
    else:
        data_dir = "../data/processed/datasets"
        model_dir = "../models/tcns"
        
    os.makedirs(model_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(data_dir, "dataset_duration_*.csv"))
    if not csv_files: return
        
    all_activities = set()
    for file in csv_files:
        df = pd.read_csv(file)
        for p_str in df['Prefix']:
            if pd.notna(p_str) and p_str != "":
                all_activities.update(p_str.split(','))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for idx, act in enumerate(sorted(all_activities), start=2): vocab[act] = idx
    
    with open(os.path.join(model_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved vocabulary to {os.path.join(model_dir, 'vocab.json')} ({len(vocab)} tokens)")
    MAX_SEQ_LEN = 10
    EMBED_SIZE = 32
    HIDDEN_DIM = 128
    NUM_LAYERS = 3
    DROPOUT = 0.2
    EPOCHS = 100
    BATCH_SIZE = 64
    LR = 0.001
    PATIENCE = 7
    
    for file in csv_files:
        activity_name = os.path.basename(file).replace("dataset_duration_", "").replace(".csv", "")
        print(f"\n--- Training TCN for Activity: {activity_name} ---")
        
        df = pd.read_csv(file)
        feature_cols = [c for c in df.columns if c not in ['Prefix', 'Duration_Seconds', 'Wait_Time_Seconds', 'Resource_ID']]
        
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols].fillna(0.0))
        joblib.dump(scaler, os.path.join(model_dir, f"scaler_{activity_name}.pkl"))
        
        dataset = EventLogDataset(df, vocab, MAX_SEQ_LEN, feature_cols)
        
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print(f"Training on {train_size} samples, testing on {test_size} samples.")
        
        model = DurationTCN(len(vocab), EMBED_SIZE, len(feature_cols), HIDDEN_DIM, NUM_LAYERS, MAX_SEQ_LEN, DROPOUT)
        criterion = nn.L1Loss() 
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        
        best_loss = float('inf')
        patience_counter = 0
        model_path = os.path.join(model_dir, f"tcn_{activity_name}.pt")
        
        start_t = time.time()
        for epoch in range(EPOCHS):
            model.train()
            total_loss, total_smape = 0, 0
            for prefixes, features, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(prefixes, features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_smape += calculate_smape(targets, outputs)
                
            avg_loss = total_loss / len(train_loader)
            avg_smape = total_smape / len(train_loader)
            if (epoch+1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:02d}/{EPOCHS} | MAE: {avg_loss:.2f}s | SMAPE: {avg_smape:.2f}%")
                
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
            if patience_counter >= PATIENCE: break
        
        end_t = time.time()
        print(f"Finished. Training time: {end_t - start_t:.2f} seconds")
        
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        test_loss, test_smape = 0, 0
        with torch.no_grad():
            for prefixes, features, targets in test_loader:
                outputs = model(prefixes, features)
                test_loss += criterion(outputs, targets).item()
                test_smape += calculate_smape(targets, outputs)
        print(f"Test Set | MAE: {test_loss / len(test_loader):.2f}s | SMAPE: {test_smape / len(test_loader):.2f}%")

if __name__ == "__main__":
    main()
