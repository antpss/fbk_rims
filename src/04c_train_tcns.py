import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#data preprocessing for PyTorch, dataset class
class EventLogDataset(Dataset):
    def __init__(self, df, vocab, max_seq_len):
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
        
        df[['Weekday', 'Daytime', 'WIP', 'RP_OC']] = df[['Weekday', 'Daytime', 'WIP', 'RP_OC']].fillna(0.0)
        features = df[['Weekday', 'Daytime', 'WIP', 'RP_OC']].values
        self.features = torch.tensor(features, dtype=torch.float32)
        
        # Load the Target variable (Processing Time in raw seconds)
        targets = df['Duration_Seconds'].values
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return self.prefixes[idx], self.features[idx], self.targets[idx]

def calculate_smape(actual, predicted):
    numerator = torch.abs(predicted - actual)
    denominator = (torch.abs(actual) + torch.abs(predicted)) / 2.0
    smape = torch.mean(numerator / (denominator + 1e-8)) * 100.0
    return smape.item()

#TCN Architecture
class DurationTCN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_continuous_features, hidden_dim, num_layers, max_seq_len):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # 1D Convolution layers
        self.conv1 = nn.Conv1d(in_channels=embed_size, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        # Final Neural Network layers (Regression head)
        self.fc1 = nn.Linear(hidden_dim + num_continuous_features, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, prefix, continuous_features):
        # Embed prefix: shape is (batch, seq_len, embed_size)
        x = self.embed(prefix)
        
        # Transpose for Conv1d: shape must be (batch, embed_size, seq_len)
        x = x.transpose(1, 2)
        
        # Slide the Convolution windows across the sequence
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        
        # Global Max Pooling: Compress the time sequence into one single max vector
        x, _ = torch.max(x, dim=2)
        
        # Concatenate with continuous features
        x = torch.cat([x, continuous_features], dim=1)
        
        # Predict Duration
        x = self.relu3(self.fc1(x))
        out = self.fc2(x)
        return out.squeeze(1)

def main():
    data_dir = "../data/processed/datasets"
    model_dir = "../models/tcns"
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
        
    # Hyperparameters
    MAX_SEQ_LEN = 10
    EMBED_SIZE = 32
    NUM_CONTINUOUS = 4
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    EPOCHS = 50
    BATCH_SIZE = 64
    LR = 0.001
    PATIENCE = 5
    
    for file in csv_files:
        activity_name = os.path.basename(file).replace("dataset_duration_", "").replace(".csv", "")
        print(f"\n--- Training TCN for Activity: {activity_name} ---")
        
        df = pd.read_csv(file)
        
        dataset = EventLogDataset(df, vocab, MAX_SEQ_LEN)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        model = DurationTCN(len(vocab), EMBED_SIZE, NUM_CONTINUOUS, HIDDEN_DIM, NUM_LAYERS, MAX_SEQ_LEN)
        criterion = nn.L1Loss() 
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        
        best_loss = float('inf')
        patience_counter = 0
        model_path = os.path.join(model_dir, f"tcn_{activity_name}.pt")
        
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            total_smape = 0
            
            for prefixes, features, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(prefixes, features)
                
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                # Calculate metrics directly on raw seconds
                total_loss += loss.item()
                total_smape += calculate_smape(targets, outputs)
                
            avg_loss = total_loss / len(dataloader)
            avg_smape = total_smape / len(dataloader)
            
            if (epoch+1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:02d}/{EPOCHS} | MAE: {avg_loss:.2f}s | SMAPE: {avg_smape:.2f}%")
                
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}. Best MAE was {best_loss:.2f}s")
                break
                
        print(f"Finished. Best model saved to {model_path}")

if __name__ == "__main__":
    main()

