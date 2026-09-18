import os
import glob
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#data preprocessing for PyTorch
class EventLogDataset(Dataset):
    def __init__(self, df, vocab, max_seq_len):
        self.max_seq_len = max_seq_len
        
        #parse and encode the categorical sequence (Prefix)
        self.prefixes = []
        for p_str in df['Prefix']:
            if pd.isna(p_str) or p_str == "":
                seq = []
            else:
                seq = [vocab.get(act, vocab['<UNK>']) for act in p_str.split(',')]
                
            # Pad (with 0s) or truncate sequences to ensure uniform tensor shapes
            if len(seq) > max_seq_len:
                seq = seq[-max_seq_len:]
            else:
                seq = seq + [vocab['<PAD>']] * (max_seq_len - len(seq))
            self.prefixes.append(seq)
            
        self.prefixes = torch.tensor(self.prefixes, dtype=torch.long)
        
        # Load the continuous features (Weekday, Daytime, WIP, RP_OC)
        features = df[['Weekday', 'Daytime', 'WIP', 'RP_OC']].values
        self.features = torch.tensor(features, dtype=torch.float32)
        
        # Load the Target variable (Processing Time)
        targets = df['Duration_Seconds'].values
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return self.prefixes[idx], self.features[idx], self.targets[idx]

#Transformer Architecture
class DurationTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_continuous_features, num_heads, hidden_dim, num_layers, max_seq_len):
        super().__init__()
        
        # Embedding layer for the sequence of activities
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, embed_size))
        
        # Core PyTorch Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Final Neural Network layers (Regression head)
        self.fc1 = nn.Linear(embed_size + num_continuous_features, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1) # Outputting 1 continuous value (Duration)

    def forward(self, prefix, continuous_features):
        #process the sequence through the transformer
        x = self.embed(prefix) + self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.mean(dim=1) # Mean pooling to condense the sequence into a single vector
        
        #concatenate the transformer output with continuous features
        x = torch.cat([x, continuous_features], dim=1)
        
        #predict the Duration
        x = self.relu(self.fc1(x))
        out = self.fc2(x)
        return out.squeeze(1)

#evaluation metric helper
def calculate_smape(actual, predicted):
    numerator = torch.abs(predicted - actual)
    denominator = (torch.abs(actual) + torch.abs(predicted)) / 2.0
    smape = torch.mean(numerator / (denominator + 1e-8)) * 100.0
    return smape.item()

def main():
    data_dir = "../data/processed/datasets"
    model_dir = "../models/transformers"
    os.makedirs(model_dir, exist_ok=True)
    
    #security check
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
    NUM_HEADS = 4
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    EPOCHS = 50
    BATCH_SIZE = 64
    LR = 0.001
    PATIENCE = 5
    
    # Train ONE model per activity
    for file in csv_files:
        activity_name = os.path.basename(file).replace("dataset_duration_", "").replace(".csv", "")
        print(f"\n--- Training Transformer for Activity: {activity_name} ---")
        
        df = pd.read_csv(file)
        
        # Initialize Dataset and DataLoader
        dataset = EventLogDataset(df, vocab, MAX_SEQ_LEN)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Initialize Model, Loss Function (MAE), and Optimizer
        model = DurationTransformer(len(vocab), EMBED_SIZE, NUM_CONTINUOUS, NUM_HEADS, HIDDEN_DIM, NUM_LAYERS, MAX_SEQ_LEN)
        criterion = nn.L1Loss() #MAE
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        
        # Training Loop
        best_loss = float('inf')
        patience_counter = 0
        model_path = os.path.join(model_dir, f"transformer_{activity_name}.pt")
        
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
                
                total_loss += loss.item()
                total_smape += calculate_smape(targets, outputs)
                
            avg_loss = total_loss / len(dataloader)
            avg_smape = total_smape / len(dataloader)
            
            #print progress
            if (epoch+1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:02d}/{EPOCHS} | MAE: {avg_loss:.2f}s | SMAPE: {avg_smape:.2f}%")
                
            # early stopping logic
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                #save the best model dynamically
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}. Best MAE was {best_loss:.2f}s")
                break
                
        print(f"Finished. Best model saved to {model_path}")

if __name__ == "__main__":
    main()
