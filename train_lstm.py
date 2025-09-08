# train_lstm.py

import torch
import torch.nn as nn
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

# --- Import our custom data pipeline from Phase 2 ---
from prep_data_lstm import VideoSequenceDataset, pad_collate_fn

# --- 1. Define the LSTM Classifier Model ---
class LSTMClassifier(nn.Module):
    """
    An LSTM model that takes a sequence of feature vectors and classifies the sequence.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout_prob=0.5):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # The LSTM layer processes the sequence. batch_first=True is crucial.
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)

        # A fully connected layer to map the LSTM's final output to a class score
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x has shape: (batch_size, seq_length, input_dim)

        # Initialize hidden and cell states for the LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Pass the sequence through the LSTM
        # We only need the output, not the final hidden/cell states
        out, _ = self.lstm(x, (h0, c0))

        # We only care about the output from the VERY LAST time step of the sequence
        # out[:, -1, :] gives us the hidden state of the last element in the sequence
        last_time_step_out = out[:, -1, :]
        
        # Pass the final output through the fully connected layer
        final_out = self.fc(last_time_step_out)

        # If it's binary classification (num_classes=1), squeeze the last dimension
        return final_out.squeeze(1)

# --- 2. Training and Validation Functions ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for sequences, labels in tqdm(dataloader, desc="Training"):
        sequences, labels = sequences.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc="Validating"):
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Convert logits to probabilities (0-1) and then to predictions (0 or 1)
            preds = torch.sigmoid(outputs).round()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_loss = running_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, zero_division=0)
    val_recall = recall_score(all_labels, all_preds, zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return val_loss, val_acc, val_precision, val_recall, val_f1

# --- 3. Main Execution Block ---
if __name__ == '__main__':
    # --- Configuration ---
    CONFIG = {
        "features_path": "features",
        "labels_path": "features/labels.csv",
        "checkpoint_dir": "checkpoints_lstm",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.0005,
        # Model Hyperparameters
        "input_dim": 2048, # This MUST match the feature dimension from Phase 1
        "hidden_dim": 256,  # Capacity of the LSTM's memory
        "num_layers": 2,    # Number of stacked LSTM layers
        "num_classes": 1,   # Single output for binary (crash/no-crash)
        "dropout": 0.5,
    }
    
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    print(f"Using device: {CONFIG['device']}")

    # --- Load Data ---
    labels_df = pd.read_csv(CONFIG["labels_path"], index_col='video_id')
    train_dataset = VideoSequenceDataset(CONFIG["features_path"], labels_df, 'train')
    val_dataset = VideoSequenceDataset(CONFIG["features_path"], labels_df, 'val')
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=pad_collate_fn)

    # --- Initialize Model, Loss, and Optimizer ---
    model = LSTMClassifier(
        CONFIG["input_dim"], CONFIG["hidden_dim"], CONFIG["num_layers"], CONFIG["num_classes"], CONFIG["dropout"]
    ).to(CONFIG["device"])
    
    # BCEWithLogitsLoss is perfect for binary classification. It's numerically stable.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # --- Training Loop ---
    best_f1 = 0.0
    for epoch in range(CONFIG["epochs"]):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['epochs']} ---")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG["device"])
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate_one_epoch(model, val_loader, criterion, CONFIG["device"])
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")
        print(f"  Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val F1-Score: {val_f1:.4f}")
        
        # Save the best model based on F1-score
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_path = os.path.join(CONFIG["checkpoint_dir"], "best_lstm_model.ckpt")
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New best model saved to {best_model_path} (F1-Score: {best_f1:.4f})")
            
    print("\n--- Training complete! ---")
    print(f"Best validation F1-score achieved: {best_f1:.4f}")