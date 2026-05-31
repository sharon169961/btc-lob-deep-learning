import evaluate as ev
import importlib
importlib.reload(ev)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class LOBDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def make_windows(df: pd.DataFrame, feature_cols: list, label_col: str, window: int=50) -> tuple[np.ndarray, np.ndarray]:
    X_list = []
    y_list = []

    features = df[feature_cols].values
    labels = df[label_col].values

    for i in range(window, len(df)):
        X_list.append(features[i-window: i])
        y_list.append(labels[i])


    X = np.array(X_list, dtype= np.float32)
    y=np.array(y_list, dtype=np.int64)

    return X, y

def walk_forward_splits(n: int, n_folds: int = 5, min_train: float = 0.5) -> list[tuple]:
    splits=[]
    test_size = int(n*(1-min_train) / n_folds)

    for fold in range(n_folds):
        test_end = n-fold*test_size
        test_start = test_end - test_size
        train_end = test_start

        if train_end < int(n*min_train):
            break

        splits.append((0, train_end, test_start, test_end))

    return list(reversed(splits))


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:,-1,:]
        return self.classifier(last_hidden)
        


class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation, dropout: float = 0.3):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.chomp = lambda x: x[:, :, :-padding] if padding > 0 else x
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.rescon = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.chomp(out)
        out = self.relu(out)
        out = self.drop(out)
        return self.relu(out + self.rescon(x))
        
class TCNModel(nn.Module):
    def __init__(self, input_size: int, num_channels: list = [64,64,64,64], kernel_size: int = 3, dropout:float=0.3, num_classes: int = 3):
        super().__init__()

        layers = []
        in_ch = input_size
        for i, out_ch in enumerate(num_channels):
            dilation = 2** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_channels[-1], num_classes)
            

    def forward(self, x):
        x = x.permute(0,2,1)
        out = self.network(x)
        out = out[:,:, -1]
        return self.classifier(out)
            

class TransformerModel(nn.Module):
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.3, num_classes: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,dropout=dropout,batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        out = self.transformer(x)
        out = out[:, -1, :]
        return self.classifier(out)
    

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return total_loss / len(loader), np.array(all_preds), np.array(all_labels)


def run_experiment(model_name: str = "lstm"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv("data/lob_features.csv", parse_dates=["timestamp"])

    feature_cols = [c for c in df.columns if c not in ["timestamp", "label", "smooth_return"]]
    label_col = "label"

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    X,y = make_windows(df, feature_cols, label_col, window = 50)
    print(f"Window: {X.shape}, Labels: {y.shape}")

    classes = np.unique(y)
    class_weights = compute_class_weight("balanced", classes=classes, y=y)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    splits = walk_forward_splits(len(X), n_folds = 5)
    fold_hit_rates = []
    fold_sharpes = []
    fold_decay_curves = [] 

    for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(splits):
        print(f"\nFold {fold_idx+1}: train[{tr_start}:{tr_end}] test[{te_start}:{te_end}]")
        X_train,y_train = X[tr_start:tr_end], y[tr_start:tr_end]
        X_test, y_test = X[te_start:te_end], y[te_start:te_end]

        train_loader = DataLoader(LOBDataset(X_train, y_train), batch_size= 128, shuffle=False)
        test_loader = DataLoader(LOBDataset(X_test, y_test), batch_size=128, shuffle=False)


        input_size = X.shape[2]
        if model_name == "lstm":
            model = LSTMModel(input_size).to(device)
        elif model_name == "tcn":
            model = TCNModel(input_size).to(device)
        elif model_name == "transformer":
            model = TransformerModel(input_size).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_loss = float("inf")
        patience_count = 0
        patience = 5
        best_state = None

        for epoch in range(50):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, _, _ = eval_epoch(model, test_loader, criterion, device)

            if (epoch + 1) % 5 == 0:
                print(f"  epoch {epoch+1:>3} | train={train_loss:.4f} | val={val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    model.load_state_dict(best_state)
                    break

        _, preds, labels = eval_epoch(model, test_loader, criterion, device)
        smooth_returns = df["smooth_return"].values[te_start + 50 : te_end + 50]
        mid_prices     = df["mid_price"].values[te_start + 50 : te_end + 50]

        hit_rate = ev.compute_hit_rate(preds, labels)
        sharpe   = ev.compute_sharpe(preds, smooth_returns)
        decay    = ev.compute_signal_decay(preds, mid_prices, max_horizon=50)

        fold_hit_rates.append(hit_rate)
        fold_sharpes.append(sharpe)
        fold_decay_curves.append(decay)

        print(f"  Hit rate : {hit_rate:.3f}")
        print(f"  Sharpe   : {sharpe:.2f}")
        
        
        
    print(f"\n{'='*40}")
    print(f"Model: {model_name.upper()}")
    print(f"Mean hit rate : {np.mean(fold_hit_rates):.3f} ± {np.std(fold_hit_rates):.3f}")
    print(f"Mean Sharpe   : {np.mean(fold_sharpes):.2f} ± {np.std(fold_sharpes):.2f}")

    return fold_hit_rates, fold_sharpes, fold_decay_curves


if __name__ == "__main__":
    all_hit_rates = {}
    all_sharpes   = {}
    all_decays    = {}

    for model_name in ["lstm", "tcn", "transformer"]:
        print(f"\n{'#'*50}")
        print(f"  MODEL: {model_name.upper()}")
        print(f"{'#'*50}")
        hr, sh, dc = run_experiment(model_name)
        all_hit_rates[model_name] = hr
        all_sharpes[model_name]   = sh
        all_decays[model_name]    = [np.mean([fold[i] for fold in dc])
                                     for i in range(50)]

    ev.plot_results(all_hit_rates, all_sharpes, all_decays)


    
                 


