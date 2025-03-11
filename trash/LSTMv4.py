# %% [markdown]
# ## **LSTM for Cloud Resource Metrics Forecasting**
# 
# **The model is designed to forecast key AWS resource metrics (EC2, RDS, and ECS CPU utilization) by leveraging historical cloud monitoring data.**

# %% [markdown]
# 1. Import libraries

# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# %% [markdown]
# 2. Mount google drive (If needed)

# %%
# Mount Google Drive (if using Colab)
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# 3. Feature engineering
# 
#     * Extracts time-based features such as the hour of day and day of week.
#     * Creates utilization ratios (e.g., EC2_CPU/MEM ratio).
#     * Computes rolling means to smooth out fluctuations.
#     * Creates lag features

# %%
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_lag_features(df, feature_cols, lag=3):
    for col in feature_cols:
        for i in range(1, lag + 1):
            df[f"{col}_lag{i}"] = df[col].shift(i)
    df.dropna(inplace=True)
    return df

def add_engineered_features(df, service):
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[~df.index.isna()]

    # Time-based features
    df.loc[:, 'hour'] = df.index.hour
    df.loc[:, 'day_of_week'] = df.index.dayofweek

    if service == 'EC2':
        df.loc[:, 'EC2_CPU_Memory_Ratio'] = np.clip(df['EC2_CPUUtilization'] / (df['EC2_MemoryUtilization'] + 1e-5), 0, 100)
        df.loc[:, 'EC2_CPU_rolling_mean'] = df['EC2_CPUUtilization'].rolling(window=5).mean()
    elif service == 'RDS':
        df.loc[:, 'RDS_Connections_Per_CPU'] = np.clip(df['RDS_DatabaseConnections'] / (df['RDS_CPUUtilization'] + 1e-5), 0, 100)
        df.loc[:, 'RDS_CPU_rolling_mean'] = df['RDS_CPUUtilization'].rolling(window=5).mean()
    elif service == 'ECS':
        df.loc[:, 'ECS_CPU_Memory_Ratio'] = np.clip(df['ECS_CPUUtilization'] / (df['ECS_MemoryUtilization'] + 1e-5), 0, 100)
        df.loc[:, 'ECS_Task_Rolling_Mean'] = df['ECS_RunningTaskCount'].rolling(window=5).mean()

    # Drop rows with NaN values after rolling calculations
    df.dropna(inplace=True)
    return df


# %% [markdown]
# 4. Dataset class

# %%
class TimeSeriesDataset(Dataset):
    def __init__(self, data, features, target_col, sequence_length=10):
        self.features = data[features].values
        self.targets = data[target_col].values
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        x = self.features[idx:idx+self.sequence_length]
        y = self.targets[idx+self.sequence_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# %% [markdown]
# 5. LSTM Model Definition

# %%
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.35):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# %% [markdown]
# 6. Training the models

# %%
def train_model(model, train_loader, val_loader, epochs=50, learning_rate=0.0005, early_stop_patience=5, early_stop_start=15):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # Scheduler with step size of 5 for more aggressive decay
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    
    model.to(device)
    best_val_loss = float('inf')
    patience = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        val_loss = evaluate_model(model, val_loader)
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping check starting from early_stop_start epoch
        if epoch+1 >= early_stop_start:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

# %% [markdown]
# 7. Evaluate models

# %%
def evaluate_model(model, val_loader):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch).squeeze()
            total_loss += criterion(y_pred, y_batch).item()
    return total_loss / len(val_loader)

def predict_and_evaluate(model, test_loader, target_scaler, target_feature):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).cpu().numpy()
            predictions.extend(y_pred)
            actuals.extend(y_batch.numpy())
    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)
    predictions = target_scaler.inverse_transform(predictions)
    actuals = target_scaler.inverse_transform(actuals)
    
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    print(f"\nEvaluation for {target_feature}: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}")
    
    # Plot prediction vs actual
    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label='Actual', alpha=0.7)
    plt.plot(predictions, label='Predicted', alpha=0.7)
    plt.title(f"Prediction vs Actual for {target_feature}")
    plt.xlabel("Sample")
    plt.ylabel(target_feature)
    plt.legend()
    plt.show()
    
    results = pd.DataFrame({"Actual": actuals.flatten(), "Predicted": predictions.flatten()})
    results.to_csv(f"{target_feature}_predictions.csv", index=False)
    return mae, mse, rmse

# %% [markdown]
# 8. Main process

# %%
dataset_info = {
    'EC2': {
        'path': "reduced_ec2_data.csv",
        'features': ['EC2_CPUUtilization', 'EC2_MemoryUtilization', 'EC2_DiskWriteOps', 'EC2_NetworkIn']
    },
    'RDS': {
        'path': "reduced_rds_data.csv",
        'features': ['RDS_CPUUtilization', 'RDS_FreeableMemory', 'RDS_DatabaseConnections', 'RDS_WriteIOPS']
    },
    'ECS': {
        'path': "reduced_ecs_data.csv",
        'features': ['ECS_CPUUtilization', 'ECS_MemoryUtilization', 'ECS_RunningTaskCount']
    }
}

# Additional engineered features for each service
engineered_features = {
    'EC2': ['hour', 'day_of_week', 'EC2_CPU_Memory_Ratio', 'EC2_CPU_rolling_mean'],
    'RDS': ['hour', 'day_of_week', 'RDS_Connections_Per_CPU', 'RDS_CPU_rolling_mean'],
    'ECS': ['hour', 'day_of_week', 'ECS_CPU_Memory_Ratio', 'ECS_Task_Rolling_Mean']
}

# Dictionary to store target scalers (one per service)
target_scalers = {}

sequence_length = 10
batch_size = 64
epochs = 50

for resource, info in dataset_info.items():
    print(f"\n=== Processing {resource} Dataset ===")
    df = pd.read_csv(info['path'], index_col=0)
    df.index = pd.to_datetime(df.index, errors='coerce')
    
    # Add engineered features
    df = add_engineered_features(df, resource)
    
    # Initial feature list: original + engineered
    base_features = info['features'] + engineered_features[resource]
    
    # Create lag features and update feature list
    df = create_lag_features(df, base_features, lag=3)
    features = [col for col in df.columns if col in base_features or any(col.startswith(f"{feat}_lag") for feat in base_features)]
    target_feature = info['features'][0]

    print(f"{resource} - Target ({target_feature}) mean: {df[target_feature].mean():.4f}, std: {df[target_feature].std():.4f}")
    print(f"{resource} - Total features: {len(features)}")  # Debug: Confirm feature count
    
    # Scale features and target
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    df[features] = feature_scaler.fit_transform(df[features])
    df[target_feature] = target_scaler.fit_transform(df[[target_feature]])
    target_scalers[resource] = target_scaler

    # Create dataset and split
    dataset = TimeSeriesDataset(df, features, target_feature, sequence_length)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    # Train and save model
    model = LSTMModel(input_size=len(features))
    print(f"Training {resource} model with input_size={len(features)}...")
    train_model(model, train_loader, val_loader, epochs=epochs, learning_rate=0.0005)
    
    model_path = f"{resource}_lstm_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved {resource} model to {model_path}")
    
    # Evaluate
    print(f"Evaluating {resource} model...")
    predict_and_evaluate(model, val_loader, target_scalers[resource], target_feature)

print("\nAll models trained, evaluated, and saved!")


