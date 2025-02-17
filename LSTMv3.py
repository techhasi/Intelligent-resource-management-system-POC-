# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Mount Google Drive (if using Colab)
from google.colab import drive
drive.mount('/content/drive')

# Load dataset from Google Drive
file_path = "/content/drive/MyDrive/FYPDataset/reduced_merged_cloud_metrics.csv"
df = pd.read_csv(file_path)

# Convert timestamp to datetime and set as index
df['Timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('Timestamp', inplace=True)

# Feature Engineering
# Extract time-based features
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek

# Create utilization ratios
df['EC2_CPU_Memory_Ratio'] = df['EC2_CPUUtilization'] / (df['EC2_MemoryUtilization'] + 1e-5)
df['RDS_Connections_Per_CPU'] = df['RDS_DatabaseConnections'] / (df['RDS_CPUUtilization'] + 1e-5)

# Compute moving averages
df['EC2_CPU_rolling_mean'] = df['EC2_CPUUtilization'].rolling(window=5).mean()
df['RDS_CPU_rolling_mean'] = df['RDS_CPUUtilization'].rolling(window=5).mean()
df.fillna(method='bfill', inplace=True)  # Fill missing values after rolling operations

# Ensure no NaN values remain
df.dropna(inplace=True)

# Select features and target variables
features = [
    'EC2_CPUUtilization', 'EC2_MemoryUtilization', 'EC2_DiskWriteOps', 'EC2_NetworkIn',
    'RDS_CPUUtilization', 'RDS_FreeableMemory', 'RDS_DatabaseConnections', 'RDS_WriteIOPS',
    'ECS_CPUUtilization', 'ECS_MemoryUtilization', 'ECS_RunningTaskCount',
    'hour', 'day_of_week', 'EC2_CPU_Memory_Ratio', 'RDS_Connections_Per_CPU',
    'EC2_CPU_rolling_mean', 'RDS_CPU_rolling_mean'
]

target_cols = ['EC2_CPUUtilization', 'RDS_CPUUtilization', 'ECS_CPUUtilization']

# Normalize data
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Convert dataset to sequences for LSTM
SEQ_LENGTH = 30  # Using past 30 timesteps to predict the next step

def create_sequences(data, target_cols, seq_length):
    sequences, labels = [], []
    target_indices = [features.index(col) for col in target_cols]
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append([data[i+seq_length, idx] for idx in target_indices])  # Ensure proper label extraction
    return np.array(sequences), np.array(labels)

# Prepare input-output sequences
data = df[features].values
sequences, labels = create_sequences(data, target_cols, SEQ_LENGTH)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42, shuffle=False)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define PyTorch dataset
class CloudDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CloudDataset(X_train_tensor, y_train_tensor)
test_dataset = CloudDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define Bidirectional LSTM Model
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Adjust output for bidirectional LSTM

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Taking only last LSTM output

# Model parameters
input_size = len(features)  # Number of input features
hidden_size = 256  # Increased LSTM hidden units
num_layers = 3  # Increased LSTM layers
output_size = len(target_cols)  # Predicting EC2, RDS, ECS CPU utilization

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)  # Learning rate scheduling

# Training loop with validation
EPOCHS = 30

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        total_train_loss += loss.item()
    scheduler.step()

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            output = model(batch_X)
            val_loss = criterion(output, batch_y)
            total_val_loss += val_loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {total_train_loss/len(train_loader):.6f}, Val Loss: {total_val_loss/len(test_loader):.6f}")

# Save the trained model to Google Drive
model_path = "/content/drive/MyDrive/lstm_cloud_modelchatgptv2.pth"
torch.save(model.state_dict(), model_path)

print(f"Model saved at: {model_path}")


# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Mount Google Drive (if using Colab)
from google.colab import drive
drive.mount('/content/drive')

# Load dataset from Google Drive
file_path = "/content/drive/MyDrive/FYPDataset/reduced_merged_cloud_metrics.csv"
df = pd.read_csv(file_path)

# Convert timestamp to datetime and set as index
df['Timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('Timestamp', inplace=True)

# Feature Engineering
# Extract time-based features
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek

# Create utilization ratios
df['EC2_CPU_Memory_Ratio'] = df['EC2_CPUUtilization'] / (df['EC2_MemoryUtilization'] + 1e-5)
df['RDS_Connections_Per_CPU'] = df['RDS_DatabaseConnections'] / (df['RDS_CPUUtilization'] + 1e-5)

# Compute moving averages
df['EC2_CPU_rolling_mean'] = df['EC2_CPUUtilization'].rolling(window=5).mean()
df['RDS_CPU_rolling_mean'] = df['RDS_CPUUtilization'].rolling(window=5).mean()
df.fillna(method='bfill', inplace=True)  # Fill missing values after rolling operations

# Ensure no NaN values remain
df.dropna(inplace=True)

# Select features and target variables
features = [
    'EC2_CPUUtilization', 'EC2_MemoryUtilization', 'EC2_DiskWriteOps', 'EC2_NetworkIn',
    'RDS_CPUUtilization', 'RDS_FreeableMemory', 'RDS_DatabaseConnections', 'RDS_WriteIOPS',
    'ECS_CPUUtilization', 'ECS_MemoryUtilization', 'ECS_RunningTaskCount',
    'hour', 'day_of_week', 'EC2_CPU_Memory_Ratio', 'RDS_Connections_Per_CPU',
    'EC2_CPU_rolling_mean', 'RDS_CPU_rolling_mean'
]

target_cols = ['EC2_CPUUtilization', 'RDS_CPUUtilization', 'ECS_CPUUtilization']

# Normalize data
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Convert dataset to sequences for LSTM
SEQ_LENGTH = 30  # Using past 30 timesteps to predict the next step

def create_sequences(data, target_cols, seq_length):
    sequences, labels = [], []
    target_indices = [features.index(col) for col in target_cols]
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append([data[i+seq_length, idx] for idx in target_indices])  # Ensure proper label extraction
    return np.array(sequences), np.array(labels)

# Prepare input-output sequences
data = df[features].values
sequences, labels = create_sequences(data, target_cols, SEQ_LENGTH)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42, shuffle=False)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define PyTorch dataset
class CloudDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CloudDataset(X_train_tensor, y_train_tensor)
test_dataset = CloudDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define Bidirectional LSTM Model
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Adjust output for bidirectional LSTM

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Taking only last LSTM output

# Model parameters
input_size = len(features)
hidden_size = 256
num_layers = 3
output_size = len(target_cols)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load("/content/drive/MyDrive/lstm_cloud_modelchatgptv2.pth"))
model.eval()

# Function to make predictions
def predict(model, X_test_tensor, device):
    model.to(device)
    X_test_tensor = X_test_tensor.to(device)
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy()
    return predictions

# Generate predictions
y_pred = predict(model, X_test_tensor, device)

# Inverse transform predictions to original scale
y_pred_original = scaler.inverse_transform(np.hstack((y_pred, np.zeros((y_pred.shape[0], len(features) - len(target_cols))))))[:, :len(target_cols)]

# Save predictions
predictions_df = pd.DataFrame(y_pred_original, columns=target_cols)
predictions_df.to_csv("/content/drive/MyDrive/lstm_predictions.csv", index=False)

# Function to calculate accuracy metrics
def calculate_accuracy(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# Compute accuracy
mae, mse, rmse = calculate_accuracy(y_test, y_pred)
print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

print("Predictions saved at: /content/drive/MyDrive/lstm_predictions.csv")



