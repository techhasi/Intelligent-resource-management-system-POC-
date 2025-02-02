# %% [markdown]
# Load and Merge Datasets
# (Have merged the dataset using SQLLite because as the dataset is too large kernal crashes. But SQLLite is good in handling these type of data it is used)

# %%
import pandas as pd
import sqlite3

ec2_df = pd.read_csv('ec2_data.csv')
rds_df = pd.read_csv('rds_data.csv')
ecs_df = pd.read_csv('ecs_data.csv')


# Ensure all datasets have a 'timestamp' column for merging
ec2_df['timestamp'] = pd.to_datetime(ec2_df['timestamp'])
rds_df['timestamp'] = pd.to_datetime(rds_df['timestamp'])
ecs_df['timestamp'] = pd.to_datetime(ecs_df['timestamp'])

# Create a connection to an SQLite database (or create one if it doesn't exist)
conn = sqlite3.connect('merged_metrics.db')

ec2_df.to_sql('ec2_data', conn, if_exists='replace', index=False)
rds_df.to_sql('rds_data', conn, if_exists='replace', index=False)
ecs_df.to_sql('ecs_data', conn, if_exists='replace', index=False)

# Verify that the tables were created
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("Tables in the database:")
print(tables)

# Perform the merge operation using SQL
query = """
SELECT 
    ec2_data.timestamp AS ec2_timestamp,
    rds_data.timestamp AS rds_timestamp,
    ecs_data.timestamp AS ecs_timestamp,
    ec2_data.*,
    rds_data.*,
    ecs_data.*
FROM ec2_data
LEFT JOIN rds_data ON ec2_data.timestamp = rds_data.timestamp
LEFT JOIN ecs_data ON ec2_data.timestamp = ecs_data.timestamp

UNION

SELECT 
    ec2_data.timestamp AS ec2_timestamp,
    rds_data.timestamp AS rds_timestamp,
    ecs_data.timestamp AS ecs_timestamp,
    ec2_data.*,
    rds_data.*,
    ecs_data.*
FROM rds_data
LEFT JOIN ec2_data ON rds_data.timestamp = ec2_data.timestamp
LEFT JOIN ecs_data ON rds_data.timestamp = ecs_data.timestamp
WHERE ec2_data.timestamp IS NULL

UNION

SELECT 
    ec2_data.timestamp AS ec2_timestamp,
    rds_data.timestamp AS rds_timestamp,
    ecs_data.timestamp AS ecs_timestamp,
    ec2_data.*,
    rds_data.*,
    ecs_data.*
FROM ecs_data
LEFT JOIN ec2_data ON ecs_data.timestamp = ec2_data.timestamp
LEFT JOIN rds_data ON ecs_data.timestamp = rds_data.timestamp
WHERE ec2_data.timestamp IS NULL AND rds_data.timestamp IS NULL;
"""

# Execute the query and load the result into a DataFrame
df = pd.read_sql_query(query, conn)

# Close the database connection
conn.close()

# Display the combined dataset
print(df.head())

# Save the merged dataset to a CSV file
df.to_csv('merged_cloud_metrics.csv', index=False)

print("Merged dataset saved as 'merged_cloud_metrics.csv'")

# %% [markdown]
# Load the Mapped Dataset

# %%
# Load the mapped dataset
df = pd.read_csv('mapped_cloud_metrics.csv')

# Display the first few rows
print(df.head())

# %% [markdown]
# Handle missing values

# %%
# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Step 4: Handle missing values
print(df.isnull().sum())
df.fillna(df.mean(), inplace=True)

# Step 5: Save the cleaned dataset
df.to_csv('cleaned_cloud_metrics.csv', index=False)

# %% [markdown]
# Feature Engineering
# 1. Temporal features
# 2. Rolling Averages
# 3. Lagged features
# 4. Utilization ratios

# %%
import numpy as np

# Load the preprocessed dataset
df = pd.read_csv('cleaned_cloud_metrics.csv')

# Ensure timestamp is in datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 1. Temporal Features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
df['month'] = df['timestamp'].dt.month

# 2. Rolling Averages (e.g., 5-time-step rolling average for CPU and memory)
window_size = 5
df['cpu_rolling_avg'] = df['EC2_CPUUtilization'].rolling(window=window_size).mean()
df['memory_rolling_avg'] = df['EC2_MemoryUtilization'].rolling(window=window_size).mean()

# Fill NaN values in rolling averages (first few rows)
df.fillna(method='bfill', inplace=True)  # Backward fill

# 3. Lagged Features (e.g., CPU usage at t-1, t-2)
df['cpu_lag_1'] = df['EC2_CPUUtilization'].shift(1)
df['cpu_lag_2'] = df['EC2_CPUUtilization'].shift(2)

# Fill NaN values in lagged features
df.bfill(inplace=True)  # Backward fill

# 4. Utilization Ratios
df['cpu_memory_ratio'] = df['EC2_CPUUtilization'] / df['EC2_MemoryUtilization']
df['network_in_out_ratio'] = df['EC2_NetworkIn'] / df['EC2_NetworkOut']

# Handle division by zero (replace infinite values with NaN and then fill with 0)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# Display the new features
print(df.head())

# %% [markdown]
# Splitting the Data

# %%
from sklearn.model_selection import train_test_split

# Sort by timestamp (ensure data is in chronological order)
df.sort_values('timestamp', inplace=True)

# Define features (X) and target (y)
# For LSTM: Target could be future CPU usage (e.g., next time step)
X = df.drop(columns=['timestamp'])  # Drop timestamp (not a feature)
y = df['EC2_CPUUtilization']  # Example target (can be adjusted)

# Time-based split (e.g., 70% train, 15% validation, 15% test)
train_size = int(0.7 * len(df))
val_size = int(0.15 * len(df))

X_train, X_val_test = X[:train_size], X[train_size:]
y_train, y_val_test = y[:train_size], y[train_size:]

X_val, X_test = X_val_test[:val_size], X_val_test[val_size:]
y_val, y_test = y_val_test[:val_size], y_val_test[val_size:]

# Verify the splits
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Testing set size: {len(X_test)}")

# %% [markdown]
# Normalize the Data

# %%
from sklearn.preprocessing import MinMaxScaler

# Normalize the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Normalize the target (if needed)
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
y_val_scaled = scaler.transform(y_val.values.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))

# %% [markdown]
# Reshape Data for LSTM

# %%
import numpy as np

# Function to create sequences
def create_sequences(data, targets, time_steps=10):
    X_seq, y_seq = [], []
    for i in range(len(data) - time_steps):
        X_seq.append(data[i:i+time_steps])
        y_seq.append(targets[i+time_steps])  # Predict the next CPU usage
    return np.array(X_seq), np.array(y_seq)

# Define time steps (e.g., 10 time steps per sequence)
time_steps = 10

# Create sequences for training, validation, and testing
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, time_steps)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, time_steps)

# Verify the shapes
print(f"Training sequences: {X_train_seq.shape}, Targets: {y_train_seq.shape}")
print(f"Validation sequences: {X_val_seq.shape}, Targets: {y_val_seq.shape}")
print(f"Testing sequences: {X_test_seq.shape}, Targets: {y_test_seq.shape}")

# %% [markdown]
# Train LSTM model

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define the LSTM model
model = Sequential()

# Add LSTM layers
model.add(LSTM(100, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
model.add(Dropout(0.3))  # Increased dropout rate
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.3))

# Add a Dense output layer
model.add(Dense(1))  # Output layer (predicts a single value)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=20,  # Number of epochs
    batch_size=32,  # Batch size
    verbose=1
)

# %% [markdown]
# Evaluate

# %%
# Evaluate on the validation set
val_loss = model.evaluate(X_val_seq, y_val_seq, verbose=0)
print(f"Validation Loss: {val_loss}")

# Evaluate on the testing set
test_loss = model.evaluate(X_test_seq, y_test_seq, verbose=0)
print(f"Testing Loss: {test_loss}")

# Make predictions
y_pred = model.predict(X_test_seq)

# Inverse transform the predictions and targets to original scale
y_pred_original = scaler.inverse_transform(y_pred)
y_test_original = scaler.inverse_transform(y_test_seq)

# Display some predictions
for i in range(5):
    print(f"Predicted: {y_pred_original[i][0]}, Actual: {y_test_original[i][0]}")

# %% [markdown]
# Save model


