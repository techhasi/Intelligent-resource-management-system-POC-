import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import joblib
import os

def load_and_preprocess_data(file_path):
    # Load data
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Basic time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
    df['month'] = df['timestamp'].dt.month
    
    # Cyclical time encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    
    # Extended rolling statistics with clip to handle outliers
    metrics = ['EC2_CPUUtilization', 'EC2_MemoryUtilization', 'RDS_CPUUtilization', 
              'RDS_FreeableMemory', 'RDS_DatabaseConnections', 'ECS_CPUUtilization', 
              'ECS_MemoryUtilization']
    
    # First, clip extreme values for each metric
    for metric in metrics:
        q_low = df[metric].quantile(0.001)
        q_high = df[metric].quantile(0.999)
        df[metric] = df[metric].clip(q_low, q_high)
    
    for metric in metrics:
        # 6-hour window
        df[f'{metric}_rolling_mean_6h'] = df[metric].rolling(window=6, min_periods=1).mean()
        df[f'{metric}_rolling_std_6h'] = df[metric].rolling(window=6, min_periods=1).std()
        df[f'{metric}_rolling_max_6h'] = df[metric].rolling(window=6, min_periods=1).max()
        df[f'{metric}_rolling_min_6h'] = df[metric].rolling(window=6, min_periods=1).min()
        
        # 12-hour window
        df[f'{metric}_rolling_mean_12h'] = df[metric].rolling(window=12, min_periods=1).mean()
        df[f'{metric}_rolling_std_12h'] = df[metric].rolling(window=12, min_periods=1).std()
        
        # Add lag features
        df[f'{metric}_lag_1'] = df[metric].shift(1)
        df[f'{metric}_lag_24'] = df[metric].shift(24)
        
        # Rate of change with clipping
        df[f'{metric}_pct_change'] = df[metric].pct_change()
        df[f'{metric}_pct_change'] = df[f'{metric}_pct_change'].clip(-5, 5)  # Limit to ±500%
    
    # Ratio features with safe division
    def safe_divide(a, b, fill_value=0.0):
        return np.where(b > 1e-6, a / b, fill_value)
    
    df['EC2_CPU_Memory_Ratio'] = safe_divide(df['EC2_CPUUtilization'], df['EC2_MemoryUtilization'])
    df['RDS_Connections_Per_CPU'] = safe_divide(df['RDS_DatabaseConnections'], df['RDS_CPUUtilization'])
    
    # Clip ratio features
    df['EC2_CPU_Memory_Ratio'] = df['EC2_CPU_Memory_Ratio'].clip(0, 10)
    df['RDS_Connections_Per_CPU'] = df['RDS_Connections_Per_CPU'].clip(0, 1000)
    
    # Handle missing values
    df = df.replace([np.inf, -np.inf], np.nan)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # Final check for any remaining infinities or NaNs
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    return df


# --- 2. Feature and Target Preparation ---
def prepare_features_targets(df):
    target_cols = ['EC2_CPUUtilization', 'EC2_MemoryUtilization', 'RDS_CPUUtilization',
                   'RDS_FreeableMemory', 'ECS_CPUUtilization', 'ECS_MemoryUtilization']
    
    feature_cols = [col for col in df.columns if col not in ['timestamp'] + target_cols]
    
    # Log transformation for target variables
    EPSILON = 1e-6
    for col in target_cols:
        df[col] = np.log1p(df[col] + EPSILON)
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    return X, y, feature_cols, target_cols

# --- 3. Custom MAPE Metric ---
class CustomMAPE(tf.keras.metrics.Metric):
    def __init__(self, name='custom_mape', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        epsilon = tf.constant(1e-6, dtype=tf.float32)
        absolute_percentage_errors = tf.abs((y_true - y_pred) / (y_true + epsilon))
        mape = tf.reduce_mean(absolute_percentage_errors) * 100
        self.total.assign_add(mape)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / self.count

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

# --- 4. Sequence Creation ---
def create_sequences(X, y, seq_length, pred_horizon):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length - pred_horizon + 1):
        X_seq.append(X[i:(i + seq_length)])
        y_seq.append(y[i + seq_length:i + seq_length + pred_horizon])
    return np.array(X_seq), np.array(y_seq)

# --- 5. Custom Loss Function ---
def custom_loss(y_true, y_pred):
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    epsilon = tf.constant(1e-6, dtype=tf.float32)
    mape = tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return 0.4 * mae + 0.3 * mse + 0.3 * mape

# --- 6. Model Creation ---
def create_model(input_shape, num_targets):
    model = Sequential([
        # CNN layers for feature extraction
        Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # LSTM layers with residual connections
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        
        LSTM(32),
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(num_targets)
    ])
    
    return model

# --- 7. Main Training Function ---
def train_model(file_path, save_dir):
    # Load and preprocess data
    df = load_and_preprocess_data(file_path)
    X, y, feature_cols, target_cols = prepare_features_targets(df)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Scale data
    scalers = {
        'features': RobustScaler(),
        'targets': RobustScaler()
    }
    
    X_train_scaled = scalers['features'].fit_transform(X_train)
    X_test_scaled = scalers['features'].transform(X_test)
    y_train_scaled = scalers['targets'].fit_transform(y_train)
    y_test_scaled = scalers['targets'].transform(y_test)
    
    # Create sequences
    sequence_length = 24
    prediction_horizon = 1
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length, prediction_horizon)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length, prediction_horizon)
    
    # Create and compile model
    input_shape = (sequence_length, X_train_scaled.shape[1])
    model = create_model(input_shape, len(target_cols))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=custom_loss,
        metrics=['mae', CustomMAPE()]
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        ModelCheckpoint(f'{save_dir}/best_model.keras', monitor='val_loss', save_best_only=True)
    ]
    
    # Train model
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_split=0.2,
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )
    
    # Save model and artifacts
    os.makedirs(save_dir, exist_ok=True)
    model.save(f'{save_dir}/final_model.keras')
    joblib.dump(scalers['features'], f'{save_dir}/feature_scaler.joblib')
    joblib.dump(scalers['targets'], f'{save_dir}/target_scaler.joblib')
    np.save(f'{save_dir}/feature_cols.npy', feature_cols)
    np.save(f'{save_dir}/target_cols.npy', target_cols)
    
    return model, history, (X_test_seq, y_test_seq), scalers

# --- 8. Evaluation Function ---
def evaluate_predictions(y_true, y_pred, target_cols, target_scaler):
    y_true_orig = np.expm1(target_scaler.inverse_transform(y_true))
    y_pred_orig = np.expm1(target_scaler.inverse_transform(y_pred))
    
    results = {}
    for i, col in enumerate(target_cols):
        mae = np.mean(np.abs(y_true_orig[:, i] - y_pred_orig[:, i]))
        mape = np.mean(np.abs((y_true_orig[:, i] - y_pred_orig[:, i]) / (y_true_orig[:, i] + 1e-6))) * 100
        results[col] = {'MAE': mae, 'MAPE': mape}
        
        print(f"\n{col}:")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.4f}%")
    
    return results

# Example usage
if __name__ == "__main__":
    file_path = 'reduced_merged_cloud_metrics.csv'
    save_dir = 'cloud_metrics_model'
    
    # Train model
    model, history, test_data, scalers = train_model(file_path, save_dir)
    
    # Make predictions
    X_test_seq, y_test_seq = test_data
    y_pred = model.predict(X_test_seq)
    
    # Evaluate results
    target_cols = np.load(f'{save_dir}/target_cols.npy')
    results = evaluate_predictions(y_test_seq, y_pred, target_cols, scalers['targets'])