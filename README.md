# Cloud Resource Optimization: Forecasting & Scaling

This repository contains two key modules for intelligent cloud resource management:

1. **LSTM for Cloud Resource Metrics Forecasting**  
   Forecasts future resource utilization (e.g., CPU usage) for AWS services (EC2, RDS, ECS) based on historical data.

2. **Deep Q-Network (DQN) for Resource Scaling**  
   Uses reinforcement learning to decide optimal scaling actions (scale up, scale down, or no action) in a simulated cloud environment.

Both modules are implemented using Python with PyTorch and scikit-learn and are designed to help optimize cloud operations by leveraging machine learning and reinforcement learning techniques.

---

## Table of Contents

- [Overview](#overview)
- [LSTM for Cloud Resource Metrics Forecasting](#lstm-for-cloud-resource-metrics-forecasting)
  - [Overview](#overview-1)
  - [Dependencies](#dependencies)
  - [Code Breakdown](#code-breakdown)
  - [How to Run](#how-to-run)
- [Deep Q-Network for Resource Scaling](#deep-q-network-for-resource-scaling)
  - [Overview](#overview-2)
  - [Dependencies](#dependencies-1)
  - [Code Breakdown](#code-breakdown-1)
  - [How to Run](#how-to-run-1)
- [Contributing](#contributing)
- [License](#license)

---

## LSTM for Cloud Resource Metrics Forecasting

### Overview

This module implements a Bidirectional LSTM model using PyTorch to forecast key AWS resource metrics. The code follows these steps:

- **Data Loading & Preprocessing:**  
  Mounts Google Drive (for Colab), loads a CSV file with resource metrics, converts timestamps, extracts time-based features, computes utilization ratios and rolling averages, and normalizes the data.

- **Sequence Generation:**  
  Converts the time-series data into sequences (using a look-back window of 30 timesteps) for LSTM input.

- **Model Definition:**  
  Defines a Bidirectional LSTM network with three layers (256 hidden units per layer) and a fully connected output layer.

- **Training:**  
  Trains the model using a custom PyTorch `Dataset` and `DataLoader`, applies gradient clipping, and schedules the learning rate. The trained model is saved to Google Drive.

- **Prediction & Evaluation:**  
  Loads the saved model, generates predictions on test data, inversely scales the predictions back to the original range, and computes evaluation metrics (MAE, MSE, RMSE). Predictions are saved as a CSV file.

### Dependencies

- Python 3.10.16  
- [pandas](https://pandas.pydata.org/)  
- [numpy](https://numpy.org/)  
- [torch (PyTorch)](https://pytorch.org/)  
- [scikit-learn](https://scikit-learn.org/)  
- Google Colab (for drive mounting, if applicable)

Install the required packages via pip:

```bash
pip install pandas numpy torch scikit-learn
