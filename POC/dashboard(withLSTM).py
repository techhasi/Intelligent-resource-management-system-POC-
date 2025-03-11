import boto3
from datetime import datetime, timedelta
import sqlite3
import threading
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import pytz  # Import pytz for timezone handling
import torch
import joblib  # For loading pre-trained models

# Initialize AWS CloudWatch and EC2 clients
cloudwatch_client = boto3.client('cloudwatch', region_name='ap-southeast-1')
ec2_client = boto3.client('ec2', region_name='ap-southeast-1')
rds_client = boto3.client('rds', region_name='ap-southeast-1')
ecs_client = boto3.client('ecs', region_name='ap-southeast-1')
autoscaling_client = boto3.client('autoscaling', region_name='ap-southeast-1')  # Auto Scaling client

# Global flag for monitoring status
monitoring_flag = False

# Set the timezone to Asia/Colombo
colombo_tz = pytz.timezone('Asia/Colombo')

# Load pre-trained LSTM model and scaler
lstm_model = joblib.load('lstm_model.joblib')
scaler = joblib.load('scaler.joblib')

# Function to create the metrics table if it doesn't exist
def create_metrics_table():
    with sqlite3.connect('cloud_metrics.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS metrics (
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            service TEXT,
            instance_id TEXT,
            metric_name TEXT,
            metric_value REAL
        )''')
        conn.commit()

# Function to get CloudWatch metrics for a specific instance and metric
def get_cloudwatch_metrics(instance_id, namespace, metric_name, dimension_name='InstanceId'):
    now_colombo = datetime.now(colombo_tz)
    start_time = now_colombo - timedelta(minutes=10)
    
    response = cloudwatch_client.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric_name,
        Dimensions=[{'Name': dimension_name, 'Value': instance_id}],
        StartTime=start_time,
        EndTime=now_colombo,
        Period=300,  # 5-minute intervals
        Statistics=['Average']
    )
    data_points = response['Datapoints']
    if data_points:
        return data_points[0]['Average']
    else:
        return None

# Function to save metrics to the database
def save_metrics(service, instance_id, metric_name, metric_value):
    with sqlite3.connect('cloud_metrics.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO metrics (service, instance_id, metric_name, metric_value)
                          VALUES (?, ?, ?, ?)''', (service, instance_id, metric_name, metric_value))
        conn.commit()

# Function to predict future utilization with LSTM model
def predict_utilization(metrics_df):
    # Scale and reshape data for LSTM
    data = scaler.transform(metrics_df)
    data = torch.FloatTensor(data).view(-1, 1, len(metrics_df.columns))
    
    # Predict the next utilization
    lstm_model.eval()
    with torch.no_grad():
        predicted = lstm_model(data)
    return scaler.inverse_transform(predicted.numpy())

# Enhanced threshold-based decision-making function for scaling based on predictions
def scale_decision(predicted_ec2, predicted_rds, predicted_ecs):
    # Initialize actions for each service
    actions = {
        "EC2": "no_action",
        "RDS": "no_action",
        "ECS": "no_action"
    }

    # EC2 Scaling Logic
    ec2_cpu, ec2_memory = predicted_ec2[0], predicted_ec2[1]
    if ec2_cpu > 75 or ec2_memory > 75:
        actions["EC2"] = "scale_up"
    elif ec2_cpu < 30 and ec2_memory < 30:
        actions["EC2"] = "scale_down"

    # RDS Scaling Logic
    rds_cpu, rds_memory = predicted_rds[0], predicted_rds[1]
    if rds_cpu > 80 or rds_memory < 25:
        actions["RDS"] = "scale_up"
    elif rds_cpu < 40 and rds_memory > 60:
        actions["RDS"] = "scale_down"

    # ECS Scaling Logic
    ecs_cpu, ecs_memory = predicted_ecs[0], predicted_ecs[1]
    if ecs_cpu > 70 or ecs_memory > 70:
        actions["ECS"] = "scale_up"
    elif ecs_cpu < 30 and ecs_memory < 30:
        actions["ECS"] = "scale_down"

    return actions

# Function to apply scaling action using AWS SDK
def apply_scaling_action(actions):
    for service, action in actions.items():
        if service == "EC2":
            asg_name = "<your_autoscaling_group_name>"  # Replace with Auto Scaling Group name
            
            # EC2 Scaling with Auto Scaling Group
            if action == "scale_up":
                autoscaling_client.update_auto_scaling_group(
                    AutoScalingGroupName=asg_name,
                    DesiredCapacity=2  # Adjust this based on your scaling needs
                )
                print(f"Scaled up EC2 instances in Auto Scaling Group: {asg_name}")
            elif action == "scale_down":
                autoscaling_client.update_auto_scaling_group(
                    AutoScalingGroupName=asg_name,
                    DesiredCapacity=1  # Minimum or reduced instance count
                )
                print(f"Scaled down EC2 instances in Auto Scaling Group: {asg_name}")

        elif service == "RDS":
            db_instance_identifier = "<your_rds_instance_id>"  # Replace with actual RDS instance identifier
            if action == "scale_up":
                rds_client.modify_db_instance(
                    DBInstanceIdentifier=db_instance_identifier,
                    AllocatedStorage=20  # Example of scaling up storage, adjust as needed
                )
                print(f"Scaled up RDS instance: {db_instance_identifier}")
            elif action == "scale_down":
                rds_client.modify_db_instance(
                    DBInstanceIdentifier=db_instance_identifier,
                    AllocatedStorage=10  # Example of scaling down storage
                )
                print(f"Scaled down RDS instance: {db_instance_identifier}")

        elif service == "ECS":
            cluster_name = "<your_ecs_cluster_name>"  # Replace with ECS cluster name
            service_name = "<your_ecs_service_name>"  # Replace with ECS service name
            if action == "scale_up":
                ecs_client.update_service(
                    cluster=cluster_name,
                    service=service_name,
                    desiredCount=2  # Increase task count
                )
                print(f"Scaled up ECS service: {service_name}")
            elif action == "scale_down":
                ecs_client.update_service(
                    cluster=cluster_name,
                    service=service_name,
                    desiredCount=1  # Decrease task count
                )
                print(f"Scaled down ECS service: {service_name}")

# Continuous monitoring and scaling loop
def monitor_and_scale():
    global monitoring_flag
    while monitoring_flag:
        # Collect recent metrics data for prediction
        metrics_df = fetch_recent_metrics()

        # Predict future utilization
        predicted_ec2 = predict_utilization(metrics_df[['ec2_cpu', 'ec2_memory']])
        predicted_rds = predict_utilization(metrics_df[['rds_cpu', 'rds_memory']])
        predicted_ecs = predict_utilization(metrics_df[['ecs_cpu', 'ecs_memory']])

        # Scale decision and action based on predictions
        decision = scale_decision(predicted_ec2, predicted_rds, predicted_ecs)
        apply_scaling_action(decision)

        time.sleep(300)  # Refresh every 5 minutes

# Fetch recent metrics data
def fetch_recent_metrics():
    conn = sqlite3.connect('cloud_metrics.db')
    df = pd.read_sql_query("SELECT * FROM metrics ORDER BY timestamp DESC LIMIT 10", conn)
    conn.close()
    metrics_pivot = df.pivot(index='timestamp', columns='metric_name', values='metric_value')
    return metrics_pivot.fillna(0)

# Streamlit Dashboard
st.title("AWS Cloud Resource Monitoring with Predictive Auto-Scaling")
create_metrics_table()

def start_monitoring():
    global monitoring_flag
    monitoring_flag = True
    monitor_thread = threading.Thread(target=monitor_and_scale)
    monitor_thread.start()

def stop_monitoring():
    global monitoring_flag
    monitoring_flag = False

# Start/Stop Monitoring Buttons
st.sidebar.title("Control Panel")
if st.sidebar.button("Start Monitoring") and not monitoring_flag:
    start_monitoring()
    time.sleep(10)
elif st.sidebar.button("Stop Monitoring") and monitoring_flag:
    stop_monitoring()
