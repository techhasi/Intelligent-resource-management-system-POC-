import boto3
from datetime import datetime, timedelta, timezone
import sqlite3
import threading
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import time

# Initialize AWS CloudWatch and EC2 clients
cloudwatch_client = boto3.client('cloudwatch', region_name='ap-southeast-1')
ec2_client = boto3.client('ec2', region_name='ap-southeast-1')

# Global flag for monitoring status
monitoring_flag = False

# Function to create the metrics table if it doesn't exist
def create_metrics_table():
    with sqlite3.connect('cloud_metrics.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                service TEXT,
                instance_id TEXT,
                metric_name TEXT,
                metric_value REAL
            )
        ''')
        conn.commit()

# Function to get CloudWatch metrics for a specific instance and metric
def get_cloudwatch_metrics(instance_id, namespace, metric_name, dimension_name='InstanceId'):
    response = cloudwatch_client.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric_name,
        Dimensions=[
            {'Name': dimension_name, 'Value': instance_id}
        ],
        StartTime=datetime.now(timezone.utc) - timedelta(minutes=10),
        EndTime=datetime.now(timezone.utc),
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
        cursor.execute('''
            INSERT INTO metrics (service, instance_id, metric_name, metric_value)
            VALUES (?, ?, ?, ?)
        ''', (service, instance_id, metric_name, metric_value))
        conn.commit()

# Enhanced threshold-based decision-making function for scaling
def scale_decision(ec2_cpu, ec2_disk_read, ec2_disk_write, ec2_memory, rds_connections, rds_cpu, rds_memory, ecs_cpu, ecs_memory):
    if ec2_cpu > 75 or rds_cpu > 80 or ecs_cpu > 70:
        return 'scale_up'
    elif ec2_cpu < 30 and rds_cpu < 40 and ecs_cpu < 30 and rds_connections < 100:
        return 'scale_down'
    else:
        return 'no_action'

# Function to apply scaling action using the AWS API
def apply_scaling_action(instance_id, action):
    if action == 'scale_up':
        ec2_client.start_instances(InstanceIds=[instance_id])
        print("Scaled up EC2 instance:", instance_id)
    elif action == 'scale_down':
        ec2_client.stop_instances(InstanceIds=[instance_id])
        print("Scaled down EC2 instance:", instance_id)

# Continuous monitoring and scaling loop
def monitor_and_scale(instance_id, rds_instance_id, ecs_service_name):
    global monitoring_flag
    while monitoring_flag:
        # Collect metrics for EC2, RDS, and ECS
        cpu_utilization_ec2 = get_cloudwatch_metrics(instance_id=instance_id, namespace='AWS/EC2', metric_name='CPUUtilization')
        disk_read_ops_ec2 = get_cloudwatch_metrics(instance_id=instance_id, namespace='AWS/EC2', metric_name='DiskReadOps')
        disk_write_ops_ec2 = get_cloudwatch_metrics(instance_id=instance_id, namespace='AWS/EC2', metric_name='DiskWriteOps')
        memory_utilization_ec2 = get_cloudwatch_metrics(instance_id=instance_id, namespace='CWAgent', metric_name='mem_used_percent')

        connections_rds = get_cloudwatch_metrics(instance_id=rds_instance_id, namespace='AWS/RDS', metric_name='DatabaseConnections', dimension_name='DBInstanceIdentifier')
        cpu_utilization_rds = get_cloudwatch_metrics(instance_id=rds_instance_id, namespace='AWS/RDS', metric_name='CPUUtilization', dimension_name='DBInstanceIdentifier')
        freeable_memory_rds = get_cloudwatch_metrics(instance_id=rds_instance_id, namespace='AWS/RDS', metric_name='FreeableMemory', dimension_name='DBInstanceIdentifier')

        cpu_utilization_ecs = get_cloudwatch_metrics(instance_id=ecs_service_name, namespace='AWS/ECS', metric_name='CPUUtilization', dimension_name='ServiceName')
        memory_utilization_ecs = get_cloudwatch_metrics(instance_id=ecs_service_name, namespace='AWS/ECS', metric_name='MemoryUtilization', dimension_name='ServiceName')

        # Save metrics to database
        if cpu_utilization_ec2 is not None:
            save_metrics('EC2', instance_id, 'CPUUtilization', cpu_utilization_ec2)
        if disk_read_ops_ec2 is not None:
            save_metrics('EC2', instance_id, 'DiskReadOps', disk_read_ops_ec2)
        if disk_write_ops_ec2 is not None:
            save_metrics('EC2', instance_id, 'DiskWriteOps', disk_write_ops_ec2)
        if memory_utilization_ec2 is not None:
            save_metrics('EC2', instance_id, 'MemoryUsage', memory_utilization_ec2)
        if connections_rds is not None:
            save_metrics('RDS', rds_instance_id, 'DatabaseConnections', connections_rds)
        if cpu_utilization_rds is not None:
            save_metrics('RDS', rds_instance_id, 'CPUUtilization', cpu_utilization_rds)
        if freeable_memory_rds is not None:
            save_metrics('RDS', rds_instance_id, 'FreeableMemory', freeable_memory_rds)
        if cpu_utilization_ecs is not None:
            save_metrics('ECS', ecs_service_name, 'CPUUtilization', cpu_utilization_ecs)
        if memory_utilization_ecs is not None:
            save_metrics('ECS', ecs_service_name, 'MemoryUtilization', memory_utilization_ecs)

        # Scale decision and action
        decision = scale_decision(cpu_utilization_ec2, disk_read_ops_ec2, disk_write_ops_ec2, memory_utilization_ec2, connections_rds, cpu_utilization_rds, freeable_memory_rds, cpu_utilization_ecs, memory_utilization_ecs)
        apply_scaling_action(instance_id, decision)
        time.sleep(300)  # Refresh every 5 minutes

# Streamlit Dashboard
st.title("AWS Cloud Resource Monitoring Dashboard")
create_metrics_table()

# Define placeholders
table_placeholder = st.empty()
graphs_placeholder = st.empty()

def start_monitoring():
    global monitoring_flag
    monitoring_flag = True
    monitor_thread = threading.Thread(target=monitor_and_scale, args=('i-0161c9ed927d94b4e', 'database-1', 'nginx-service'))
    monitor_thread.start()

def stop_monitoring():
    global monitoring_flag
    monitoring_flag = False

# Start/Stop Monitoring Buttons
if st.button("Start Monitoring") and not monitoring_flag:
    start_monitoring()
elif st.button("Stop Monitoring") and monitoring_flag:
    stop_monitoring()


# Function to fetch data from the database
def fetch_data():
    conn = sqlite3.connect('cloud_metrics.db')
    df = pd.read_sql_query("SELECT * FROM metrics ORDER BY timestamp DESC LIMIT 100", conn)
    conn.close()
    return df

# Function to update the table in real-time
def update_table():
    df = fetch_data()
    with table_placeholder:
        st.subheader("Metrics Table")
        st.write(df)

# Continuous update for table and graphs
while monitoring_flag:
    conn = sqlite3.connect('cloud_metrics.db')
    df = pd.read_sql_query("SELECT * FROM metrics ORDER BY timestamp DESC LIMIT 100", conn)
    conn.close()
    
    # Update table
    table_placeholder.write(df)
    
    # Update graphs
    with graphs_placeholder:
        st.subheader("Metrics Graphs")
        for metric_name in df['metric_name'].unique():
            metric_df = df[df['metric_name'] == metric_name]
            fig = px.line(metric_df, x="timestamp", y="metric_value", title=metric_name)
            unique_key = f"plot_{metric_name}_{int(time.time() * 1000)}"  # Unique key with timestamp
            st.plotly_chart(fig, key=unique_key)
    
    time.sleep(5)  # Update every 5 seconds
