import boto3
from datetime import datetime, timedelta
import sqlite3
import threading
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import pytz  # Import pytz for timezone handling
import queue

# Initialize AWS CloudWatch and EC2 clients
cloudwatch_client = boto3.client('cloudwatch', region_name='ap-southeast-1')
ec2_client = boto3.client('ec2', region_name='ap-southeast-1')
rds_client = boto3.client('rds', region_name='ap-southeast-1')
ecs_client = boto3.client('ecs', region_name='ap-southeast-1')
autoscaling_client = boto3.client('autoscaling', region_name='ap-southeast-1')  # Auto Scaling client

decision_queue = queue.Queue()

# Global flag for monitoring status
monitoring_flag = False

# Global variable to store the latest scaling decision
latest_scaling_decision = ""

# Set the timezone to Asia/Colombo
colombo_tz = pytz.timezone('Asia/Colombo')

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
    
    # Function to get ECS metrics
def get_ecs_metrics(service_name, cluster_name, metric_name):
    now_colombo = datetime.now(colombo_tz)
    start_time = now_colombo - timedelta(minutes=10)
    
    response = cloudwatch_client.get_metric_statistics(
        Namespace='AWS/ECS',
        MetricName=metric_name,
        Dimensions=[
            {'Name': 'ClusterName', 'Value': cluster_name},
            {'Name': 'ServiceName', 'Value': service_name}
        ],
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

# Enhanced threshold-based decision-making function for scaling
def scale_decision(ec2_cpu, ec2_disk_read, ec2_disk_write, ec2_memory, rds_connections, rds_cpu, rds_memory, ecs_cpu, ecs_memory):
    # Initialize actions for each service
    actions = {
        "EC2": "no_action",
        "RDS": "no_action",
        "ECS": "no_action"
    }

    # EC2 Scaling Logic
    if ec2_cpu is not None and ec2_memory is not None:
        if ec2_cpu > 75 or ec2_memory > 75:
            actions["EC2"] = "scale_up"
        elif (ec2_cpu < 30 or ec2_memory < 30):
            actions["EC2"] = "scale_down"

    # RDS Scaling Logic
    if rds_cpu is not None and rds_memory is not None:
        if rds_cpu > 80 or rds_memory < 25:  # Assuming threshold for RDS freeable memory at 25%
            actions["RDS"] = "scale_up"
        elif (rds_cpu < 40 and rds_memory > 60 and rds_connections is not None and
              rds_connections < 100):
            actions["RDS"] = "scale_down"

    # ECS Scaling Logic
    if ecs_cpu is not None and ecs_memory is not None:
        if ecs_cpu > 70 or ecs_memory > 70:
            actions["ECS"] = "scale_up"
        elif ecs_cpu < 30 and ecs_memory < 30:
            actions["ECS"] = "scale_down"

    return actions

# Function to apply scaling action using AWS SDK
def apply_scaling_action(actions):
    global latest_scaling_decision  # Access the global variable to store the latest decision

    latest_scaling_decision = ""  # Reset the latest decision

    for service, action in actions.items():
        if action == "no_action":
            latest_scaling_decision = f"No scaling done"
            continue  # Skip if no scaling is required

        if service == "EC2":
            instance_id = "i-0275ce6aa1f61ca9f"  # Replace with actual EC2 instance ID
            asg_name = "ec2-scaling"  # Replace with Auto Scaling Group name if applicable

            # EC2 Scaling with Auto Scaling Group
            if asg_name:  # Check if Auto Scaling Group is used
                try:
                    if action == "scale_up":
                        autoscaling_client.update_auto_scaling_group(
                            AutoScalingGroupName=asg_name,
                            DesiredCapacity=2  # Adjust this based on your scaling needs
                        )
                        latest_scaling_decision = f"Scaled up EC2 instances in Auto Scaling Group: {asg_name}"
                    elif action == "scale_down":
                        autoscaling_client.update_auto_scaling_group(
                            AutoScalingGroupName=asg_name,
                            DesiredCapacity=1  # Minimum or reduced instance count
                        )
                        latest_scaling_decision = f"Scaled down EC2 instances in Auto Scaling Group: {asg_name}"
                except Exception as e:
                    latest_scaling_decision = f"Error updating Auto Scaling Group {asg_name}: {e}"

        elif service == "RDS":
            db_instance_identifier = "database-2"  # Replace with actual RDS instance identifier
            if action == "scale_up":
                rds_client.modify_db_instance(
                    DBInstanceIdentifier=db_instance_identifier,
                    DBInstanceClass="db.t3.small"  # Example of scaling up, adjust as needed
                )
                latest_scaling_decision = f"Scaled up RDS instance: {db_instance_identifier}"
            elif action == "scale_down":
                rds_client.modify_db_instance(
                    DBInstanceIdentifier=db_instance_identifier,
                    DBInstanceClass="db.t3.micro"  # Example of scaling down
                )
                latest_scaling_decision = f"Scaled down RDS instance: {db_instance_identifier}"

        elif service == "ECS":
            cluster_name = "testCLuster1"  # Replace with ECS cluster name
            service_name = "nginx-service"  # Replace with ECS service name
            if action == "scale_up":
                ecs_client.update_service(
                    cluster=cluster_name,
                    service=service_name,
                    desiredCount=2  # Increase task count
                )
                latest_scaling_decision = f"Scaled up ECS service: {service_name}"
            elif action == "scale_down":
                ecs_client.update_service(
                    cluster=cluster_name,
                    service=service_name,
                    desiredCount=1  # Decrease task count
                )
                latest_scaling_decision = f"Scaled down ECS service: {service_name}"

    return latest_scaling_decision  # Return the latest decision


# Continuous monitoring and scaling loop
def monitor_and_scale(instance_id, rds_instance_id, ecs_service_name, ecs_cluster_name):
    global monitoring_flag
    while monitoring_flag:
        # Collect metrics for EC2, RDS, and ECS
        cpu_utilization_ec2 = get_cloudwatch_metrics(instance_id=instance_id, namespace='AWS/EC2', metric_name='CPUUtilization')
        disk_read_ops_ec2 = get_cloudwatch_metrics(instance_id=instance_id, namespace='AWS/EC2', metric_name='DiskReadOps')
        disk_write_ops_ec2 = get_cloudwatch_metrics(instance_id=instance_id, namespace='AWS/EC2', metric_name='DiskWriteOps')
        memory_utilization_ec2 = get_cloudwatch_metrics(instance_id='ip-172-31-28-191.ap-southeast-1.compute.internal', namespace='CWAgent', metric_name='mem_used_percent', dimension_name='host'
)

        connections_rds = get_cloudwatch_metrics(instance_id=rds_instance_id, namespace='AWS/RDS', metric_name='DatabaseConnections', dimension_name='DBInstanceIdentifier')
        cpu_utilization_rds = get_cloudwatch_metrics(instance_id=rds_instance_id, namespace='AWS/RDS', metric_name='CPUUtilization', dimension_name='DBInstanceIdentifier')
        freeable_memory_rds = get_cloudwatch_metrics(instance_id=rds_instance_id, namespace='AWS/RDS', metric_name='FreeableMemory', dimension_name='DBInstanceIdentifier')

        cpu_utilization_ecs = get_ecs_metrics(ecs_service_name, ecs_cluster_name, 'CPUUtilization')
        memory_utilization_ecs = get_ecs_metrics(ecs_service_name, ecs_cluster_name, 'MemoryUtilization')
        running_task_count_ecs = get_ecs_metrics(ecs_service_name, ecs_cluster_name, 'RunningTaskCount')

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
        if running_task_count_ecs is not None:
            save_metrics('ECS', ecs_service_name, 'RunningTaskCount', running_task_count_ecs)


        # Scale decision and action
        decision = scale_decision(cpu_utilization_ec2, disk_read_ops_ec2, disk_write_ops_ec2, memory_utilization_ec2, connections_rds, cpu_utilization_rds, freeable_memory_rds, cpu_utilization_ecs, memory_utilization_ecs)
        latest_decision = apply_scaling_action(decision)  # Get the latest decision
        update_latest_decision(latest_decision)  # Update the dashboard with the latest decision
        time.sleep(300)  # Refresh every 5 minutes

# Streamlit Dashboard
st.title("AWS Cloud Resource Monitoring Dashboard")
create_metrics_table()

# Define placeholders for each service
ec2_placeholder = st.empty()
rds_placeholder = st.empty()
ecs_placeholder = st.empty()

# Placeholder for the latest scaling decision
latest_decision_placeholder = st.sidebar.empty()

# Function to update the latest decision on the dashboard
def update_dashboard_with_latest_decision():
    try:
        # Check if there's a new decision in the queue
        while not decision_queue.empty():
            decision_text = decision_queue.get_nowait()
            latest_decision_placeholder.write(f"**Latest Scaling Decision:** {decision_text}")
    except queue.Empty:
        pass  # No new decisions in the queue

def update_latest_decision(decision_text):
    # Push the decision into the queue
    decision_queue.put(decision_text)

def start_monitoring():
    global monitoring_flag
    monitoring_flag = True
    monitor_thread = threading.Thread(target=monitor_and_scale, args=('i-0275ce6aa1f61ca9f', 'database-2', 'nginx-service', 'testCluster1'))
    monitor_thread.start()

def stop_monitoring():
    global monitoring_flag
    monitoring_flag = False

# Fixed buttons at the top of the page
st.sidebar.title("Control Panel")

if st.sidebar.button("Start Monitoring") and not monitoring_flag:
    start_monitoring()
    time.sleep(10)  # Delay loading graphs by 10 seconds
elif st.sidebar.button("Stop Monitoring") and monitoring_flag:
    stop_monitoring()

# Function to fetch data from the database
def fetch_data():
    conn = sqlite3.connect('cloud_metrics.db')
    df = pd.read_sql_query("SELECT * FROM metrics ORDER BY timestamp DESC LIMIT 100", conn)
    conn.close()
    return df

# Continuous update for graphs only
while monitoring_flag:
    try:
        # Fetch data
        df = fetch_data()

        # Update EC2 graphs
        with ec2_placeholder:
            st.subheader("EC2 Metrics")
            ec2_metrics = ['CPUUtilization', 'DiskReadOps', 'DiskWriteOps', 'MemoryUsage']
            ec2_df = df[(df['service'] == 'EC2') & (df['metric_name'].isin(ec2_metrics))]
            if not ec2_df.empty:
                fig = px.line(ec2_df, x="timestamp", y="metric_value", color="metric_name", title="EC2 Metrics Over Time")
                unique_key = f"ec2_metrics_{time.time()}"
                st.plotly_chart(fig, key=unique_key)

        # Update RDS graphs
        with rds_placeholder:
            st.subheader("RDS Metrics")
            rds_metrics = ['DatabaseConnections', 'CPUUtilization', 'FreeableMemory']
            rds_df = df[(df['service'] == 'RDS') & (df['metric_name'].isin(rds_metrics))]
            if not rds_df.empty:
                fig = px.line(rds_df, x="timestamp", y="metric_value", color="metric_name", title="RDS Metrics Over Time")
                unique_key = f"rds_metrics_{time.time()}"
                st.plotly_chart(fig, key=unique_key)


        # Update ECS graphs
        with ecs_placeholder:
            st.subheader("ECS Metrics")
            for metric_name in ['CPUUtilization', 'MemoryUtilization']:
                metric_df = df[(df['service'] == 'ECS') & (df['metric_name'] == metric_name)]
                if not metric_df.empty:
                    fig = px.line(metric_df, x="timestamp", y="metric_value", title=f"ECS {metric_name}")
                    unique_key = f"ecs_{metric_name}_{time.time()}"
                    st.plotly_chart(fig, key=unique_key)
                    
        # Update the dashboard with the latest scaling decision
        update_dashboard_with_latest_decision()

    except Exception as e:
        st.error(f"An error occurred while fetching or plotting data: {e}")

    # Wait for 5 minutes before the next update
    time.sleep(300)  # Refresh every 5 minutes
    
