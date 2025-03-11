import boto3
from datetime import datetime, timedelta
import sqlite3
import pytz

# Initialize AWS clients
cloudwatch = boto3.client('cloudwatch', region_name='ap-southeast-1') 
ec2 = boto3.client('ec2', region_name='ap-southeast-1')
rds = boto3.client('rds', region_name='ap-southeast-1')
ecs = boto3.client('ecs', region_name='ap-southeast-1')

# SQLite database setup
DATABASE_PATH = '/home/ec2-user/cloud_metrics.db'  # Path to store the SQLite database

# Define the timezone
colombo_tz = pytz.timezone('Asia/Colombo')

def create_database():
    """Create SQLite database and metrics table if it doesn't exist."""
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                timestamp DATETIME,
                service TEXT,
                instance_id TEXT,
                metric_name TEXT,
                metric_value REAL
            )
        ''')
        conn.commit()

def fetch_ec2_metrics(instance_id):
    """Fetch EC2 metrics from CloudWatch."""
    now = datetime.utcnow()
    start_time = now - timedelta(minutes=5)  # Fetch data from the last 5 minutes

    metrics = []
    for metric_name in ['CPUUtilization', 'NetworkIn', 'NetworkOut', 'DiskReadOps', 'DiskWriteOps']:
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/EC2',
            MetricName=metric_name,
            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
            StartTime=start_time,
            EndTime=now,
            Period=60,  # 1-minute intervals
            Statistics=['Average', 'Sum', 'Minimum', 'Maximum', 'SampleCount']
        )
        for datapoint in response['Datapoints']:
            metrics.append({
                'timestamp': datapoint['Timestamp'],
                'metric_name': metric_name,
                'metric_value': datapoint['Average'],
            })
    return metrics

def fetch_rds_metrics(db_instance_id):
    """Fetch RDS metrics from CloudWatch."""
    now = datetime.utcnow()
    start_time = now - timedelta(minutes=5)

    metrics = []
    for metric_name in ['CPUUtilization', 'DatabaseConnections', 'FreeableMemory']:
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/RDS',
            MetricName=metric_name,
            Dimensions=[{'Name': 'DBInstanceIdentifier', 'Value': db_instance_id}],
            StartTime=start_time,
            EndTime=now,
            Period=60,  # 1-minute intervals
            Statistics=['Average', 'Sum', 'Minimum', 'Maximum', 'SampleCount']
        )
        for datapoint in response['Datapoints']:
            metrics.append({
                'timestamp': datapoint['Timestamp'],
                'metric_name': metric_name,
                'metric_value': datapoint['Average'], 
            })
    return metrics

def fetch_ecs_task_metrics(cluster_name, task_id):
    """Fetch ECS task metrics from CloudWatch."""
    now = datetime.utcnow()
    start_time = now - timedelta(minutes=5)

    metrics = []
    for metric_name in ['CPUUtilization', 'MemoryUtilization']:
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/ECS',
            MetricName=metric_name,
            Dimensions=[
                {'Name': 'ClusterName', 'Value': cluster_name},
                {'Name': 'TaskId', 'Value': task_id}
            ],
            StartTime=start_time,
            EndTime=now,
            Period=60,  # 1-minute intervals
            Statistics=['Average', 'Sum', 'Minimum', 'Maximum', 'SampleCount']
        )
        for datapoint in response['Datapoints']:
            metrics.append({
                'timestamp': datapoint['Timestamp'],
                'metric_name': metric_name,
                'metric_value': datapoint['Average'], 
            })
    return metrics

def save_metrics(service, instance_id, metrics):
    """Save metrics to SQLite database."""
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        for metric in metrics:
            # Convert UTC timestamp to Asia/Colombo timezone
            timestamp = metric['timestamp'].astimezone(colombo_tz).strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute('''
                INSERT INTO metrics (timestamp, service, instance_id, metric_name, metric_value)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, service, instance_id, metric['metric_name'], metric['metric_value']))
        conn.commit()

def main():
    """Main function to collect and save metrics."""
    create_database()

    # Fetch and save EC2 metrics
    ec2_instance_id = 'i-0275ce6aa1f61ca9f' 
    ec2_metrics = fetch_ec2_metrics(ec2_instance_id)
    save_metrics('EC2', ec2_instance_id, ec2_metrics)

    # Fetch and save RDS metrics
    rds_instance_id = 'database-2'  
    rds_metrics = fetch_rds_metrics(rds_instance_id)
    save_metrics('RDS', rds_instance_id, rds_metrics)

    # Fetch and save ECS task metrics
    ecs_cluster_name = 'your-ecs-cluster-name' 
    ecs_task_id = '1b624d97ed3c4f40a3709d80613061ac' 
    ecs_metrics = fetch_ecs_task_metrics(ecs_cluster_name, ecs_task_id)
    save_metrics('ECS', ecs_task_id, ecs_metrics)

    print("Metrics collected and saved successfully!")

if __name__ == "__main__":
    main()