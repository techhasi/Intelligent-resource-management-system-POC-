from flask import Flask, jsonify
import torch.nn as nn
import boto3
import torch
import numpy as np
from datetime import datetime, timedelta, timezone
import sqlite3
import threading
import time

# Flask App Initialization
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.35):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

class DQN(nn.Module):
    def __init__(self, state_size, hidden_dim=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head_ec2 = nn.Linear(hidden_dim, 3)  # 3 actions for EC2
        self.head_rds = nn.Linear(hidden_dim, 3)  # 3 actions for RDS
        self.head_ecs = nn.Linear(hidden_dim, 3)  # 3 actions for ECS
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_ec2 = self.head_ec2(x)
        q_rds = self.head_rds(x)
        q_ecs = self.head_ecs(x)
        return q_ec2, q_rds, q_ecs

# Define input sizes
EC2_INPUT_SIZE = 32
RDS_INPUT_SIZE = 32
ECS_INPUT_SIZE = 28

# Model loading
def load_models():
    global ec2_lstm_model, rds_lstm_model, ecs_lstm_model, rl_model
    try:
        ec2_lstm_model = LSTMModel(input_size=EC2_INPUT_SIZE)
        ec2_lstm_model.load_state_dict(torch.load('./Models/EC2_lstm_model.pth', map_location=torch.device('cpu')))
        ec2_lstm_model.eval()
        
        rds_lstm_model = LSTMModel(input_size=RDS_INPUT_SIZE)
        rds_lstm_model.load_state_dict(torch.load('./Models/RDS_lstm_model.pth', map_location=torch.device('cpu')))
        rds_lstm_model.eval()
        
        ecs_lstm_model = LSTMModel(input_size=ECS_INPUT_SIZE)
        ecs_lstm_model.load_state_dict(torch.load('./Models/ECS_lstm_model.pth', map_location=torch.device('cpu')))
        ecs_lstm_model.eval()
        
        rl_model = DQN(state_size=3)
        rl_model.load_state_dict(torch.load('./dqn_scaling_model.pth', map_location=torch.device('cpu')))
        rl_model.eval()
        
        print("Models loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        ec2_lstm_model, rds_lstm_model, ecs_lstm_model, rl_model = None, None, None, None
        return False

# Initialize AWS Clients
try:
    cloudwatch_client = boto3.client('cloudwatch', region_name='ap-southeast-1')
    ec2_client = boto3.client('ec2', region_name='ap-southeast-1')
    autoscaling_client = boto3.client('autoscaling', region_name='ap-southeast-1')
    rds_client = boto3.client('rds', region_name='ap-southeast-1')
    ecs_client = boto3.client('ecs', region_name='ap-southeast-1')
    print("AWS clients initialized successfully")
except Exception as e:
    print(f"Error initializing AWS clients: {e}")
    cloudwatch_client, ec2_client, autoscaling_client, rds_client, ecs_client = None, None, None, None, None

# Flask App
app = Flask(__name__)
dashboard_predictions = {}

# SQLite Database
def create_metrics_table():
    try:
        with sqlite3.connect('cloud_metrics.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS metrics (
                                timestamp DATETIME,
                                service TEXT,
                                instance_id TEXT,
                                metric_name TEXT,
                                metric_value REAL
                            )''')
            conn.commit()
        print("Database table initialized successfully")
    except Exception as e:
        print(f"Error initializing database table: {e}")

# EC2 Hostname
def get_ec2_hostname(instance_id):
    try:
        response = ec2_client.describe_instances(InstanceIds=[instance_id])
        hostname = response['Reservations'][0]['Instances'][0]['PrivateDnsName']
        return hostname
    except Exception as e:
        print(f"Error fetching hostname for instance {instance_id}: {str(e)}")
        return None

# Scaling Functions
def scale_ec2(decision, asg_name):
    try:
        response = autoscaling_client.describe_auto_scaling_groups(AutoScalingGroupNames=[asg_name])
        current_capacity = response['AutoScalingGroups'][0]['DesiredCapacity']
        
        if decision == "scale up":
            new_capacity = current_capacity + 1
            autoscaling_client.set_desired_capacity(AutoScalingGroupName=asg_name, DesiredCapacity=new_capacity)
            print(f"Scaled EC2 ASG {asg_name} up to {new_capacity} instances")
        elif decision == "scale down" and current_capacity > 1:
            new_capacity = current_capacity - 1
            autoscaling_client.set_desired_capacity(AutoScalingGroupName=asg_name, DesiredCapacity=new_capacity)
            print(f"Scaled EC2 ASG {asg_name} down to {new_capacity} instances")
        else:
            print(f"No scaling action for EC2 ASG {asg_name}")
    except Exception as e:
        print(f"Error scaling EC2 ASG {asg_name}: {str(e)}")

def scale_rds(decision, db_instance_id='database-2'):
    try:
        response = rds_client.describe_db_instances(DBInstanceIdentifier=db_instance_id)
        current_class = response['DBInstances'][0]['DBInstanceClass']
        
        instance_classes = ['db.t3.micro', 'db.t3.small', 'db.t3.medium', 'db.t3.large']
        current_idx = instance_classes.index(current_class) if current_class in instance_classes else 0
        
        if decision == "scale up" and current_idx < len(instance_classes) - 1:
            new_class = instance_classes[current_idx + 1]
            rds_client.modify_db_instance(
                DBInstanceIdentifier=db_instance_id, 
                DBInstanceClass=new_class, 
                ApplyImmediately=True
            )
            print(f"Scaled RDS {db_instance_id} up to {new_class}")
        elif decision == "scale down" and current_idx > 0:
            new_class = instance_classes[current_idx - 1]
            rds_client.modify_db_instance(
                DBInstanceIdentifier=db_instance_id, 
                DBInstanceClass=new_class, 
                ApplyImmediately=True
            )
            print(f"Scaled RDS {db_instance_id} down to {new_class}")
        else:
            print(f"No scaling action for RDS {db_instance_id}")
    except Exception as e:
        print(f"Error scaling RDS {db_instance_id}: {str(e)}")

def scale_ecs(decision, cluster_name, task_definition):
    try:
        response = ecs_client.list_tasks(cluster=cluster_name, desiredStatus='RUNNING')
        current_count = len(response['taskArns'])
        print(f"Current running tasks in {cluster_name}: {current_count}")

        if decision == "scale up":
            response = ecs_client.run_task(
                cluster=cluster_name,
                taskDefinition=task_definition,
                count=1,
                launchType='FARGATE',
                networkConfiguration={
                    'awsvpcConfiguration': {
                        'subnets': ['subnet-022cc8297953122fd'],
                        'securityGroups': ['sg-0e09152973f9c89ae'],
                        'assignPublicIp': 'ENABLED'
                    }
                }
            )
            task_arn = response['tasks'][0]['taskArn']
            print(f"Started new task {task_arn} in cluster {cluster_name}, new count: {current_count + 1}")
        elif decision == "scale down" and current_count > 1:
            task_to_stop = response['taskArns'][0]
            ecs_client.stop_task(
                cluster=cluster_name,
                task=task_to_stop,
                reason='Scaling down based on prediction'
            )
            print(f"Stopped task {task_to_stop} in cluster {cluster_name}, new count: {current_count - 1}")
        else:
            print(f"No scaling action for cluster {cluster_name}: {decision}")
    except Exception as e:
        print(f"Error scaling ECS cluster {cluster_name}: {str(e)}")

# Fetch Metrics
def fetch_ec2_metrics(instance_id):
    try:
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(minutes=5)
        metrics_data = {}
        ec2_metric_names = ['CPUUtilization', 'DiskWriteOps', 'NetworkIn']
        cwagent_metric_names = ['mem_used_percent']

        for metric in ec2_metric_names:
            response = cloudwatch_client.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName=metric,
                Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                StartTime=start_time,
                EndTime=now,
                Period=300,
                Statistics=['Average']
            )
            if response['Datapoints']:
                latest_value = sorted(response['Datapoints'], key=lambda x: x['Timestamp'], reverse=True)[0]['Average']
                metrics_data[metric] = float(latest_value)
            else:
                metrics_data[metric] = None
                print(f"No data points for EC2 metric {metric} for instance {instance_id}")

        hostname = get_ec2_hostname(instance_id)
        if hostname:
            for metric in cwagent_metric_names:
                response = cloudwatch_client.get_metric_statistics(
                    Namespace='CWAgent',
                    MetricName=metric,
                    Dimensions=[{'Name': 'host', 'Value': hostname}],
                    StartTime=start_time,
                    EndTime=now,
                    Period=300,
                    Statistics=['Average']
                )
                if response['Datapoints']:
                    latest_value = sorted(response['Datapoints'], key=lambda x: x['Timestamp'], reverse=True)[0]['Average']
                    metrics_data['MemoryUtilization'] = float(latest_value)
                else:
                    metrics_data['MemoryUtilization'] = None
                    print(f"No data points for CWAgent metric {metric} for host {hostname}")
        else:
            metrics_data['MemoryUtilization'] = None
        
        return metrics_data
    except Exception as e:
        print(f"Error fetching EC2 metrics for {instance_id}: {str(e)}")
        return {}

def fetch_rds_metrics(db_instance_id):
    try:
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(minutes=5)
        metrics_data = {}
        metric_names = ['CPUUtilization', 'FreeableMemory', 'DatabaseConnections', 'WriteIOPS']

        for metric in metric_names:
            response = cloudwatch_client.get_metric_statistics(
                Namespace='AWS/RDS',
                MetricName=metric,
                Dimensions=[{'Name': 'DBInstanceIdentifier', 'Value': db_instance_id}],
                StartTime=start_time,
                EndTime=now,
                Period=300,
                Statistics=['Average']
            )
            if response['Datapoints']:
                latest_value = sorted(response['Datapoints'], key=lambda x: x['Timestamp'], reverse=True)[0]['Average']
                metrics_data[metric] = float(latest_value)
            else:
                metrics_data[metric] = None
                print(f"No data points for RDS metric {metric} for instance {db_instance_id}")
        return metrics_data
    except Exception as e:
        print(f"Error fetching RDS metrics for {db_instance_id}: {str(e)}")
        return {}

def fetch_ecs_metrics(cluster_name):
    try:
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(minutes=5)
        metrics_data = {}
        metric_names = ['CPUUtilization', 'MemoryUtilization']

        for metric in metric_names:
            response = cloudwatch_client.get_metric_statistics(
                Namespace='AWS/ECS',
                MetricName=metric,
                Dimensions=[{'Name': 'ClusterName', 'Value': cluster_name}],
                StartTime=start_time,
                EndTime=now,
                Period=300,
                Statistics=['Average']
            )
            if response['Datapoints']:
                latest_value = sorted(response['Datapoints'], key=lambda x: x['Timestamp'], reverse=True)[0]['Average']
                metrics_data[metric] = float(latest_value)
                print(f"ECS {metric} for {cluster_name}: {latest_value}")
            else:
                metrics_data[metric] = None
                print(f"No data points for ECS metric {metric} for cluster {cluster_name}")

        response = ecs_client.list_tasks(cluster=cluster_name, desiredStatus='RUNNING')
        task_count = len(response['taskArns'])
        metrics_data['RunningTaskCount'] = float(task_count) if task_count > 0 else None
        if task_count > 0:
            print(f"Found {task_count} running tasks in cluster {cluster_name}")
        else:
            print(f"No running tasks found for cluster {cluster_name}")

        return metrics_data
    except Exception as e:
        print(f"Error fetching ECS metrics for cluster {cluster_name}: {str(e)}")
        return {}

# API Endpoint
@app.route('/predictions', methods=['GET'])
def get_predictions():
    serializable_predictions = {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in dashboard_predictions.items()
    }
    return jsonify(serializable_predictions)

# Monitoring Thread
def monitor_and_scale():
    global dashboard_predictions
    models_loaded = load_models()
    
    def predict_usage(model, input_data):
        if model is None or not input_data or None in input_data:
            return 0.0
        try:
            model.eval()
            expected_size = model.lstm.weight_ih_l0.shape[1]
            input_data = [v if v is not None else 0.0 for v in input_data]
            if len(input_data) > expected_size:
                input_data = input_data[:expected_size]
            elif len(input_data) < expected_size:
                input_data = input_data + [0.0] * (expected_size - len(input_data))
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                prediction = model(input_tensor).detach().numpy()
            return float(prediction.flatten()[0])
        except Exception as e:
            print(f"Error making LSTM prediction: {e}")
            return 0.0

    if not models_loaded:
        print("Monitoring aborted due to model loading failure")
        return

    rl_decision_map = {0: "scale up", 1: "scale down", 2: "no change"}
    cycle_interval = 300  # 5 minutes
    instance_id = 'i-0275ce6aa1f61ca9f'
    asg_name = 'ec2-scaling'
    rds_instance_id = 'database-2'
    ecs_cluster = 'my-ecs-cluster'
    task_definition = 'my-task-definition'

    while True:
        print("Starting monitoring cycle...")
        
        ec2_metrics = fetch_ec2_metrics(instance_id)
        rds_metrics = fetch_rds_metrics(rds_instance_id)
        ecs_metrics = fetch_ecs_metrics(ecs_cluster)  # Fixed: Removed extra None argument

        print("\nFetched EC2 Metrics:")
        for metric_name, value in ec2_metrics.items():
            print(f"  {metric_name}: {value}")
        
        print("\nFetched RDS Metrics:")
        for metric_name, value in rds_metrics.items():
            print(f"  {metric_name}: {value}")
        
        print("\nFetched ECS Metrics:")
        for metric_name, value in ecs_metrics.items():
            print(f"  {metric_name}: {value}")

        ec2_values = [v if v is not None else 0.0 for v in ec2_metrics.values()]
        rds_values = [v if v is not None else 0.0 for v in rds_metrics.values()]
        ecs_values = [v if v is not None else 0.0 for v in ecs_metrics.values()]

        ec2_pred = predict_usage(ec2_lstm_model, ec2_values)
        rds_pred = predict_usage(rds_lstm_model, rds_values)
        ecs_pred = predict_usage(ecs_lstm_model, ecs_values)

        try:
            rl_input = torch.tensor([ec2_pred, rds_pred, ecs_pred], dtype=torch.float32).unsqueeze(0)
            q_ec2, q_rds, q_ecs = rl_model(rl_input)
            ec2_decision = rl_decision_map[torch.argmax(q_ec2, dim=1).item()]
            rds_decision = rl_decision_map[torch.argmax(q_rds, dim=1).item()]
            ecs_decision = rl_decision_map[torch.argmax(q_ecs, dim=1).item()]
        except Exception as e:
            print(f"Error in RL decision: {e}")
            ec2_decision = rds_decision = ecs_decision = "no change"

        scale_ec2(ec2_decision, asg_name)
        scale_rds(rds_decision, rds_instance_id)
        scale_ecs(ecs_decision, ecs_cluster, task_definition)

        dashboard_predictions = {
            "EC2_CPU": ec2_metrics.get('CPUUtilization', 0.0),
            "RDS_CPU": rds_metrics.get('CPUUtilization', 0.0),
            "ECS_CPU": ecs_metrics.get('CPUUtilization', 0.0),
            "EC2_LSTM_Prediction": ec2_pred,
            "RDS_LSTM_Prediction": rds_pred,
            "ECS_LSTM_Prediction": ecs_pred,
            "EC2_RL_Decision": ec2_decision,
            "RDS_RL_Decision": rds_decision,
            "ECS_RL_Decision": ecs_decision
        }

        print("\nPredictions:")
        for key, value in dashboard_predictions.items():
            print(f"  {key}: {value}")

        print("\nNext prediction cycle in:")
        for remaining in range(cycle_interval, 0, -1):
            minutes, seconds = divmod(remaining, 60)
            print(f"  {minutes:02d}:{seconds:02d}", end="\r")
            time.sleep(1)
        print(" " * 20)

monitoring_thread = threading.Thread(target=monitor_and_scale, daemon=True)
monitoring_thread.start()

if __name__ == '__main__':
    create_metrics_table()
    app.run(host='0.0.0.0', port=5002, debug=False)