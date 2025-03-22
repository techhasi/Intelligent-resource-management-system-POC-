from flask import Flask, jsonify, request, abort
from flask_cors import CORS
import torch.nn as nn
import boto3
import torch
import numpy as np
from datetime import datetime, timedelta, timezone
import sqlite3
import threading
import time
import logging
from botocore.config import Config
from botocore.exceptions import ClientError
from collections import deque
import os

# Configuration Variables (Edit these as needed)
# API Configuration
API_KEY = os.getenv('API_KEY', 'hasithaw54') 
FRONTEND_ORIGIN = "http://localhost:3000"
HOST = '0.0.0.0'
PORT = 5002

# AWS Configuration
AWS_REGION = 'ap-southeast-1'
BOTO_TIMEOUT = 20  # seconds for connect and read timeout
EC2_INSTANCE_ID = 'i-0275ce6aa1f61ca9f'
EC2_ASG_NAME = 'ec2-scaling'
EC2_EXCLUDE_INSTANCE = "i-0d586a401b59d560f"
RDS_INSTANCE_ID = 'database-2'
ECS_CLUSTER_NAME = 'my-ecs-cluster'
ECS_TASK_DEFINITION = 'my-fluctuate-task'
ECS_SUBNETS = ['subnet-022cc8297953122fd']
ECS_SECURITY_GROUPS = ['sg-0e09152973f9c89ae']

# Model Paths
EC2_MODEL_PATH = '../models/EC2_lstm_model.pth'
RDS_MODEL_PATH = '../models/RDS_lstm_model.pth'
ECS_MODEL_PATH = '../models/ECS_lstm_model.pth'
RL_MODEL_PATH = '../models/dqn_scaling_model.pth'

# Model Parameters
EC2_INPUT_SIZE = 32  # 4 base + 4 engineered + 24 lags
RDS_INPUT_SIZE = 32  # 4 base + 4 engineered + 24 lags
ECS_INPUT_SIZE = 28  # 3 base + 4 engineered + 21 lags
STATE_SIZE = 9
SEQUENCE_LENGTH = 10
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 3
LSTM_DROPOUT = 0.35
DQN_HIDDEN_DIM = 384

# Monitoring Configuration
MONITORING_INTERVAL = 300  # 5 minutes in seconds
DATABASE_PATH = 'cloud_metrics.db'

# Set up logging to a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

# Flask App Initialization
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN}})

# API Key Authentication Decorator
def require_api_key(func):
    def wrapper(*args, **kwargs):
        key = request.headers.get("x-api-key")
        if key != API_KEY:
            logger.warning(f"Unauthorized access attempt with invalid API key: {request.remote_addr}")
            return jsonify({"message": "Invalid or missing API key"}), 403
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

# Model Definitions
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=LSTM_HIDDEN_SIZE, num_layers=LSTM_NUM_LAYERS, dropout=LSTM_DROPOUT):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

class DQN(nn.Module):
    def __init__(self, state_size=STATE_SIZE, hidden_dim=DQN_HIDDEN_DIM):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head_ec2 = nn.Linear(hidden_dim, 3)
        self.head_rds = nn.Linear(hidden_dim, 3)
        self.head_ecs = nn.Linear(hidden_dim, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_ec2 = self.head_ec2(x)
        q_rds = self.head_rds(x)
        q_ecs = self.head_ecs(x)
        return q_ec2, q_rds, q_ecs

# Global Variables
ec2_lstm_model = None
rds_lstm_model = None
ecs_lstm_model = None
rl_model = None
dashboard_predictions = {}
state_history = deque(maxlen=3)
ec2_sequence_history = deque(maxlen=SEQUENCE_LENGTH)
rds_sequence_history = deque(maxlen=SEQUENCE_LENGTH)
ecs_sequence_history = deque(maxlen=SEQUENCE_LENGTH)

# Model Loading
def load_models():
    global ec2_lstm_model, rds_lstm_model, ecs_lstm_model, rl_model
    try:
        ec2_lstm_model = LSTMModel(input_size=EC2_INPUT_SIZE)
        ec2_lstm_model.load_state_dict(torch.load(EC2_MODEL_PATH, map_location=torch.device('cpu')))
        ec2_lstm_model.eval()
        
        rds_lstm_model = LSTMModel(input_size=RDS_INPUT_SIZE)
        rds_lstm_model.load_state_dict(torch.load(RDS_MODEL_PATH, map_location=torch.device('cpu')))
        rds_lstm_model.eval()
        
        ecs_lstm_model = LSTMModel(input_size=ECS_INPUT_SIZE)
        ecs_lstm_model.load_state_dict(torch.load(ECS_MODEL_PATH, map_location=torch.device('cpu')))
        ecs_lstm_model.eval()
        
        rl_model = DQN(state_size=STATE_SIZE)
        rl_model.load_state_dict(torch.load(RL_MODEL_PATH, map_location=torch.device('cpu')))
        rl_model.eval()
        
        logger.info("Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        ec2_lstm_model, rds_lstm_model, ecs_lstm_model, rl_model = None, None, None, None
        return False

# Initialize AWS Clients with timeout
boto_config = Config(connect_timeout=BOTO_TIMEOUT, read_timeout=BOTO_TIMEOUT)
try:
    cloudwatch_client = boto3.client('cloudwatch', region_name=AWS_REGION, config=boto_config)
    ec2_client = boto3.client('ec2', region_name=AWS_REGION, config=boto_config)
    autoscaling_client = boto3.client('autoscaling', region_name=AWS_REGION, config=boto_config)
    rds_client = boto3.client('rds', region_name=AWS_REGION, config=boto_config)
    ecs_client = boto3.client('ecs', region_name=AWS_REGION, config=boto_config)
    logger.info("AWS clients initialized successfully")
except Exception as e:
    logger.error(f"Error initializing AWS clients: {e}")
    cloudwatch_client, ec2_client, autoscaling_client, rds_client, ecs_client = None, None, None, None, None

# SQLite Database
def create_metrics_table():
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS metrics (
                                timestamp DATETIME,
                                service TEXT,
                                instance_id TEXT,
                                metric_name TEXT,
                                metric_value REAL
                            )''')
            conn.commit()
        logger.info("Database table initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database table: {e}")

# EC2 Hostname
def get_ec2_hostname(instance_id):
    try:
        response = ec2_client.describe_instances(InstanceIds=[instance_id])
        hostname = response['Reservations'][0]['Instances'][0]['PrivateDnsName']
        logger.info(f"Fetched hostname for {instance_id}: {hostname}")
        return hostname
    except Exception as e:
        logger.error(f"Error fetching hostname for instance {instance_id}: {str(e)}")
        return None

# Scaling Functions
def scale_ec2(decision, asg_name=EC2_ASG_NAME):
    try:
        response = autoscaling_client.describe_auto_scaling_groups(AutoScalingGroupNames=[asg_name])
        current_capacity = response['AutoScalingGroups'][0]['DesiredCapacity']
        if decision == "scale up":
            new_capacity = current_capacity + 1
            autoscaling_client.set_desired_capacity(AutoScalingGroupName=asg_name, DesiredCapacity=new_capacity)
            logger.info(f"Scaled EC2 ASG {asg_name} up to {new_capacity} instances")
            return f"Scaled EC2 ASG {asg_name} up to {new_capacity} instances"
        elif decision == "scale down" and current_capacity > 1:
            new_capacity = current_capacity - 1
            autoscaling_client.set_desired_capacity(AutoScalingGroupName=asg_name, DesiredCapacity=new_capacity)
            logger.info(f"Scaled EC2 ASG {asg_name} down to {new_capacity} instances")
            return f"Scaled EC2 ASG {asg_name} down to {new_capacity} instances"
        else:
            logger.info(f"No scaling action for EC2 ASG {asg_name}")
            return f"No scaling action for EC2 ASG {asg_name}"
    except ClientError as e:
        logger.error(f"AWS ClientError scaling EC2 ASG {asg_name}: {e}")
        return f"AWS Error: {e}"
    except Exception as e:
        logger.error(f"Error scaling EC2 ASG {asg_name}: {str(e)}")
        return f"Error scaling EC2 ASG {asg_name}: {str(e)}"

def scale_rds(decision, db_instance_id=RDS_INSTANCE_ID):
    try:
        response = rds_client.describe_db_instances(DBInstanceIdentifier=db_instance_id)
        current_class = response['DBInstances'][0]['DBInstanceClass']
        instance_classes = ['db.t3.micro', 'db.t3.small', 'db.t3.medium', 'db.t3.large']
        current_idx = instance_classes.index(current_class) if current_class in instance_classes else 0
        if decision == "scale up" and current_idx < len(instance_classes) - 1:
            new_class = instance_classes[current_idx + 1]
            rds_client.modify_db_instance(DBInstanceIdentifier=db_instance_id, DBInstanceClass=new_class, ApplyImmediately=True)
            logger.info(f"Scaled RDS {db_instance_id} up to {new_class}")
            return f"Scaled RDS {db_instance_id} up to {new_class}"
        elif decision == "scale down" and current_idx > 0:
            new_class = instance_classes[current_idx - 1]
            rds_client.modify_db_instance(DBInstanceIdentifier=db_instance_id, DBInstanceClass=new_class, ApplyImmediately=True)
            logger.info(f"Scaled RDS {db_instance_id} down to {new_class}")
            return f"Scaled RDS {db_instance_id} down to {new_class}"
        else:
            logger.info(f"No scaling action for RDS {db_instance_id}")
            return f"No scaling action for RDS {db_instance_id}"
    except ClientError as e:
        logger.error(f"AWS ClientError scaling RDS {db_instance_id}: {e}")
        return f"AWS Error: {e}"
    except Exception as e:
        logger.error(f"Error scaling RDS {db_instance_id}: {str(e)}")
        return f"Error scaling RDS {db_instance_id}: {str(e)}"

def scale_ecs(decision, cluster_name=ECS_CLUSTER_NAME, task_definition=ECS_TASK_DEFINITION):
    try:
        response = ecs_client.list_tasks(cluster=cluster_name, desiredStatus='RUNNING')
        current_count = len(response['taskArns'])
        if decision == "scale up":
            response = ecs_client.run_task(
                cluster=cluster_name, taskDefinition=task_definition, count=1, launchType='FARGATE',
                networkConfiguration={'awsvpcConfiguration': {'subnets': ECS_SUBNETS, 'securityGroups': ECS_SECURITY_GROUPS, 'assignPublicIp': 'ENABLED'}}
            )
            task_arn = response['tasks'][0]['taskArn']
            logger.info(f"Started new task {task_arn} in cluster {cluster_name}")
            return f"Started new task {task_arn} in cluster {cluster_name}"
        elif decision == "scale down" and current_count > 1:
            task_to_stop = response['taskArns'][0]
            ecs_client.stop_task(cluster=cluster_name, task=task_to_stop, reason='Scaling down manually')
            logger.info(f"Stopped task {task_to_stop} in cluster {cluster_name}")
            return f"Stopped task {task_to_stop} in cluster {cluster_name}"
        else:
            logger.info(f"No scaling action for cluster {cluster_name}")
            return f"No scaling action for cluster {cluster_name}"
    except ClientError as e:
        logger.error(f"AWS ClientError scaling ECS cluster {cluster_name}: {e}")
        return f"AWS Error: {e}"
    except Exception as e:
        logger.error(f"Error scaling ECS cluster {cluster_name}: {str(e)}")
        return f"Error scaling ECS cluster {cluster_name}: {str(e)}"

def stop_ec2():
    try:
        response = ec2_client.describe_instances(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}])
        instances = [instance['InstanceId'] for reservation in response['Reservations'] 
                     for instance in reservation['Instances'] if instance['InstanceId'] != EC2_EXCLUDE_INSTANCE]
        if not instances:
            logger.info("No running EC2 instances found to stop")
            return "No running EC2 instances found to stop"
        ec2_client.stop_instances(InstanceIds=instances)
        logger.info(f"Stopped EC2 instances: {', '.join(instances)}")
        return f"Stopped EC2 instances: {', '.join(instances)}"
    except ClientError as e:
        logger.error(f"AWS ClientError stopping EC2 instances: {e}")
        return f"AWS Error: {e}"
    except Exception as e:
        logger.error(f"Error stopping EC2 instances: {str(e)}")
        return f"Error stopping EC2 instances: {str(e)}"

def stop_rds():
    try:
        response = rds_client.describe_db_instances()
        instances = [db['DBInstanceIdentifier'] for db in response['DBInstances'] if db['DBInstanceStatus'] in ['available', 'starting']]
        stopped = []
        for db_instance_id in instances:
            try:
                rds_client.stop_db_instance(DBInstanceIdentifier=db_instance_id)
                stopped.append(db_instance_id)
            except rds_client.exceptions.InvalidDBInstanceStateFault:
                continue
        result = f"Stopped RDS instances: {', '.join(stopped)}" if stopped else "No stoppable RDS instances found"
        logger.info(result)
        return result
    except ClientError as e:
        logger.error(f"AWS ClientError stopping RDS instances: {e}")
        return f"AWS Error: {e}"
    except Exception as e:
        logger.error(f"Error stopping RDS instances: {str(e)}")
        return f"Error stopping RDS instances: {str(e)}"

def stop_ecs(cluster_name=ECS_CLUSTER_NAME):
    try:
        response = ecs_client.list_tasks(cluster=cluster_name, desiredStatus='RUNNING')
        task_arns = response['taskArns']
        for task_arn in task_arns:
            ecs_client.stop_task(cluster=cluster_name, task=task_arn, reason='Manual stop')
        result = f"Stopped {len(task_arns)} tasks in cluster {cluster_name}" if task_arns else "No running tasks found"
        logger.info(result)
        return result
    except ClientError as e:
        logger.error(f"AWS ClientError stopping ECS cluster {cluster_name}: {e}")
        return f"AWS Error: {e}"
    except Exception as e:
        logger.error(f"Error stopping ECS cluster {cluster_name}: {str(e)}")
        return f"Error stopping ECS cluster {cluster_name}: {str(e)}"

# Fetch Metrics
def fetch_ec2_metrics(instance_id=EC2_INSTANCE_ID):
    try:
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(minutes=5)
        metrics_data = {}
        ec2_metric_names = ['CPUUtilization', 'DiskWriteOps', 'NetworkIn']
        cwagent_metric_names = ['mem_used_percent']
        for metric in ec2_metric_names:
            response = cloudwatch_client.get_metric_statistics(
                Namespace='AWS/EC2', MetricName=metric, Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                StartTime=start_time, EndTime=now, Period=300, Statistics=['Average']
            )
            metrics_data[metric] = float(sorted(response['Datapoints'], key=lambda x: x['Timestamp'], reverse=True)[0]['Average']) if response['Datapoints'] else 0.0
        hostname = get_ec2_hostname(instance_id)
        if hostname:
            for metric in cwagent_metric_names:
                response = cloudwatch_client.get_metric_statistics(
                    Namespace='CWAgent', MetricName=metric, Dimensions=[{'Name': 'host', 'Value': hostname}],
                    StartTime=start_time, EndTime=now, Period=300, Statistics=['Average']
                )
                metrics_data['MemoryUtilization'] = float(sorted(response['Datapoints'], key=lambda x: x['Timestamp'], reverse=True)[0]['Average']) if response['Datapoints'] else 0.0
        else:
            metrics_data['MemoryUtilization'] = 0.0
        logger.info(f"EC2 metrics fetched: {metrics_data}")
        return metrics_data
    except Exception as e:
        logger.error(f"Error fetching EC2 metrics for {instance_id}: {str(e)}")
        return {'CPUUtilization': 0.0, 'DiskWriteOps': 0.0, 'NetworkIn': 0.0, 'MemoryUtilization': 0.0}

def fetch_rds_metrics(db_instance_id=RDS_INSTANCE_ID):
    try:
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(minutes=5)
        metrics_data = {}
        metric_names = ['CPUUtilization', 'FreeableMemory', 'DatabaseConnections', 'WriteIOPS']
        for metric in metric_names:
            response = cloudwatch_client.get_metric_statistics(
                Namespace='AWS/RDS', MetricName=metric, Dimensions=[{'Name': 'DBInstanceIdentifier', 'Value': db_instance_id}],
                StartTime=start_time, EndTime=now, Period=300, Statistics=['Average']
            )
            metrics_data[metric] = float(sorted(response['Datapoints'], key=lambda x: x['Timestamp'], reverse=True)[0]['Average']) if response['Datapoints'] else 0.0
        logger.info(f"RDS metrics fetched: {metrics_data}")
        return metrics_data
    except Exception as e:
        logger.error(f"Error fetching RDS metrics for {db_instance_id}: {str(e)}")
        return {'CPUUtilization': 0.0, 'FreeableMemory': 0.0, 'DatabaseConnections': 0.0, 'WriteIOPS': 0.0}

def fetch_ecs_metrics(cluster_name=ECS_CLUSTER_NAME):
    try:
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(minutes=5)
        metrics_data = {}
        metric_names = ['CPUUtilization', 'MemoryUtilization']
        for metric in metric_names:
            response = cloudwatch_client.get_metric_statistics(
                Namespace='AWS/ECS', MetricName=metric, Dimensions=[{'Name': 'ClusterName', 'Value': cluster_name}],
                StartTime=start_time, EndTime=now, Period=300, Statistics=['Average']
            )
            metrics_data[metric] = float(sorted(response['Datapoints'], key=lambda x: x['Timestamp'], reverse=True)[0]['Average']) if response['Datapoints'] else 0.0
        response = ecs_client.list_tasks(cluster=cluster_name, desiredStatus='RUNNING')
        task_count = len(response['taskArns'])
        metrics_data['RunningTaskCount'] = float(task_count) if task_count > 0 else 0.0
        logger.info(f"ECS metrics fetched: {metrics_data}")
        return metrics_data
    except Exception as e:
        logger.error(f"Error fetching ECS metrics for {cluster_name}: {str(e)}")
        return {'CPUUtilization': 0.0, 'MemoryUtilization': 0.0, 'RunningTaskCount': 0.0}

# Feature Engineering for Real-Time Prediction
def prepare_lstm_input(metrics, service, sequence_history):
    now = datetime.now()
    hour = float(now.hour)
    day_of_week = float(now.weekday())
    
    if service == 'EC2':
        base_features = ['CPUUtilization', 'MemoryUtilization', 'DiskWriteOps', 'NetworkIn']
        cpu = float(metrics.get('CPUUtilization', 0.0))
        mem = float(metrics.get('MemoryUtilization', 0.0))
        cpu_mem_ratio = cpu / (mem + 1e-5) if mem != 0.0 else 0.0
        cpu_vals = [h[0] if h[0] is not None else 0.0 for h in list(sequence_history)[-5:]]
        cpu_rolling_mean = sum(cpu_vals) / len(cpu_vals) if cpu_vals else cpu
        current_data = [float(metrics.get(f, 0.0)) for f in base_features] + [hour, day_of_week, cpu_mem_ratio, cpu_rolling_mean]
    elif service == 'RDS':
        base_features = ['CPUUtilization', 'FreeableMemory', 'DatabaseConnections', 'WriteIOPS']
        cpu = float(metrics.get('CPUUtilization', 0.0))
        conn = float(metrics.get('DatabaseConnections', 0.0))
        conn_per_cpu = conn / (cpu + 1e-5) if cpu != 0.0 else 0.0
        cpu_vals = [h[0] if h[0] is not None else 0.0 for h in list(sequence_history)[-5:]]
        cpu_rolling_mean = sum(cpu_vals) / len(cpu_vals) if cpu_vals else cpu
        current_data = [float(metrics.get(f, 0.0)) for f in base_features] + [hour, day_of_week, conn_per_cpu, cpu_rolling_mean]
    elif service == 'ECS':
        base_features = ['CPUUtilization', 'MemoryUtilization', 'RunningTaskCount']
        cpu = float(metrics.get('CPUUtilization', 0.0))
        mem = float(metrics.get('MemoryUtilization', 0.0))
        cpu_mem_ratio = cpu / (mem + 1e-5) if mem != 0.0 else 0.0
        task_vals = [h[2] if h[2] is not None else 0.0 for h in list(sequence_history)[-5:]]
        task_rolling_mean = sum(task_vals) / len(task_vals) if task_vals else float(metrics.get('RunningTaskCount', 0.0))
        current_data = [float(metrics.get(f, 0.0)) for f in base_features] + [hour, day_of_week, cpu_mem_ratio, task_rolling_mean]
    
    current_data = [0.0 if x is None else x for x in current_data]
    sequence_history.append(current_data)
    
    if len(sequence_history) < SEQUENCE_LENGTH:
        padded_history = [([0.0] * len(current_data)) for _ in range(SEQUENCE_LENGTH - len(sequence_history))] + list(sequence_history)
    else:
        padded_history = list(sequence_history)
    
    feature_array = np.array(padded_history, dtype=np.float32)
    input_data = []
    for t in range(SEQUENCE_LENGTH):
        time_data = feature_array[t].tolist()
        lags = []
        for lag in range(1, 4):
            if t - lag >= 0:
                lags.extend(feature_array[t - lag])
            else:
                lags.extend([0.0] * len(time_data))
        input_data.append(time_data + lags)
    
    input_array = np.array(input_data)
    logger.debug(f"{service} LSTM input shape: {input_array.shape}, Expected input_size: {EC2_INPUT_SIZE if service in ['EC2', 'RDS'] else ECS_INPUT_SIZE}")
    return input_array

# API Endpoints
@app.route('/')
def home():
    return "Flask server is running!"

@app.route('/predictions', methods=['GET'])
@require_api_key
def get_predictions():
    serializable_predictions = {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in dashboard_predictions.items()
    }
    logger.info(f"Returning predictions: {serializable_predictions}")
    return jsonify(serializable_predictions)

@app.route('/stop/ec2', methods=['POST'])
@require_api_key
def api_stop_ec2():
    logger.info(f"Received request for /stop/ec2 with data: {request.json}")
    result = stop_ec2()
    logger.info(f"Stop EC2 result: {result}")
    return jsonify({"message": result})

@app.route('/stop/rds', methods=['POST'])
@require_api_key
def api_stop_rds():
    logger.info(f"Received request for /stop/rds with data: {request.json}")
    result = stop_rds()
    logger.info(f"Stop RDS result: {result}")
    return jsonify({"message": result})

@app.route('/stop/ecs', methods=['POST'])
@require_api_key
def api_stop_ecs():
    logger.info(f"Received request for /stop/ecs with data: {request.json}")
    result = stop_ecs()
    logger.info(f"Stop ECS result: {result}")
    return jsonify({"message": result})

@app.route('/scale/ec2', methods=['POST'])
@require_api_key
def api_scale_ec2():
    logger.info(f"Received request for /scale/ec2 with data: {request.json}")
    decision = request.json.get('decision', 'no change')
    result = scale_ec2(decision)
    logger.info(f"Scale EC2 result: {result}")
    return jsonify({"message": result})

@app.route('/scale/rds', methods=['POST'])
@require_api_key
def api_scale_rds():
    logger.info(f"Received request for /scale/rds with data: {request.json}")
    decision = request.json.get('decision', 'no change')
    result = scale_rds(decision)
    logger.info(f"Scale RDS result: {result}")
    return jsonify({"message": result})

@app.route('/scale/ecs', methods=['POST'])
@require_api_key
def api_scale_ecs():
    logger.info(f"Received request for /scale/ecs with data: {request.json}")
    decision = request.json.get('decision', 'no change')
    result = scale_ecs(decision)
    logger.info(f"Scale ECS result: {result}")
    return jsonify({"message": result})

# Monitoring Thread
def monitor_and_scale():
    global dashboard_predictions, state_history, ec2_sequence_history, rds_sequence_history, ecs_sequence_history
    models_loaded = load_models()
    
    def predict_usage(model, input_data):
        if model is None or not input_data.size:
            logger.warning("No model or invalid input data for prediction")
            return 0.0
        try:
            model.eval()
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                prediction = model(input_tensor).detach().numpy()
            return float(prediction.flatten()[0])
        except Exception as e:
            logger.error(f"Error making LSTM prediction: {e}")
            return 0.0

    if not models_loaded:
        logger.error("Monitoring aborted due to model loading failure")
        return

    rl_decision_map = {0: "scale up", 1: "scale down", 2: "no change"}
    cycle_interval = MONITORING_INTERVAL

    state_history.extend([np.array([0.0, 0.0, 0.0])] * 3)
    ec2_sequence_history.extend([[0.0] * 8] * SEQUENCE_LENGTH)
    rds_sequence_history.extend([[0.0] * 8] * SEQUENCE_LENGTH)
    ecs_sequence_history.extend([[0.0] * 7] * SEQUENCE_LENGTH)

    while True:
        logger.info("Starting monitoring cycle...")
        try:
            ec2_metrics = fetch_ec2_metrics()
            rds_metrics = fetch_rds_metrics()
            ecs_metrics = fetch_ecs_metrics()
            logger.info(f"EC2 Metrics: {ec2_metrics}")
            logger.info(f"RDS Metrics: {rds_metrics}")
            logger.info(f"ECS Metrics: {ecs_metrics}")

            ec2_input = prepare_lstm_input(ec2_metrics, 'EC2', ec2_sequence_history)
            rds_input = prepare_lstm_input(rds_metrics, 'RDS', rds_sequence_history)
            ecs_input = prepare_lstm_input(ecs_metrics, 'ECS', ecs_sequence_history)

            ec2_pred = predict_usage(ec2_lstm_model, ec2_input)
            rds_pred = predict_usage(rds_lstm_model, rds_input)
            ecs_pred = predict_usage(ecs_lstm_model, ecs_input)

            ec2_cpu = float(ec2_metrics.get('CPUUtilization', 0.0)) / 100.0
            rds_cpu = float(rds_metrics.get('CPUUtilization', 0.0)) / 100.0
            ecs_cpu = float(ecs_metrics.get('CPUUtilization', 0.0)) / 100.0
            current_state = np.array([ec2_cpu, rds_cpu, ecs_cpu])
            state_history.append(current_state)
            rl_state = np.concatenate(list(state_history)).flatten()

            rl_input = torch.tensor(rl_state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_ec2, q_rds, q_ecs = rl_model(rl_input)
            ec2_decision = rl_decision_map[torch.argmax(q_ec2, dim=1).item()]
            rds_decision = rl_decision_map[torch.argmax(q_rds, dim=1).item()]
            ecs_decision = rl_decision_map[torch.argmax(q_ecs, dim=1).item()]
            logger.info(f"Decisions - EC2: {ec2_decision}, RDS: {rds_decision}, ECS: {ecs_decision}")

            scale_ec2(ec2_decision)
            scale_rds(rds_decision)
            scale_ecs(ec2_decision)

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
            logger.info(f"Dashboard Predictions: {dashboard_predictions}")
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {str(e)}")
        
        logger.info(f"Next cycle in {cycle_interval} seconds")
        for remaining in range(cycle_interval, 0, -1):
            minutes, seconds = divmod(remaining, 60)
            print(f"  {minutes:02d}:{seconds:02d}", end="\r")
            time.sleep(1)
        print(" " * 20)

monitoring_thread = threading.Thread(target=monitor_and_scale, daemon=True)
monitoring_thread.start()

if __name__ == '__main__':
    create_metrics_table()
    app.run(host=HOST, port=PORT, debug=False)