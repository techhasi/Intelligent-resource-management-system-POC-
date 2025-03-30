from flask import Flask, jsonify, request, abort
from flask_cors import CORS
import torch
import torch.nn as nn
import boto3
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
import pickle

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

# Updated Model Paths - point to the newly created models
EC2_MODEL_PATH = '../models/LSTMv5/ec2_lstm_model_improved.pth'  # Updated path for new EC2 LSTM model
RDS_MODEL_PATH = '../models/LSTMv5/rds_lstm_model_improved.pth'  # Updated path for new RDS LSTM model 
ECS_MODEL_PATH = '../models/LSTMv5/ecs_lstm_model_improved.pth'  # Updated path for new ECS LSTM model
RL_MODEL_PATH = '../models/RLv5/enhanced_cloud_q_best.pkl'     # Updated path for new RL model

# Model Parameters
SEQUENCE_LENGTH = 24  # Updated to match LSTMv5.py
LSTM_HIDDEN_SIZE = 256  # Updated to match LSTMv5.py
LSTM_NUM_LAYERS = 3    # Updated to match LSTMv5.py 
LSTM_DROPOUT = 0.3     # Updated to match LSTMv5.py
BINS_PER_DIM = 8       # From RLv5.py

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

# Updated AttentionLSTM Model Definition - based on LSTMv5.py
class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Global Variables
ec2_lstm_model = None
rds_lstm_model = None
ecs_lstm_model = None
rl_model = None
rl_scaler = None
dashboard_predictions = {}
state_history = deque(maxlen=3)  # For RL input state
ec2_sequence_history = deque(maxlen=SEQUENCE_LENGTH)
rds_sequence_history = deque(maxlen=SEQUENCE_LENGTH)
ecs_sequence_history = deque(maxlen=SEQUENCE_LENGTH)

# Model Loading - Updated to handle the new model formats
def load_models():
    global ec2_lstm_model, rds_lstm_model, ecs_lstm_model, rl_model
    try:
        # Load EC2 LSTM model
        ec2_checkpoint = torch.load(EC2_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        ec2_input_size = ec2_checkpoint.get('input_size', 32)
        ec2_hidden_size = ec2_checkpoint.get('hidden_size', LSTM_HIDDEN_SIZE)
        ec2_num_layers = ec2_checkpoint.get('num_layers', LSTM_NUM_LAYERS)
        
        ec2_lstm_model = AttentionLSTM(
            input_size=ec2_input_size, 
            hidden_size=ec2_hidden_size, 
            num_layers=ec2_num_layers, 
            output_size=1, 
            dropout=LSTM_DROPOUT
        )
        ec2_lstm_model.load_state_dict(ec2_checkpoint['model_state_dict'])
        ec2_lstm_model.eval()
        logger.info(f"EC2 LSTM model loaded: input_size={ec2_input_size}, hidden_size={ec2_hidden_size}")
        
        # Load RDS LSTM model
        rds_checkpoint = torch.load(RDS_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        rds_input_size = rds_checkpoint.get('input_size', 32)
        rds_hidden_size = rds_checkpoint.get('hidden_size', LSTM_HIDDEN_SIZE)
        rds_num_layers = rds_checkpoint.get('num_layers', LSTM_NUM_LAYERS)
        
        rds_lstm_model = AttentionLSTM(
            input_size=rds_input_size, 
            hidden_size=rds_hidden_size, 
            num_layers=rds_num_layers, 
            output_size=1, 
            dropout=LSTM_DROPOUT
        )
        rds_lstm_model.load_state_dict(rds_checkpoint['model_state_dict'])
        rds_lstm_model.eval()
        logger.info(f"RDS LSTM model loaded: input_size={rds_input_size}, hidden_size={rds_hidden_size}")
        
        # Load ECS LSTM model
        ecs_checkpoint = torch.load(ECS_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        ecs_input_size = ecs_checkpoint.get('input_size', 28)
        ecs_hidden_size = ecs_checkpoint.get('hidden_size', LSTM_HIDDEN_SIZE)
        ecs_num_layers = ecs_checkpoint.get('num_layers', LSTM_NUM_LAYERS)
        
        ecs_lstm_model = AttentionLSTM(
            input_size=ecs_input_size, 
            hidden_size=ecs_hidden_size, 
            num_layers=ecs_num_layers, 
            output_size=1, 
            dropout=LSTM_DROPOUT
        )
        ecs_lstm_model.load_state_dict(ecs_checkpoint['model_state_dict'])
        ecs_lstm_model.eval()
        logger.info(f"ECS LSTM model loaded: input_size={ecs_input_size}, hidden_size={ecs_hidden_size}")
        
        # Load RL model - using pickle as in RLv5.py
        with open(RL_MODEL_PATH, 'rb') as f:
            rl_model_data = pickle.load(f)
            
        # Create EnhancedQAgent as defined in RLv5.py
        from collections import defaultdict
        class EnhancedQAgent:
            def __init__(self, bins_per_dim=BINS_PER_DIM, action_size_per_service=3, num_services=3):
                self.bins_per_dim = bins_per_dim
                self.action_size_per_service = action_size_per_service
                self.num_services = num_services
                self.q_table = defaultdict(float)
                self.epsilon = 0.0  # No exploration during production
            
            def load_model(self, data):
                self.q_table = defaultdict(float, data['q_table'])
                self.epsilon = 0.0  # Force no exploration
                return True
                
            def get_q_value(self, state_tuple, service_idx, action):
                key = (state_tuple, service_idx, action)
                return self.q_table[key]
            
            def select_action(self, state):
                state_tuple = self.safe_discretize_state(state)
                actions = {}
                for i in range(self.num_services):
                    q_values = [self.get_q_value(state_tuple, i, a) for a in range(3)]
                    best_action = np.argmax(q_values)
                    actions[i] = best_action
                return actions
            
            def safe_discretize_state(self, state):
                discrete_state = []
                try:
                    # Resource counts - finer discretization up to 10 instances
                    for i in range(3):  # 3 services
                        if i < len(state) and not np.isnan(state[i]):
                            resource_count = min(int(state[i]), 10)
                            bin_idx = min(int((resource_count - 1) * self.bins_per_dim / 10), self.bins_per_dim - 1)
                            discrete_state.append(bin_idx)
                        else:
                            discrete_state.append(1)  # Default to low-mid range
                    
                    # CPU utilization (normalized to [0,1]) - use finer discretization
                    for i in range(3, 9):  # CPU utilization dimensions
                        if i < len(state) and not np.isnan(state[i]):
                            value = max(0.0, min(1.0, state[i]))
                            bin_idx = min(int(value * self.bins_per_dim), self.bins_per_dim - 1)
                            discrete_state.append(bin_idx)
                        else:
                            discrete_state.append(self.bins_per_dim // 2)
                    
                    # Add time of day awareness (if available)
                    if len(state) > 9 and not np.isnan(state[9]):
                        hour = int(state[9]) % 24
                        time_bin = hour // 6
                        discrete_state.append(time_bin)
                    else:
                        discrete_state.append(2)
                    
                    # Add day of week awareness (if available)
                    if len(state) > 10 and not np.isnan(state[10]):
                        day = int(state[10]) % 7
                        day_bin = 1 if day >= 5 else 0
                        discrete_state.append(day_bin)
                    else:
                        discrete_state.append(0)
                        
                except Exception as e:
                    logger.error(f"Error in discretize_state: {e}")
                    return tuple([1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0])
                
                return tuple(discrete_state)
            
        rl_model = EnhancedQAgent()
        rl_model.load_model(rl_model_data)
        logger.info("RL model (Q-Agent) loaded successfully")
        
        # Also load feature and target scalers for each LSTM model if available
        global ec2_feature_scaler, ec2_target_scaler
        global rds_feature_scaler, rds_target_scaler
        global ecs_feature_scaler, ecs_target_scaler
        
        ec2_feature_scaler = ec2_checkpoint.get('feature_scaler', None)
        ec2_target_scaler = ec2_checkpoint.get('target_scaler', None)
        rds_feature_scaler = rds_checkpoint.get('feature_scaler', None)
        rds_target_scaler = rds_checkpoint.get('target_scaler', None)
        ecs_feature_scaler = ecs_checkpoint.get('feature_scaler', None)
        ecs_target_scaler = ecs_checkpoint.get('target_scaler', None)
        
        logger.info("All models loaded successfully")
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
            
            # Add a table to track scaling decisions and their outcomes
            cursor.execute('''CREATE TABLE IF NOT EXISTS scaling_decisions (
                                timestamp DATETIME,
                                service TEXT,
                                decision TEXT,
                                cpu_before REAL,
                                cpu_after REAL,
                                success BOOLEAN
                            )''')
            conn.commit()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database tables: {e}")

# EC2 Hostname
def get_ec2_hostname(instance_id):
    try:
        response = ec2_client.describe_instances(InstanceIds=[instance_id])
        hostname = response['Reservations'][0]['Instances'][0]['PrivateDnsName']
        return hostname
    except Exception as e:
        logger.error(f"Error fetching hostname for instance {instance_id}: {str(e)}")
        return None

# Scaling Functions - Using updated logic from the RL agent
def scale_ec2(decision, asg_name=EC2_ASG_NAME):
    """
    Scale EC2 instances based on RL decision
    decision: "scale up", "scale down", or "no change"
    """
    try:
        # Log CPU before scaling
        ec2_metrics = fetch_ec2_metrics()
        cpu_before = ec2_metrics.get('CPUUtilization', 0.0)
        
        response = autoscaling_client.describe_auto_scaling_groups(AutoScalingGroupNames=[asg_name])
        current_capacity = response['AutoScalingGroups'][0]['DesiredCapacity']
        
        if decision == "scale up":
            new_capacity = current_capacity + 1
            autoscaling_client.set_desired_capacity(AutoScalingGroupName=asg_name, DesiredCapacity=new_capacity)
            logger.info(f"Scaled EC2 ASG {asg_name} up to {new_capacity} instances")
            result = f"Scaled EC2 ASG {asg_name} up to {new_capacity} instances"
            success = True
        elif decision == "scale down" and current_capacity > 1:
            new_capacity = current_capacity - 1
            autoscaling_client.set_desired_capacity(AutoScalingGroupName=asg_name, DesiredCapacity=new_capacity)
            logger.info(f"Scaled EC2 ASG {asg_name} down to {new_capacity} instances")
            result = f"Scaled EC2 ASG {asg_name} down to {new_capacity} instances"
            success = True
        else:
            logger.info(f"No scaling action for EC2 ASG {asg_name}")
            result = f"No scaling action for EC2 ASG {asg_name}"
            success = True
        
        # Wait to measure effect
        time.sleep(5)
        ec2_metrics_after = fetch_ec2_metrics()
        cpu_after = ec2_metrics_after.get('CPUUtilization', 0.0)
        
        # Record decision and outcome
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO scaling_decisions VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now(), "EC2", decision, cpu_before, cpu_after, success)
            )
            conn.commit()
            
        return result
    except ClientError as e:
        logger.error(f"AWS ClientError scaling EC2 ASG {asg_name}: {e}")
        # Record failure
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO scaling_decisions VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now(), "EC2", decision, cpu_before, cpu_before, False)
            )
            conn.commit()
        return f"AWS Error: {e}"
    except Exception as e:
        logger.error(f"Error scaling EC2 ASG {asg_name}: {str(e)}")
        return f"Error scaling EC2 ASG {asg_name}: {str(e)}"

def scale_rds(decision, db_instance_id=RDS_INSTANCE_ID):
    """
    Scale RDS instance based on RL decision
    decision: "scale up", "scale down", or "no change"
    """
    try:
        # Log CPU before scaling
        rds_metrics = fetch_rds_metrics()
        cpu_before = rds_metrics.get('CPUUtilization', 0.0)
        
        response = rds_client.describe_db_instances(DBInstanceIdentifier=db_instance_id)
        current_class = response['DBInstances'][0]['DBInstanceClass']
        instance_classes = ['db.t3.micro', 'db.t3.small', 'db.t3.medium', 'db.t3.large', 'db.r5.large']
        current_idx = instance_classes.index(current_class) if current_class in instance_classes else 0
        
        if decision == "scale up" and current_idx < len(instance_classes) - 1:
            new_class = instance_classes[current_idx + 1]
            rds_client.modify_db_instance(DBInstanceIdentifier=db_instance_id, DBInstanceClass=new_class, ApplyImmediately=True)
            logger.info(f"Scaled RDS {db_instance_id} up to {new_class}")
            result = f"Scaled RDS {db_instance_id} up to {new_class}"
            success = True
        elif decision == "scale down" and current_idx > 0:
            new_class = instance_classes[current_idx - 1]
            rds_client.modify_db_instance(DBInstanceIdentifier=db_instance_id, DBInstanceClass=new_class, ApplyImmediately=True)
            logger.info(f"Scaled RDS {db_instance_id} down to {new_class}")
            result = f"Scaled RDS {db_instance_id} down to {new_class}"
            success = True
        else:
            logger.info(f"No scaling action for RDS {db_instance_id}")
            result = f"No scaling action for RDS {db_instance_id}"
            success = True
        
        # Wait to measure effect (though RDS scaling is usually not immediate)
        time.sleep(5)
        rds_metrics_after = fetch_rds_metrics()
        cpu_after = rds_metrics_after.get('CPUUtilization', 0.0)
        
        # Record decision and outcome
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO scaling_decisions VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now(), "RDS", decision, cpu_before, cpu_after, success)
            )
            conn.commit()
            
        return result
    except ClientError as e:
        logger.error(f"AWS ClientError scaling RDS {db_instance_id}: {e}")
        # Record failure
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO scaling_decisions VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now(), "RDS", decision, cpu_before, cpu_before, False)
            )
            conn.commit()
        return f"AWS Error: {e}"
    except Exception as e:
        logger.error(f"Error scaling RDS {db_instance_id}: {str(e)}")
        return f"Error scaling RDS {db_instance_id}: {str(e)}"

def scale_ecs(decision, cluster_name=ECS_CLUSTER_NAME, task_definition=ECS_TASK_DEFINITION):
    """
    Scale ECS tasks based on RL decision
    decision: "scale up", "scale down", or "no change"
    """
    try:
        # Log CPU before scaling
        ecs_metrics = fetch_ecs_metrics()
        cpu_before = ecs_metrics.get('CPUUtilization', 0.0)
        
        response = ecs_client.list_tasks(cluster=cluster_name, desiredStatus='RUNNING')
        current_count = len(response['taskArns'])
        
        if decision == "scale up":
            # Use RunTask API to start a new task
            response = ecs_client.run_task(
                cluster=cluster_name, taskDefinition=task_definition, count=1, launchType='FARGATE',
                networkConfiguration={'awsvpcConfiguration': {'subnets': ECS_SUBNETS, 'securityGroups': ECS_SECURITY_GROUPS, 'assignPublicIp': 'ENABLED'}}
            )
            if response.get('tasks'):
                task_arn = response['tasks'][0]['taskArn']
                logger.info(f"Started new task {task_arn} in cluster {cluster_name}")
                result = f"Started new task {task_arn} in cluster {cluster_name}"
                success = True
            else:
                logger.error(f"Failed to start new task: {response.get('failures')}")
                result = f"Failed to start new task: {response.get('failures')}"
                success = False
        elif decision == "scale down" and current_count > 1:
            task_to_stop = response['taskArns'][0]
            ecs_client.stop_task(cluster=cluster_name, task=task_to_stop, reason='Scaling down through RL agent')
            logger.info(f"Stopped task {task_to_stop} in cluster {cluster_name}")
            result = f"Stopped task {task_to_stop} in cluster {cluster_name}"
            success = True
        else:
            logger.info(f"No scaling action for cluster {cluster_name}")
            result = f"No scaling action for cluster {cluster_name}"
            success = True
        
        # Wait to measure effect
        time.sleep(5)
        ecs_metrics_after = fetch_ecs_metrics()
        cpu_after = ecs_metrics_after.get('CPUUtilization', 0.0)
        
        # Record decision and outcome
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO scaling_decisions VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now(), "ECS", decision, cpu_before, cpu_after, success)
            )
            conn.commit()
            
        return result
    except ClientError as e:
        logger.error(f"AWS ClientError scaling ECS cluster {cluster_name}: {e}")
        # Record failure
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO scaling_decisions VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now(), "ECS", decision, cpu_before, cpu_before, False)
            )
            conn.commit()
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
            except Exception:
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

# Fetch Metrics - using the CloudWatch API
# Fetch Metrics - using the CloudWatch API
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
        
        # Store metrics in database
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            for metric_name, metric_value in metrics_data.items():
                cursor.execute(
                    "INSERT INTO metrics VALUES (?, ?, ?, ?, ?)",
                    (datetime.now(), 'EC2', instance_id, metric_name, metric_value)
                )
            conn.commit()
            
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
        
        # Store metrics in database
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            for metric_name, metric_value in metrics_data.items():
                cursor.execute(
                    "INSERT INTO metrics VALUES (?, ?, ?, ?, ?)",
                    (datetime.now(), 'RDS', db_instance_id, metric_name, metric_value)
                )
            conn.commit()
            
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
        
        # Get running task count
        response = ecs_client.list_tasks(cluster=cluster_name, desiredStatus='RUNNING')
        task_count = len(response['taskArns'])
        metrics_data['RunningTaskCount'] = float(task_count) if task_count > 0 else 0.0
        
        # Add network metrics if available
        try:
            for network_metric in ['NetworkRxBytes', 'NetworkTxBytes']:
                response = cloudwatch_client.get_metric_statistics(
                    Namespace='AWS/ECS', MetricName=network_metric, Dimensions=[{'Name': 'ClusterName', 'Value': cluster_name}],
                    StartTime=start_time, EndTime=now, Period=300, Statistics=['Average']
                )
                metrics_data[network_metric] = float(sorted(response['Datapoints'], key=lambda x: x['Timestamp'], reverse=True)[0]['Average']) if response['Datapoints'] else 0.0
        except Exception as network_error:
            logger.warning(f"Could not fetch ECS network metrics: {network_error}")
            metrics_data['NetworkRxBytes'] = 0.0
            metrics_data['NetworkTxBytes'] = 0.0
            
        # Store metrics in database
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            for metric_name, metric_value in metrics_data.items():
                cursor.execute(
                    "INSERT INTO metrics VALUES (?, ?, ?, ?, ?)",
                    (datetime.now(), 'ECS', cluster_name, metric_name, metric_value)
                )
            conn.commit()
            
        logger.info(f"ECS metrics fetched: {metrics_data}")
        return metrics_data
    except Exception as e:
        logger.error(f"Error fetching ECS metrics for {cluster_name}: {str(e)}")
        return {'CPUUtilization': 0.0, 'MemoryUtilization': 0.0, 'RunningTaskCount': 0.0, 'NetworkRxBytes': 0.0, 'NetworkTxBytes': 0.0}

def prepare_lstm_input(metrics, service, sequence_history):
    """
    Prepare input data to match exactly the features used during model training
    """
    now = datetime.now()
    hour = float(now.hour)
    day_of_week = float(now.weekday())
    day_of_month = float(now.day)
    
    # Feature engineering to match training data
    if service == 'EC2':
        # Base metrics from CloudWatch
        cpu = float(metrics.get('CPUUtilization', 0.0))
        mem = float(metrics.get('MemoryUtilization', 0.0))
        disk = float(metrics.get('DiskWriteOps', 0.0))
        network = float(metrics.get('NetworkIn', 0.0))
        
        # Get historical CPU for lag features
        cpu_history = []
        if len(sequence_history) > 0:
            for entry in list(sequence_history):
                if isinstance(entry, (list, tuple)) and len(entry) > 0:
                    cpu_history.append(entry[0] if entry[0] is not None else 0.0)
        
        # Ensure we have at least 3 values for lag calculation
        while len(cpu_history) < 3:
            cpu_history.insert(0, cpu)
            
        # Calculate features matching training data
        cpu_lag_1 = cpu_history[-1] if len(cpu_history) > 0 else cpu
        cpu_lag_2 = cpu_history[-2] if len(cpu_history) > 1 else cpu
        cpu_diff_1 = cpu - cpu_lag_1
        cpu_diff_3 = cpu - cpu_history[-3] if len(cpu_history) > 2 else 0.0
        
        # Ratio calculations
        cpu_mem_ratio = cpu / (mem + 1e-8)
        cpu_disk_ratio = cpu / (disk + 1e-8) 
        cpu_network_ratio = cpu / (network + 1e-8)
        
        # Rolling statistics from recent history
        cpu_roll_6h = np.mean(cpu_history) if cpu_history else cpu
        cpu_std_6h = np.std(cpu_history) if len(cpu_history) > 1 else 0.0
        
        # Quantiles
        cpu_q25 = np.percentile(cpu_history, 25) if len(cpu_history) >= 4 else cpu * 0.8
        cpu_q75 = np.percentile(cpu_history, 75) if len(cpu_history) >= 4 else cpu * 1.2
        
        # Time encoding
        day_of_month_sin = np.sin(2 * np.pi * day_of_month / 31)
        
        # Assemble features in exact order matching training data
        current_data = [
            cpu_lag_1, cpu_diff_1, cpu_disk_ratio, cpu_diff_3, 
            cpu_mem_ratio, cpu_network_ratio, disk, mem, 
            network, cpu_std_6h, cpu_roll_6h, cpu_lag_2, 
            cpu_q25, cpu_q75, day_of_month_sin, cpu
        ]
    
    elif service == 'RDS':
        # Assuming RDS uses similar features to EC2 based on the feature list
        cpu = float(metrics.get('CPUUtilization', 0.0))
        memory = float(metrics.get('FreeableMemory', 0.0)) 
        conn = float(metrics.get('DatabaseConnections', 0.0))
        io = float(metrics.get('WriteIOPS', 0.0))
        
        # Get historical CPU for lag features
        cpu_history = []
        if len(sequence_history) > 0:
            for entry in list(sequence_history):
                if isinstance(entry, (list, tuple)) and len(entry) > 0:
                    cpu_history.append(entry[0] if entry[0] is not None else 0.0)
        
        # Ensure we have at least 3 values for lag calculation
        while len(cpu_history) < 3:
            cpu_history.insert(0, cpu)
            
        # Calculate features matching training data (similar to EC2)
        cpu_lag_1 = cpu_history[-1] if len(cpu_history) > 0 else cpu
        cpu_lag_2 = cpu_history[-2] if len(cpu_history) > 1 else cpu
        cpu_diff_1 = cpu - cpu_lag_1
        cpu_diff_3 = cpu - cpu_history[-3] if len(cpu_history) > 2 else 0.0
        
        # Ratio calculations
        memory_usage = 100 - (memory / 1000)  # Approximate - adjust based on your data
        cpu_mem_ratio = cpu / (memory_usage + 1e-8)
        cpu_io_ratio = cpu / (io + 1e-8)  # Similar to disk ratio
        cpu_conn_ratio = cpu / (conn + 1e-8)  # Similar to network ratio
        
        # Rolling statistics from recent history
        cpu_roll_6h = np.mean(cpu_history) if cpu_history else cpu
        cpu_std_6h = np.std(cpu_history) if len(cpu_history) > 1 else 0.0
        
        # Quantiles
        cpu_q25 = np.percentile(cpu_history, 25) if len(cpu_history) >= 4 else cpu * 0.8
        cpu_q75 = np.percentile(cpu_history, 75) if len(cpu_history) >= 4 else cpu * 1.2
        
        # Time encoding
        day_of_month_sin = np.sin(2 * np.pi * day_of_month / 31)
        
        # Assemble features in exact order matching training data
        current_data = [
            cpu_lag_1, cpu_diff_1, cpu_io_ratio, cpu_diff_3, 
            cpu_mem_ratio, cpu_conn_ratio, io, memory, 
            conn, cpu_std_6h, cpu_roll_6h, cpu_lag_2, 
            cpu_q25, cpu_q75, day_of_month_sin, cpu
        ]
    
    elif service == 'ECS':
        # Assuming ECS uses similar features to EC2 based on the feature list
        cpu = float(metrics.get('CPUUtilization', 0.0))
        mem = float(metrics.get('MemoryUtilization', 0.0))
        tasks = float(metrics.get('RunningTaskCount', 0.0))
        network = float(metrics.get('NetworkRxBytes', 0.0))
        
        # Get historical CPU for lag features
        cpu_history = []
        if len(sequence_history) > 0:
            for entry in list(sequence_history):
                if isinstance(entry, (list, tuple)) and len(entry) > 0:
                    cpu_history.append(entry[0] if entry[0] is not None else 0.0)
        
        # Ensure we have at least 3 values for lag calculation
        while len(cpu_history) < 3:
            cpu_history.insert(0, cpu)
            
        # Calculate features matching training data (similar to EC2)
        cpu_lag_1 = cpu_history[-1] if len(cpu_history) > 0 else cpu
        cpu_lag_2 = cpu_history[-2] if len(cpu_history) > 1 else cpu
        cpu_diff_1 = cpu - cpu_lag_1
        cpu_diff_3 = cpu - cpu_history[-3] if len(cpu_history) > 2 else 0.0
        
        # Ratio calculations
        cpu_mem_ratio = cpu / (mem + 1e-8)
        cpu_task_ratio = cpu / (tasks + 1e-8)  # Similar to disk ratio
        cpu_network_ratio = cpu / (network + 1e-8)
        
        # Rolling statistics from recent history
        cpu_roll_6h = np.mean(cpu_history) if cpu_history else cpu
        cpu_std_6h = np.std(cpu_history) if len(cpu_history) > 1 else 0.0
        
        # Quantiles
        cpu_q25 = np.percentile(cpu_history, 25) if len(cpu_history) >= 4 else cpu * 0.8
        cpu_q75 = np.percentile(cpu_history, 75) if len(cpu_history) >= 4 else cpu * 1.2
        
        # Time encoding
        day_of_month_sin = np.sin(2 * np.pi * day_of_month / 31)
        
        # Assemble features in exact order matching training data
        current_data = [
            cpu_lag_1, cpu_diff_1, cpu_task_ratio, cpu_diff_3, 
            cpu_mem_ratio, cpu_network_ratio, tasks, mem, 
            network, cpu_std_6h, cpu_roll_6h, cpu_lag_2, 
            cpu_q25, cpu_q75, day_of_month_sin, cpu
        ]
    
    # Ensure no None or NaN values
    current_data = [0.0 if x is None or np.isnan(x) else float(x) for x in current_data]
    
    # Add to sequence history
    sequence_history.append(current_data[:]) # Create a copy
    
    # Create input tensor from a single timestep
    # We don't need to build a sequence with lag features - the model was trained using 
    # features that already include lags as individual columns
    input_array = np.array([current_data], dtype=np.float32)
    
    logger.debug(f"{service} LSTM input shape: {input_array.shape}")
    return input_array

# API Endpoints
@app.route('/')
def home():
    return "Intelligent Cloud Resource Management API is running!"

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": all([ec2_lstm_model, rds_lstm_model, ecs_lstm_model, rl_model]) 
    })

@app.route('/predictions', methods=['GET'])
@require_api_key
def get_predictions():
    serializable_predictions = {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in dashboard_predictions.items()
    }
    logger.info(f"Returning predictions: {serializable_predictions}")
    return jsonify(serializable_predictions)

@app.route('/metrics', methods=['GET'])
@require_api_key
def get_metrics():
    try:
        ec2_metrics = fetch_ec2_metrics()
        rds_metrics = fetch_rds_metrics()
        ecs_metrics = fetch_ecs_metrics()
        
        return jsonify({
            "ec2": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in ec2_metrics.items()},
            "rds": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in rds_metrics.items()},
            "ecs": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in ecs_metrics.items()},
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({"error": str(e)}), 500

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

# Monitoring Thread - Updated to match new model formats
def monitor_and_scale():
    global dashboard_predictions, state_history, ec2_sequence_history, rds_sequence_history, ecs_sequence_history
    
    import traceback
    
    def predict_usage(model, input_data, target_scaler=None):
        """
        Make LSTM predictions and denormalize if target_scaler is provided
        """
        if model is None or not input_data.size:
            logger.warning("No model or invalid input data for prediction")
            return 0.0
        try:
            model.eval()
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                prediction = model(input_tensor).detach().numpy()
            
            # If we have a target scaler, use it to convert the prediction back to original scale
            if target_scaler is not None:
                prediction = target_scaler.inverse_transform(prediction)
            
            return float(prediction.flatten()[0])
        except Exception as e:
            logger.error(f"Error making LSTM prediction: {e}")
            logger.error(traceback.format_exc())
            return 0.0

    def safe_discretize_state(state):
        """
        Discretize the continuous state space for RL model
        This mirrors the function in RLv5.py
        """
        discrete_state = []
        
        try:
            # Resource counts - finer discretization up to 10 instances
            for i in range(3):  # 3 services
                if i < len(state) and not np.isnan(state[i]):
                    resource_count = min(int(state[i]), 10)
                    bin_idx = min(int((resource_count - 1) * BINS_PER_DIM / 10), BINS_PER_DIM - 1)
                    discrete_state.append(bin_idx)
                else:
                    discrete_state.append(1)  # Default to low-mid range
            
            # CPU utilization (normalized to [0,1]) - use finer discretization
            for i in range(3, 9):  # CPU utilization dimensions
                if i < len(state) and not np.isnan(state[i]):
                    value = max(0.0, min(1.0, state[i]))
                    bin_idx = min(int(value * BINS_PER_DIM), BINS_PER_DIM - 1)
                    discrete_state.append(bin_idx)
                else:
                    discrete_state.append(BINS_PER_DIM // 2)
            
            # Add time of day awareness (if available)
            if len(state) > 9 and not np.isnan(state[9]):
                hour = int(state[9]) % 24
                time_bin = hour // 6
                discrete_state.append(time_bin)
            else:
                discrete_state.append(2)
            
            # Add day of week awareness (if available)
            if len(state) > 10 and not np.isnan(state[10]):
                day = int(state[10]) % 7
                day_bin = 1 if day >= 5 else 0
                discrete_state.append(day_bin)
            else:
                discrete_state.append(0)
                
        except Exception as e:
            logger.error(f"Error in discretize_state: {e}")
            logger.error(traceback.format_exc())
            # Return a safe default state if any error occurs
            return tuple([1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0])
        
        return tuple(discrete_state)

    # Load models if not already loaded
    models_loaded = load_models()

    if not models_loaded:
        logger.error("Monitoring aborted due to model loading failure")
        return

    # For the new RL model from RLv5.py, the action mapping is:
    # 0: Scale down, 1: No change, 2: Scale up (different from original mapping)
    rl_decision_map = {0: "scale down", 1: "no change", 2: "scale up"}
    cycle_interval = MONITORING_INTERVAL

    # Initialize history structures with zeros
    # For RL model state
    state_history.extend([np.array([0.0, 0.0, 0.0])] * 3)
    
    # For LSTM sequence history - updated for more features
    ec2_sequence_history.extend([[0.0] * 13] * SEQUENCE_LENGTH)  # Base + engineered features
    rds_sequence_history.extend([[0.0] * 13] * SEQUENCE_LENGTH)  # Base + engineered features
    ecs_sequence_history.extend([[0.0] * 13] * SEQUENCE_LENGTH)  # Base + engineered features

    while True:
        logger.info("Starting monitoring cycle...")
        try:
            # Fetch current metrics
            ec2_metrics = fetch_ec2_metrics()
            rds_metrics = fetch_rds_metrics()
            ecs_metrics = fetch_ecs_metrics()
            logger.info(f"EC2 Metrics: {ec2_metrics}")
            logger.info(f"RDS Metrics: {rds_metrics}")
            logger.info(f"ECS Metrics: {ecs_metrics}")

            # Prepare input for LSTM models with the updated feature engineering
            ec2_input = prepare_lstm_input(ec2_metrics, 'EC2', ec2_sequence_history)
            rds_input = prepare_lstm_input(rds_metrics, 'RDS', rds_sequence_history)
            ecs_input = prepare_lstm_input(ecs_metrics, 'ECS', ecs_sequence_history)

            # Make predictions using LSTM models with target scaling
            ec2_pred = predict_usage(ec2_lstm_model, ec2_input, ec2_target_scaler)
            rds_pred = predict_usage(rds_lstm_model, rds_input, rds_target_scaler)
            ecs_pred = predict_usage(ecs_lstm_model, ecs_input, ecs_target_scaler)
            
            # Log predictions
            logger.info(f"LSTM Predictions - EC2: {ec2_pred:.2f}, RDS: {rds_pred:.2f}, ECS: {ecs_pred:.2f}")

            # Prepare state for the RL agent
            # Get current resource counts from ASG and current services
            try:
                # Get EC2 instance count
                ec2_response = autoscaling_client.describe_auto_scaling_groups(AutoScalingGroupNames=[EC2_ASG_NAME])
                ec2_count = float(ec2_response['AutoScalingGroups'][0]['DesiredCapacity'])
                
                # Get RDS instance type as a numeric value (0=micro, 1=small, 2=medium, etc.)
                rds_response = rds_client.describe_db_instances(DBInstanceIdentifier=RDS_INSTANCE_ID)
                current_class = rds_response['DBInstances'][0]['DBInstanceClass']
                instance_classes = ['db.t3.micro', 'db.t3.small', 'db.t3.medium', 'db.t3.large', 'db.r5.large']
                rds_count = float(instance_classes.index(current_class) if current_class in instance_classes else 0)
                
                # Get ECS task count
                ecs_response = ecs_client.list_tasks(cluster=ECS_CLUSTER_NAME, desiredStatus='RUNNING')
                ecs_count = float(len(ecs_response['taskArns']))
            except Exception as e:
                logger.error(f"Error getting resource counts: {e}")
                logger.error(traceback.format_exc())
                # Default values if we can't get actual counts
                ec2_count, rds_count, ecs_count = 2.0, 1.0, 2.0
            
            # Normalize CPU values for state
            ec2_cpu = float(ec2_metrics.get('CPUUtilization', 0.0)) / 100.0
            rds_cpu = float(rds_metrics.get('CPUUtilization', 0.0)) / 100.0
            ecs_cpu = float(ecs_metrics.get('CPUUtilization', 0.0)) / 100.0
            
            # Current raw state includes resource counts and CPU values
            current_state = np.array([
                ec2_count, rds_count, ecs_count,  # Resource counts
                ec2_cpu, rds_cpu, ecs_cpu,        # Current CPU values
                ec2_pred/100.0, rds_pred/100.0, ecs_pred/100.0,  # Predicted CPU values (normalized)
                float(datetime.now().hour),        # Current hour
                float(datetime.now().weekday())    # Current day of week
            ])
            
            # Update state history
            state_history.append(current_state[:3])  # Only keep resource counts in history
            
            # Get discrete state for Q-table lookup (for the updated RL model)
            # This function comes from RLv5.py
            discrete_state = safe_discretize_state(current_state)
            
            # Select action based on the Q-table (for the updated RL model)
            actions = rl_model.select_action(current_state)
            
            # Convert actions to decisions
            ec2_decision = rl_decision_map[actions[0]]
            rds_decision = rl_decision_map[actions[1]]
            ecs_decision = rl_decision_map[actions[2]]
            
            logger.info(f"RL Decisions - EC2: {ec2_decision}, RDS: {rds_decision}, ECS: {ecs_decision}")

            # Execute scaling decisions
            ec2_result = scale_ec2(ec2_decision)
            rds_result = scale_rds(rds_decision)
            ecs_result = scale_ecs(ecs_decision)
            
            logger.info(f"Scaling Results - EC2: {ec2_result}, RDS: {rds_result}, ECS: {ecs_result}")

            # Current resource utilization
            ec2_utilization = ec2_cpu * 100.0 / ec2_count if ec2_count > 0 else 100.0
            rds_utilization = rds_cpu * 100.0 / (rds_count + 1) if rds_count >= 0 else 100.0
            ecs_utilization = ecs_cpu * 100.0 / ecs_count if ecs_count > 0 else 100.0

            # Update the dashboard predictions for the UI
            dashboard_predictions = {
                "timestamp": datetime.now().isoformat(),
                # Current metrics
                "EC2_CPU": round(ec2_metrics.get('CPUUtilization', 0.0), 2),
                "RDS_CPU": round(rds_metrics.get('CPUUtilization', 0.0), 2),
                "ECS_CPU": round(ecs_metrics.get('CPUUtilization', 0.0), 2),
                "EC2_MEM": round(ec2_metrics.get('MemoryUtilization', 0.0), 2),
                "RDS_MEM": round(100.0 - (rds_metrics.get('FreeableMemory', 0.0) / 1000), 2),
                "ECS_MEM": round(ecs_metrics.get('MemoryUtilization', 0.0), 2),
                
                # Current allocations
                "EC2_Count": int(ec2_count),
                "RDS_Type": instance_classes[int(rds_count)] if int(rds_count) < len(instance_classes) else "unknown",
                "ECS_Count": int(ecs_count),
                
                # Utilization per resource
                "EC2_Utilization": round(ec2_utilization, 2),
                "RDS_Utilization": round(rds_utilization, 2),
                "ECS_Utilization": round(ecs_utilization, 2),
                
                # Predictions
                "EC2_LSTM_Prediction": round(ec2_pred, 2),
                "RDS_LSTM_Prediction": round(rds_pred, 2),
                "ECS_LSTM_Prediction": round(ecs_pred, 2),
                
                # Scaling decisions
                "EC2_RL_Decision": ec2_decision,
                "RDS_RL_Decision": rds_decision,
                "ECS_RL_Decision": ecs_decision,
                
                # Scaling results
                "EC2_Scaling_Result": ec2_result,
                "RDS_Scaling_Result": rds_result,
                "ECS_Scaling_Result": ecs_result
            }
            
            # Store monitoring data in database
            try:
                with sqlite3.connect(DATABASE_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO metrics VALUES (?, ?, ?, ?, ?)",
                        (datetime.now(), 'Monitor', 'ALL', 'Dashboard', str(dashboard_predictions))
                    )
                    conn.commit()
            except Exception as db_error:
                logger.error(f"Database error: {db_error}")
                logger.error(traceback.format_exc())
                
            logger.info(f"Dashboard updated with latest predictions and decisions")
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {str(e)}")
            logger.error(traceback.format_exc())
            # Continue to next cycle even if there's an error
        
        logger.info(f"Next cycle in {cycle_interval} seconds")
        remaining_time = cycle_interval
        try:
            while remaining_time > 0:
                minutes, seconds = divmod(remaining_time, 60)
                print(f"  Next monitoring cycle in {minutes:02d}:{seconds:02d}", end="\r")
                # Sleep in shorter intervals to allow for more responsive shutdown
                sleep_interval = min(5, remaining_time)
                time.sleep(sleep_interval)
                remaining_time -= sleep_interval
            print(" " * 50, end="\r")  # Clear the status line
        except KeyboardInterrupt:
            print("\nMonitoring interrupted by user. Shutting down...")
            break

# Additional API endpoints for manual control and historical data
@app.route('/history/metrics', methods=['GET'])
@require_api_key
def get_historical_metrics():
    try:
        service = request.args.get('service', 'all').lower()
        hours = int(request.args.get('hours', 24))
        limit = int(request.args.get('limit', 1000))
        
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            if service == 'all':
                cursor.execute(
                    "SELECT timestamp, service, metric_name, metric_value FROM metrics WHERE timestamp > datetime('now', ?) ORDER BY timestamp DESC LIMIT ?",
                    (f"-{hours} hours", limit)
                )
            else:
                cursor.execute(
                    "SELECT timestamp, service, metric_name, metric_value FROM metrics WHERE service = ? AND timestamp > datetime('now', ?) ORDER BY timestamp DESC LIMIT ?",
                    (service.upper(), f"-{hours} hours", limit)
                )
            
            rows = cursor.fetchall()
            
            # Group by service and metric
            results = {}
            for timestamp, service, metric_name, value in rows:
                if service not in results:
                    results[service] = {}
                if metric_name not in results[service]:
                    results[service][metric_name] = []
                
                results[service][metric_name].append({
                    "timestamp": timestamp,
                    "value": value
                })
            
            return jsonify(results)
    except Exception as e:
        logger.error(f"Error getting historical metrics: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/history/decisions', methods=['GET'])
@require_api_key
def get_scaling_decisions():
    try:
        service = request.args.get('service', 'all').lower()
        hours = int(request.args.get('hours', 24))
        limit = int(request.args.get('limit', 100))
        
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            if service == 'all':
                cursor.execute(
                    "SELECT timestamp, service, decision, cpu_before, cpu_after, success FROM scaling_decisions WHERE timestamp > datetime('now', ?) ORDER BY timestamp DESC LIMIT ?",
                    (f"-{hours} hours", limit)
                )
            else:
                cursor.execute(
                    "SELECT timestamp, service, decision, cpu_before, cpu_after, success FROM scaling_decisions WHERE service = ? AND timestamp > datetime('now', ?) ORDER BY timestamp DESC LIMIT ?",
                    (service.upper(), f"-{hours} hours", limit)
                )
            
            rows = cursor.fetchall()
            
            results = []
            for timestamp, service, decision, cpu_before, cpu_after, success in rows:
                results.append({
                    "timestamp": timestamp,
                    "service": service,
                    "decision": decision,
                    "cpu_before": cpu_before,
                    "cpu_after": cpu_after,
                    "success": bool(success),
                    "effect": cpu_after - cpu_before
                })
            
            return jsonify(results)
    except Exception as e:
        logger.error(f"Error getting scaling decisions: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/manage/reload', methods=['POST'])
@require_api_key
def reload_models():
    try:
        success = load_models()
        if success:
            return jsonify({"status": "success", "message": "Models reloaded successfully"})
        else:
            return jsonify({"status": "error", "message": "Failed to reload models"}), 500
    except Exception as e:
        logger.error(f"Error reloading models: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": f"Error: {str(e)}"}), 500

# Create the monitoring thread but don't start it yet
monitoring_thread = threading.Thread(target=monitor_and_scale, daemon=True)

# Create a shutdown function for graceful termination
def shutdown_server():
    logger.info("Shutting down server...")
    # Cleanup code can go here if needed before shutdown

# Main entry point
if __name__ == '__main__':
    # Import traceback for better error reporting
    import traceback
    
    try:
        create_metrics_table()
        logger.info(f"Starting server on {HOST}:{PORT}")
        
        # Initialize models before starting monitoring thread
        models_loaded = load_models()
        logger.info(f"Models loaded successfully: {models_loaded}")
        
        if models_loaded:
            # Start monitoring thread if models loaded successfully
            monitoring_thread.start()
            logger.info("Monitoring thread started")
        else:
            logger.warning("Monitoring thread not started due to model loading failure")
        
        app.run(host=HOST, port=PORT, debug=False)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        logger.error(traceback.format_exc())