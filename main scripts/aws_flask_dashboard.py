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
import traceback

# Configuration Variables 

# API Configuration
API_KEY = os.getenv('API_KEY', 'hasithaw54') 
FRONTEND_ORIGIN = "http://localhost:3000"
HOST = '0.0.0.0'
PORT = 5002

# AWS Configuration
AWS_REGION = 'ap-southeast-1'
BOTO_TIMEOUT = 20  
EC2_INSTANCE_ID = 'i-0275ce6aa1f61ca9f'
EC2_ASG_NAME = 'ec2-scaling'
EC2_EXCLUDE_INSTANCE = "i-0d586a401b59d560f"
RDS_INSTANCE_ID = 'database-2'
ECS_CLUSTER_NAME = 'my-ecs-cluster'
ECS_TASK_DEFINITION = 'my-highload-task'
ECS_SUBNETS = ['subnet-022cc8297953122fd']
ECS_SECURITY_GROUPS = ['sg-0e09152973f9c89ae']

# Model Paths
EC2_MODEL_PATH = '../models/ec2_lstm_model.pth' 
RDS_MODEL_PATH = '../models/rds_lstm_model.pth' 
ECS_MODEL_PATH = '../models/ecs_lstm_model.pth'  
RL_MODEL_PATH = '../models/cloud_dqn.pkl'    

# Model Parameters
SEQUENCE_LENGTH = 24  
LSTM_HIDDEN_SIZE = 256  
LSTM_NUM_LAYERS = 3   
LSTM_DROPOUT = 0.3    
BINS_PER_DIM = 8       

# Monitoring Configuration
MONITORING_INTERVAL = 300 
DATABASE_PATH = 'cloud_metrics.db'

# Log configuration
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

# API Key Authentication
def require_api_key(func):
    def wrapper(*args, **kwargs):
        key = request.headers.get("x-api-key")
        if key != API_KEY:
            logger.warning(f"Unauthorized access attempt with invalid API key: {request.remote_addr}")
            return jsonify({"message": "Invalid or missing API key"}), 403
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)

        self.ln = nn.LayerNorm(hidden_size)

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                elif 'attention' in name or 'fc' in name:
                    if len(param.shape) >= 2:
                        nn.init.xavier_uniform_(param.data)
                    else:
                        nn.init.uniform_(param.data, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
                if 'lstm' in name and 'bias_hh' in name:
                    n = param.size(0)
                    param.data[self.hidden_size:2*self.hidden_size].fill_(1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = self.ln(lstm_out)

        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)

        out = self.fc1(context)
        out = self.ln2(out)
        out = nn.functional.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        out = torch.sigmoid(out)
        return out

ec2_lstm_model = None
rds_lstm_model = None
ecs_lstm_model = None
rl_model = None
rl_scaler = None
dashboard_predictions = {}
state_history = deque(maxlen=3) 
ec2_sequence_history = deque(maxlen=SEQUENCE_LENGTH)
rds_sequence_history = deque(maxlen=SEQUENCE_LENGTH)
ecs_sequence_history = deque(maxlen=SEQUENCE_LENGTH)

def load_models():
    global ec2_lstm_model, rds_lstm_model, ecs_lstm_model, rl_model
    global ec2_feature_scaler, ec2_target_scaler
    global rds_feature_scaler, rds_target_scaler  
    global ecs_feature_scaler, ecs_target_scaler
    
    try:
        logger.info("Starting model loading process...")
        
        logger.info(f"Loading EC2 model from {EC2_MODEL_PATH}")
        ec2_checkpoint = torch.load(EC2_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        logger.info(f"EC2 model loaded. Architecture: "
                   f"input_size={ec2_checkpoint['input_size']}, "
                   f"hidden_size={ec2_checkpoint['hidden_size']}, "
                   f"num_layers={ec2_checkpoint['num_layers']}, "
                   f"dropout={ec2_checkpoint['dropout']}")
        
        ec2_lstm_model = LSTM(
            input_size=ec2_checkpoint['input_size'],
            hidden_size=ec2_checkpoint['hidden_size'],
            num_layers=ec2_checkpoint['num_layers'],
            output_size=1,
            dropout=ec2_checkpoint['dropout']
        )
        ec2_lstm_model.load_state_dict(ec2_checkpoint['model_state_dict'])
        ec2_lstm_model.eval()
        logger.info("EC2 model initialized and set to eval mode")
        
        logger.info(f"Loading RDS model from {RDS_MODEL_PATH}")
        rds_checkpoint = torch.load(RDS_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        logger.info(f"RDS model loaded. Architecture: "
                   f"input_size={rds_checkpoint['input_size']}, "
                   f"hidden_size={rds_checkpoint['hidden_size']}, "
                   f"num_layers={rds_checkpoint['num_layers']}, "
                   f"dropout={rds_checkpoint['dropout']}")
        
        rds_lstm_model = LSTM(
            input_size=rds_checkpoint['input_size'],
            hidden_size=rds_checkpoint['hidden_size'],
            num_layers=rds_checkpoint['num_layers'],
            output_size=1,
            dropout=rds_checkpoint['dropout']
        )
        rds_lstm_model.load_state_dict(rds_checkpoint['model_state_dict'])
        rds_lstm_model.eval()
        logger.info("RDS model initialized and set to eval mode")
        
        logger.info(f"Loading ECS model from {ECS_MODEL_PATH}")
        ecs_checkpoint = torch.load(ECS_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        logger.info(f"ECS model loaded. Architecture: "
                   f"input_size={ecs_checkpoint['input_size']}, "
                   f"hidden_size={ecs_checkpoint['hidden_size']}, "
                   f"num_layers={ecs_checkpoint['num_layers']}, "
                   f"dropout={ecs_checkpoint['dropout']}")
        
        ecs_lstm_model = LSTM(
            input_size=ecs_checkpoint['input_size'],
            hidden_size=ecs_checkpoint['hidden_size'],
            num_layers=ecs_checkpoint['num_layers'],
            output_size=1,
            dropout=ecs_checkpoint['dropout']
        )
        ecs_lstm_model.load_state_dict(ecs_checkpoint['model_state_dict'])
        ecs_lstm_model.eval()
        logger.info("ECS model initialized and set to eval mode")
        
        try:
            logger.info("Loading feature scalers...")
            
            ec2_scaler_path = EC2_MODEL_PATH.replace(".pth", "_scalers.pkl")
            logger.info(f"Loading EC2 scalers from {ec2_scaler_path}")
            with open(ec2_scaler_path, 'rb') as f:
                ec2_scalers = pickle.load(f)
                ec2_feature_scaler = ec2_scalers['feature_scaler']
                ec2_target_scaler = ec2_scalers['target_scaler']
                logger.info("EC2 scalers loaded successfully")
            
            rds_scaler_path = RDS_MODEL_PATH.replace(".pth", "_scalers.pkl")
            logger.info(f"Loading RDS scalers from {rds_scaler_path}")
            with open(rds_scaler_path, 'rb') as f:
                rds_scalers = pickle.load(f)
                rds_feature_scaler = rds_scalers['feature_scaler']
                rds_target_scaler = rds_scalers['target_scaler']
                logger.info("RDS scalers loaded successfully")
            
            ecs_scaler_path = ECS_MODEL_PATH.replace(".pth", "_scalers.pkl")
            logger.info(f"Loading ECS scalers from {ecs_scaler_path}")
            with open(ecs_scaler_path, 'rb') as f:
                ecs_scalers = pickle.load(f)
                ecs_feature_scaler = ecs_scalers['feature_scaler']
                ecs_target_scaler = ecs_scalers['target_scaler']
                logger.info("ECS scalers loaded successfully")
                
        except Exception as scaler_error:
            logger.error(f"Error loading scalers: {scaler_error}")
            logger.error(traceback.format_exc())
            return False

        
        with open(RL_MODEL_PATH, 'rb') as f:
            rl_model_data = pickle.load(f)
            
        from collections import defaultdict
        class EnhancedQAgent:
            def __init__(self, bins_per_dim=BINS_PER_DIM, action_size_per_service=3, num_services=3):
                self.bins_per_dim = bins_per_dim
                self.action_size_per_service = action_size_per_service
                self.num_services = num_services
                self.q_table = defaultdict(float)
                self.epsilon = 0.0 
            
            def load_model(self, data):
                self.q_table = defaultdict(float, data['q_table'])
                self.epsilon = 0.0 
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
                    for i in range(3): 
                        if i < len(state) and not np.isnan(state[i]):
                            resource_count = min(int(state[i]), 10)
                            bin_idx = min(int((resource_count - 1) * self.bins_per_dim / 10), self.bins_per_dim - 1)
                            discrete_state.append(bin_idx)
                        else:
                            discrete_state.append(1)
                    
                    for i in range(3, 9):
                        if i < len(state) and not np.isnan(state[i]):
                            value = max(0.0, min(1.0, state[i]))
                            bin_idx = min(int(value * self.bins_per_dim), self.bins_per_dim - 1)
                            discrete_state.append(bin_idx)
                        else:
                            discrete_state.append(self.bins_per_dim // 2)
                    
                    if len(state) > 9 and not np.isnan(state[9]):
                        hour = int(state[9]) % 24
                        time_bin = hour // 6
                        discrete_state.append(time_bin)
                    else:
                        discrete_state.append(2)
                    
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
        logger.info("All models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        ec2_lstm_model, rds_lstm_model, ecs_lstm_model, rl_model = None, None, None, None
        return False

# Initialize AWS Clients
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

# Database creation
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

def get_ec2_hostname(instance_id):
    try:
        response = ec2_client.describe_instances(InstanceIds=[instance_id])
        hostname = response['Reservations'][0]['Instances'][0]['PrivateDnsName']
        return hostname
    except Exception as e:
        logger.error(f"Error fetching hostname for instance {instance_id}: {str(e)}")
        return None

def scale_ec2(decision, current_metrics=None, asg_name=EC2_ASG_NAME):
    try:
        cpu_before = current_metrics.get('CPUUtilization', 0.0) if current_metrics else 0.0
        
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
        
        ec2_metrics_after = fetch_ec2_metrics() if not current_metrics else current_metrics
        cpu_after = ec2_metrics_after.get('CPUUtilization', 0.0)
        
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

def scale_rds(decision, current_metrics=None, db_instance_id=RDS_INSTANCE_ID):
    try:
        cpu_before = current_metrics.get('CPUUtilization', 0.0) if current_metrics else 0.0
        
        response = rds_client.describe_db_instances(DBInstanceIdentifier=db_instance_id)
        current_class = response['DBInstances'][0]['DBInstanceClass']
        instance_classes = ['db.t3.micro', 'db.t3.small', 'db.t3.medium', 'db.t3.large']
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
        
        rds_metrics_after = fetch_rds_metrics() if not current_metrics else current_metrics
        cpu_after = rds_metrics_after.get('CPUUtilization', 0.0)
        
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

def scale_ecs(decision, current_metrics=None, cluster_name=ECS_CLUSTER_NAME, task_definition=ECS_TASK_DEFINITION):
    """
    Scale ECS tasks based on RL decision
    decision: "scale up", "scale down", or "no change"
    """
    try:
        cpu_before = current_metrics.get('CPUUtilization', 0.0) if current_metrics else 0.0
        
        response = ecs_client.list_tasks(cluster=cluster_name, desiredStatus='RUNNING')
        current_count = len(response['taskArns'])
        
        if decision == "scale up":
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
        
        ecs_metrics_after = fetch_ecs_metrics() if not current_metrics else current_metrics
        cpu_after = ecs_metrics_after.get('CPUUtilization', 0.0)
        
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
                Namespace='AWS/RDS',
                MetricName=metric,
                Dimensions=[{'Name': 'DBInstanceIdentifier', 'Value': db_instance_id}],
                StartTime=start_time,
                EndTime=now,
                Period=300,
                Statistics=['Average']
            )
            metrics_data[metric] = float(sorted(response['Datapoints'], key=lambda x: x['Timestamp'], reverse=True)[0]['Average']) if response['Datapoints'] else 0.0
        
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
        return {'CPUUtilization': 0.0,'FreeableMemory': 0.0,'DatabaseConnections': 0.0,'WriteIOPS': 0.0}

def fetch_ecs_metrics(cluster_name=ECS_CLUSTER_NAME):
    try:
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(minutes=5)
        metrics_data = {}
        metric_names = ['CpuUtilized', 'MemoryUtilized']
        
        for metric in metric_names:
            response = cloudwatch_client.get_metric_statistics(
                Namespace='ECS/ContainerInsights', MetricName=metric, Dimensions=[{'Name': 'ClusterName', 'Value': cluster_name}],
                StartTime=start_time, EndTime=now, Period=300, Statistics=['Average']
            )
            metrics_data[metric] = float(sorted(response['Datapoints'], key=lambda x: x['Timestamp'], reverse=True)[0]['Average']) if response['Datapoints'] else 0.0
        
        if 'CpuUtilized' in metrics_data:
            metrics_data['CPUUtilization'] = (metrics_data['CpuUtilized'] / 256.0) * 100.0
        
        if 'MemoryUtilized' in metrics_data:
            metrics_data['MemoryUtilization'] = (metrics_data['MemoryUtilized'] / 512.0) * 100.0
        
        response = ecs_client.list_tasks(cluster=cluster_name, desiredStatus='RUNNING')
        task_count = len(response['taskArns'])
        metrics_data['RunningTaskCount'] = float(task_count) if task_count > 0 else 0.0
        
        try:
            for network_metric in ['NetworkRxBytes', 'NetworkTxBytes']:
                response = cloudwatch_client.get_metric_statistics(
                    Namespace='ECS/ContainerInsights', MetricName=network_metric, Dimensions=[{'Name': 'ClusterName', 'Value': cluster_name}],
                    StartTime=start_time, EndTime=now, Period=300, Statistics=['Average']
                )
                metrics_data[network_metric] = float(sorted(response['Datapoints'], key=lambda x: x['Timestamp'], reverse=True)[0]['Average']) if response['Datapoints'] else 0.0
        except Exception as network_error:
            logger.warning(f"Could not fetch ECS network metrics: {network_error}")
            metrics_data['NetworkRxBytes'] = 0.0
            metrics_data['NetworkTxBytes'] = 0.0
            
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
    now = datetime.now()
    hour = float(now.hour)
    day_of_week = float(now.weekday())
    day_of_month = float(now.day)
    
    try:
        logger.debug(f"Preparing LSTM input for {service}")

        if service == 'EC2':
            # Base metrics
            cpu = float(metrics.get('CPUUtilization', 0.0))
            mem = float(metrics.get('MemoryUtilization', 0.0))
            disk = float(metrics.get('DiskWriteOps', 0.0))
            network = float(metrics.get('NetworkIn', 0.0))
            
            current_data = [
                cpu,  
                float(sequence_history[-1][0]) if sequence_history else cpu,  
                cpu - (float(sequence_history[-1][0]) if sequence_history else cpu), 
                disk,  
                mem,  
                network,  
                np.mean([h[0] for h in list(sequence_history)[-6:]]) if sequence_history else cpu,  
                np.std([h[0] for h in list(sequence_history)[-6:]]) if len(sequence_history) >= 2 else 0.0, 
                float(sequence_history[-2][0]) if len(sequence_history) >= 2 else cpu,  
                cpu - (float(sequence_history[-3][0]) if len(sequence_history) >= 3 else cpu), 
                np.log1p(cpu) / (np.log1p(disk) + 1e-8),  
                np.log1p(cpu) / (np.log1p(mem) + 1e-8),   
                np.sin(2 * np.pi * day_of_month / 31),     
                np.cos(2 * np.pi * day_of_month / 31)     
            ]
            
        elif service == 'RDS':
            cpu = float(metrics.get('CPUUtilization', 0.0))
            memory = float(metrics.get('FreeableMemory', 0.0)) 
            conn = float(metrics.get('DatabaseConnections', 0.0))
            io = float(metrics.get('WriteIOPS', 0.0))
            
            memory_usage = 100 - (memory / 1000)  
            current_data = [
                cpu,  
                float(sequence_history[-1][0]) if sequence_history else cpu, 
                cpu - (float(sequence_history[-1][0]) if sequence_history else cpu), 
                io,  
                memory_usage, 
                conn,  
                np.mean([h[0] for h in list(sequence_history)[-6:]]) if sequence_history else cpu, 
                np.std([h[0] for h in list(sequence_history)[-6:]]) if len(sequence_history) >= 2 else 0.0, 
                float(sequence_history[-2][0]) if len(sequence_history) >= 2 else cpu, 
                cpu - (float(sequence_history[-3][0]) if len(sequence_history) >= 3 else cpu),
                np.log1p(cpu) / (np.log1p(io) + 1e-8),    
                np.log1p(cpu) / (np.log1p(memory_usage) + 1e-8),
                np.percentile([h[0] for h in list(sequence_history)[-6:]], 25) if len(sequence_history) >= 4 else cpu * 0.8,  
                np.percentile([h[0] for h in list(sequence_history)[-6:]], 75) if len(sequence_history) >= 4 else cpu * 1.2, 
                np.sin(2 * np.pi * hour / 24),  
                np.cos(2 * np.pi * hour / 24), 
                np.sin(2 * np.pi * day_of_month / 31) 
            ]
            
        elif service == 'ECS':
            cpu = float(metrics.get('CPUUtilization', 0.0))
            mem = float(metrics.get('MemoryUtilization', 0.0))
            tasks = float(metrics.get('RunningTaskCount', 0.0))
            network = float(metrics.get('NetworkRxBytes', 0.0))
            
            current_data = [
                cpu,
                float(sequence_history[-1][0]) if sequence_history else cpu,
                cpu - (float(sequence_history[-1][0]) if sequence_history else cpu),
                tasks,  
                mem, 
                network, 
                np.mean([h[0] for h in list(sequence_history)[-6:]]) if sequence_history else cpu,
                np.std([h[0] for h in list(sequence_history)[-6:]]) if len(sequence_history) >= 2 else 0.0,
                float(sequence_history[-2][0]) if len(sequence_history) >= 2 else cpu, 
                cpu - (float(sequence_history[-3][0]) if len(sequence_history) >= 3 else cpu), 
                np.log1p(cpu) / (np.log1p(tasks) + 1e-8),  
                np.log1p(cpu) / (np.log1p(mem) + 1e-8),   
                np.percentile([h[0] for h in list(sequence_history)[-6:]], 10) if len(sequence_history) >= 4 else cpu * 0.9,  
                np.percentile([h[0] for h in list(sequence_history)[-6:]], 25) if len(sequence_history) >= 4 else cpu * 0.8, 
                np.percentile([h[0] for h in list(sequence_history)[-6:]], 75) if len(sequence_history) >= 4 else cpu * 1.2, 
                np.percentile([h[0] for h in list(sequence_history)[-6:]], 90) if len(sequence_history) >= 4 else cpu * 1.1, 
                np.sin(2 * np.pi * hour / 24),     
                np.cos(2 * np.pi * hour / 24)      
            ]

        expected_sizes = {'EC2': 14, 'RDS': 17, 'ECS': 18}
        if len(current_data) != expected_sizes[service]:
            raise ValueError(f"Feature dimension mismatch for {service}. Expected {expected_sizes[service]}, got {len(current_data)}")

        current_data = [0.0 if x is None or np.isnan(x) else float(x) for x in current_data]
        
        sequence_history.append(current_data[:])
        logger.debug(f"Prepared {service} input with {len(current_data)} features")
        
        return np.array([current_data], dtype=np.float32)
        
    except Exception as e:
        logger.error(f"Error preparing {service} input: {str(e)}")
        logger.error(traceback.format_exc())
        default_size = {'EC2': 14, 'RDS': 17, 'ECS': 18}[service]
        return np.zeros((1, default_size), dtype=np.float32)

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
    
    def predict_usage(model, input_data, target_scaler=None, current_metrics=None, service_type=None):
        if current_metrics:
            if service_type in ['EC2', 'ECS'] and current_metrics.get('CPUUtilization', 1) == 0:
                logger.info(f"{service_type} service stopped - forcing 0% prediction")
                return 0.0
                
            if service_type == 'RDS' and current_metrics.get('DatabaseConnections', 1) == 0 and current_metrics.get('WriteIOPS', 1) == 0:
                logger.info(f"RDS appears idle - forcing 0% prediction")
                return 0.0

        if model is None or not input_data.size:
            logger.warning("No model or invalid input data for prediction")
            return 0.0
            
        try:
            model.eval()
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                prediction = model(input_tensor).detach().numpy()
                
                if target_scaler is not None:
                    prediction = target_scaler.inverse_transform(prediction)
                
                prediction = np.clip(prediction, 0, 100)
                return float(prediction.flatten()[0])
                
        except Exception as e:
            logger.error(f"Prediction error for {service_type}: {e}")
            logger.error(traceback.format_exc())
            return 0.0

    def safe_discretize_state(state):
        discrete_state = []
        
        try:
            for i in range(3): 
                if i < len(state) and not np.isnan(state[i]):
                    resource_count = min(int(state[i]), 10)
                    bin_idx = min(int((resource_count - 1) * BINS_PER_DIM / 10), BINS_PER_DIM - 1)
                    discrete_state.append(bin_idx)
                else:
                    discrete_state.append(1) 
            
            for i in range(3, 9): 
                if i < len(state) and not np.isnan(state[i]):
                    value = max(0.0, min(1.0, state[i]))
                    bin_idx = min(int(value * BINS_PER_DIM), BINS_PER_DIM - 1)
                    discrete_state.append(bin_idx)
                else:
                    discrete_state.append(BINS_PER_DIM // 2)
            
            if len(state) > 9 and not np.isnan(state[9]):
                hour = int(state[9]) % 24
                time_bin = hour // 6
                discrete_state.append(time_bin)
            else:
                discrete_state.append(2)
            
            if len(state) > 10 and not np.isnan(state[10]):
                day = int(state[10]) % 7
                day_bin = 1 if day >= 5 else 0
                discrete_state.append(day_bin)
            else:
                discrete_state.append(0)
                
        except Exception as e:
            logger.error(f"Error in discretize_state: {e}")
            logger.error(traceback.format_exc())
            return tuple([1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0])
        
        return tuple(discrete_state)

    rl_decision_map = {0: "scale down", 1: "no change", 2: "scale up"}
    cycle_interval = MONITORING_INTERVAL

    state_history.extend([np.array([0.0, 0.0, 0.0])] * 3)
    
    ec2_sequence_history.extend([[0.0] * 13] * SEQUENCE_LENGTH) 
    rds_sequence_history.extend([[0.0] * 13] * SEQUENCE_LENGTH)  
    ecs_sequence_history.extend([[0.0] * 13] * SEQUENCE_LENGTH) 

    while True:
        logger.info("Starting monitoring cycle...")
        try:
            ec2_metrics = fetch_ec2_metrics()
            rds_metrics = fetch_rds_metrics()
            ecs_metrics = fetch_ecs_metrics()

            ec2_input = prepare_lstm_input(ec2_metrics, 'EC2', ec2_sequence_history)
            rds_input = prepare_lstm_input(rds_metrics, 'RDS', rds_sequence_history)
            ecs_input = prepare_lstm_input(ecs_metrics, 'ECS', ecs_sequence_history)

            ec2_pred = predict_usage(ec2_lstm_model, ec2_input, ec2_target_scaler, ec2_metrics, 'EC2')
            rds_pred = predict_usage(rds_lstm_model, rds_input, rds_target_scaler, rds_metrics, 'RDS') 
            ecs_pred = predict_usage(ecs_lstm_model, ecs_input, ecs_target_scaler, ecs_metrics, 'ECS')
            
            logger.info(f"LSTM Predictions - EC2: {ec2_pred:.2f}, RDS: {rds_pred:.2f}, ECS: {ecs_pred:.2f}")

            try:
                ec2_response = autoscaling_client.describe_auto_scaling_groups(AutoScalingGroupNames=[EC2_ASG_NAME])
                ec2_count = float(ec2_response['AutoScalingGroups'][0]['DesiredCapacity'])
                
                rds_response = rds_client.describe_db_instances(DBInstanceIdentifier=RDS_INSTANCE_ID)
                current_class = rds_response['DBInstances'][0]['DBInstanceClass']
                instance_classes = ['db.t3.micro', 'db.t3.small', 'db.t3.medium', 'db.t3.large', 'db.r5.large']
                rds_count = float(instance_classes.index(current_class) if current_class in instance_classes else 0)
                
                ecs_response = ecs_client.list_tasks(cluster=ECS_CLUSTER_NAME, desiredStatus='RUNNING')
                ecs_count = float(len(ecs_response['taskArns']))
            except Exception as e:
                logger.error(f"Error getting resource counts: {e}")
                logger.error(traceback.format_exc())
                ec2_count, rds_count, ecs_count = 2.0, 1.0, 2.0
            
            ec2_cpu = float(ec2_metrics.get('CPUUtilization', 0.0)) / 100.0
            rds_cpu = float(rds_metrics.get('CPUUtilization', 0.0)) / 100.0
            ecs_cpu = float(ecs_metrics.get('CPUUtilization', 0.0)) / 100.0
            
            ec2_pred_norm = min(1.0, max(0.0, ec2_pred/100.0))
            rds_pred_norm = min(1.0, max(0.0, rds_pred/100.0))
            ecs_pred_norm = min(1.0, max(0.0, ecs_pred/100.0))
            
            current_state = np.array([
            ec2_count, rds_count, ecs_count, 
            ec2_cpu, rds_cpu, ecs_cpu,      
            ec2_pred_norm, rds_pred_norm, ecs_pred_norm,  
            float(datetime.now().hour),       
            float(datetime.now().weekday())   
        ])
            state_history.append(current_state[:3]) 
            
            discrete_state = safe_discretize_state(current_state)
            
            actions = rl_model.select_action(current_state)
            
            ec2_decision = rl_decision_map[actions[0]]
            rds_decision = rl_decision_map[actions[1]]
            ecs_decision = rl_decision_map[actions[2]]
            
            logger.info(f"RL Decisions - EC2: {ec2_decision}, RDS: {rds_decision}, ECS: {ecs_decision}")

            ec2_result = scale_ec2(ec2_decision, ec2_metrics)
            rds_result = scale_rds(rds_decision, rds_metrics)
            ecs_result = scale_ecs(ecs_decision, ecs_metrics)
            
            logger.info(f"Scaling Results - EC2: {ec2_result}, RDS: {rds_result}, ECS: {ecs_result}")

            dashboard_predictions = {
                "timestamp": datetime.now().isoformat(),
                
                "EC2_LSTM_Prediction": round(ec2_pred, 2),
                "RDS_LSTM_Prediction": round(rds_pred, 2),
                "ECS_LSTM_Prediction": round(ecs_pred, 2),
                
                "EC2_RL_Decision": ec2_decision,
                "RDS_RL_Decision": rds_decision,
                "ECS_RL_Decision": ecs_decision,  
            }
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
        
        logger.info(f"Next cycle in {cycle_interval} seconds")
        time.sleep(cycle_interval)

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

monitoring_thread = threading.Thread(target=monitor_and_scale, daemon=True)

def shutdown_server():
    logger.info("Shutting down server...")

if __name__ == '__main__':
    try:
        create_metrics_table()
        logger.info(f"Starting server on {HOST}:{PORT}")
        
        models_loaded = load_models()
        logger.info(f"Models loaded successfully: {models_loaded}")
        
        if models_loaded:
            monitoring_thread.start()
            logger.info("Monitoring thread started")
        else:
            logger.warning("Monitoring thread not started due to model loading failure")
        
        app.run(host=HOST, port=PORT, debug=False)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        logger.error(traceback.format_exc())