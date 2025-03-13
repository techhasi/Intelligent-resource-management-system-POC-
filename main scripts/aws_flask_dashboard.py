from flask import Flask, jsonify, request
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
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # Restrict to Grafana's origin for security

# Model Definitions (unchanged)
class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class DQN(nn.Module):
    def __init__(self, state_size=3, hidden_size=64, action_size=3):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Constants (unchanged)
EC2_INPUT_SIZE = 4
RDS_INPUT_SIZE = 4
ECS_INPUT_SIZE = 3

# Global Variables
ec2_lstm_model = None
rds_lstm_model = None
ecs_lstm_model = None
rl_model = None
dashboard_predictions = {}

# Model Loading
def load_models():
    global ec2_lstm_model, rds_lstm_model, ecs_lstm_model, rl_model
    try:
        ec2_lstm_model = LSTMModel(input_size=EC2_INPUT_SIZE)
        ec2_lstm_model.load_state_dict(torch.load('./models/EC2_lstm_model.pth', map_location=torch.device('cpu')))
        ec2_lstm_model.eval()
        
        rds_lstm_model = LSTMModel(input_size=RDS_INPUT_SIZE)
        rds_lstm_model.load_state_dict(torch.load('./models/RDS_lstm_model.pth', map_location=torch.device('cpu')))
        rds_lstm_model.eval()
        
        ecs_lstm_model = LSTMModel(input_size=ECS_INPUT_SIZE)
        ecs_lstm_model.load_state_dict(torch.load('./models/ECS_lstm_model.pth', map_location=torch.device('cpu')))
        ecs_lstm_model.eval()
        
        rl_model = DQN(state_size=3)
        rl_model.load_state_dict(torch.load('./dqn_scaling_model.pth', map_location=torch.device('cpu')))
        rl_model.eval()
        
        logger.info("Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        ec2_lstm_model, rds_lstm_model, ecs_lstm_model, rl_model = None, None, None, None
        return False

# Initialize AWS Clients with timeout
boto_config = Config(connect_timeout=20, read_timeout=20)
try:
    cloudwatch_client = boto3.client('cloudwatch', region_name='ap-southeast-1', config=boto_config)
    ec2_client = boto3.client('ec2', region_name='ap-southeast-1', config=boto_config)
    autoscaling_client = boto3.client('autoscaling', region_name='ap-southeast-1', config=boto_config)
    rds_client = boto3.client('rds', region_name='ap-southeast-1', config=boto_config)
    ecs_client = boto3.client('ecs', region_name='ap-southeast-1', config=boto_config)
    logger.info("AWS clients initialized successfully")
except Exception as e:
    logger.error(f"Error initializing AWS clients: {e}")
    cloudwatch_client, ec2_client, autoscaling_client, rds_client, ecs_client = None, None, None, None, None

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
def scale_ec2(decision, asg_name):
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
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"AWS ClientError scaling EC2 ASG {asg_name}: {error_code} - {error_message}")
        return f"AWS Error: {error_code} - {error_message}"
    except Exception as e:
        logger.error(f"Error scaling EC2 ASG {asg_name}: {str(e)}")
        return f"Error scaling EC2 ASG {asg_name}: {str(e)}"

def scale_rds(decision, db_instance_id='database-2'):
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
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"AWS ClientError scaling RDS {db_instance_id}: {error_code} - {error_message}")
        return f"AWS Error: {error_code} - {error_message}"
    except Exception as e:
        logger.error(f"Error scaling RDS {db_instance_id}: {str(e)}")
        return f"Error scaling RDS {db_instance_id}: {str(e)}"

def scale_ecs(decision, cluster_name, task_definition):
    try:
        response = ecs_client.list_tasks(cluster=cluster_name, desiredStatus='RUNNING')
        current_count = len(response['taskArns'])
        if decision == "scale up":
            response = ecs_client.run_task(
                cluster=cluster_name, taskDefinition=task_definition, count=1, launchType='FARGATE',
                networkConfiguration={'awsvpcConfiguration': {'subnets': ['subnet-022cc8297953122fd'], 'securityGroups': ['sg-0e09152973f9c89ae'], 'assignPublicIp': 'ENABLED'}}
            )
            task_arn = response['tasks'][0]['taskArn']
            logger.info(f"Started new task {task_arn} in cluster {cluster_name}, new count: {current_count + 1}")
            return f"Started new task {task_arn} in cluster {cluster_name}, new count: {current_count + 1}"
        elif decision == "scale down" and current_count > 1:
            task_to_stop = response['taskArns'][0]
            ecs_client.stop_task(cluster=cluster_name, task=task_to_stop, reason='Scaling down manually')
            logger.info(f"Stopped task {task_to_stop} in cluster {cluster_name}, new count: {current_count - 1}")
            return f"Stopped task {task_to_stop} in cluster {cluster_name}, new count: {current_count - 1}"
        else:
            logger.info(f"No scaling action for cluster {cluster_name}: {decision}")
            return f"No scaling action for cluster {cluster_name}: {decision}"
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"AWS ClientError scaling ECS cluster {cluster_name}: {error_code} - {error_message}")
        return f"AWS Error: {error_code} - {error_message}"
    except Exception as e:
        logger.error(f"Error scaling ECS cluster {cluster_name}: {str(e)}")
        return f"Error scaling ECS cluster {cluster_name}: {str(e)}"

def stop_ec2():
    try:
        response = ec2_client.describe_instances(
            Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
        )
        # Exclude the specified instance from the list
        exclude_instance = "i-0d586a401b59d560f"
        instances = [
            instance['InstanceId']
            for reservation in response['Reservations']
            for instance in reservation['Instances']
            if instance['InstanceId'] != exclude_instance
        ]
        if not instances:
            logger.info("No running EC2 instances found (excluding the specified instance)")
            return "No running EC2 instances found to stop (excluding the specified instance)"
        ec2_client.stop_instances(InstanceIds=instances)
        logger.info(f"Stopped EC2 instances: {', '.join(instances)}")
        return f"Stopped EC2 instances: {', '.join(instances)}"
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"AWS ClientError stopping EC2 instances: {error_code} - {error_message}")
        return f"AWS Error: {error_code} - {error_message}"
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
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"AWS ClientError stopping RDS instances: {error_code} - {error_message}")
        return f"AWS Error: {error_code} - {error_message}"
    except Exception as e:
        logger.error(f"Error stopping RDS instances: {str(e)}")
        return f"Error stopping RDS instances: {str(e)}"

def stop_ecs(cluster_name='my-ecs-cluster'):
    try:
        response = ecs_client.list_tasks(cluster=cluster_name, desiredStatus='RUNNING')
        task_arns = response['taskArns']
        for task_arn in task_arns:
            ecs_client.stop_task(cluster=cluster_name, task=task_arn, reason='Manual stop')
        result = f"Stopped {len(task_arns)} tasks in cluster {cluster_name}" if task_arns else "No running tasks found"
        logger.info(result)
        return result
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"AWS ClientError stopping ECS cluster {cluster_name}: {error_code} - {error_message}")
        return f"AWS Error: {error_code} - {error_message}"
    except Exception as e:
        logger.error(f"Error stopping ECS cluster {cluster_name}: {str(e)}")
        return f"Error stopping ECS cluster {cluster_name}: {str(e)}"

# Fetch Metrics (unchanged)
def fetch_ec2_metrics(instance_id):
    try:
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(minutes=5)
        metrics_data = {}
        ec2_metric_names = ['CPUUtilization', 'DiskWriteOps', 'NetworkIn']
        cwagent_metric_names = ['mem_used_percent']
        logger.info(f"Fetching EC2 metrics for {instance_id}")
        for metric in ec2_metric_names:
            response = cloudwatch_client.get_metric_statistics(
                Namespace='AWS/EC2', MetricName=metric, Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                StartTime=start_time, EndTime=now, Period=300, Statistics=['Average']
            )
            metrics_data[metric] = float(sorted(response['Datapoints'], key=lambda x: x['Timestamp'], reverse=True)[0]['Average']) if response['Datapoints'] else None
            logger.debug(f"EC2 {metric}: {metrics_data[metric]}")
        hostname = get_ec2_hostname(instance_id)
        if hostname:
            for metric in cwagent_metric_names:
                response = cloudwatch_client.get_metric_statistics(
                    Namespace='CWAgent', MetricName=metric, Dimensions=[{'Name': 'host', 'Value': hostname}],
                    StartTime=start_time, EndTime=now, Period=300, Statistics=['Average']
                )
                metrics_data['MemoryUtilization'] = float(sorted(response['Datapoints'], key=lambda x: x['Timestamp'], reverse=True)[0]['Average']) if response['Datapoints'] else None
                logger.debug(f"EC2 MemoryUtilization: {metrics_data['MemoryUtilization']}")
        else:
            metrics_data['MemoryUtilization'] = None
        logger.info(f"EC2 metrics fetched: {metrics_data}")
        return metrics_data
    except Exception as e:
        logger.error(f"Error fetching EC2 metrics for {instance_id}: {str(e)}")
        return {}

def fetch_rds_metrics(db_instance_id):
    try:
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(minutes=5)
        metrics_data = {}
        metric_names = ['CPUUtilization', 'FreeableMemory', 'DatabaseConnections', 'WriteIOPS']
        logger.info(f"Fetching RDS metrics for {db_instance_id}")
        for metric in metric_names:
            response = cloudwatch_client.get_metric_statistics(
                Namespace='AWS/RDS', MetricName=metric, Dimensions=[{'Name': 'DBInstanceIdentifier', 'Value': db_instance_id}],
                StartTime=start_time, EndTime=now, Period=300, Statistics=['Average']
            )
            metrics_data[metric] = float(sorted(response['Datapoints'], key=lambda x: x['Timestamp'], reverse=True)[0]['Average']) if response['Datapoints'] else None
            logger.debug(f"RDS {metric}: {metrics_data[metric]}")
        logger.info(f"RDS metrics fetched: {metrics_data}")
        return metrics_data
    except Exception as e:
        logger.error(f"Error fetching RDS metrics for {db_instance_id}: {str(e)}")
        return {}

def fetch_ecs_metrics(cluster_name):
    try:
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(minutes=5)
        metrics_data = {}
        metric_names = ['CPUUtilization', 'MemoryUtilization']
        logger.info(f"Fetching ECS metrics for {cluster_name}")
        for metric in metric_names:
            response = cloudwatch_client.get_metric_statistics(
                Namespace='AWS/ECS', MetricName=metric, Dimensions=[{'Name': 'ClusterName', 'Value': cluster_name}],
                StartTime=start_time, EndTime=now, Period=300, Statistics=['Average']
            )
            metrics_data[metric] = float(sorted(response['Datapoints'], key=lambda x: x['Timestamp'], reverse=True)[0]['Average']) if response['Datapoints'] else None
            logger.debug(f"ECS {metric}: {metrics_data[metric]}")
        response = ecs_client.list_tasks(cluster=cluster_name, desiredStatus='RUNNING')
        task_count = len(response['taskArns'])
        metrics_data['RunningTaskCount'] = float(task_count) if task_count > 0 else None
        logger.info(f"ECS metrics fetched: {metrics_data}")
        return metrics_data
    except Exception as e:
        logger.error(f"Error fetching ECS metrics for {cluster_name}: {str(e)}")
        return {}

# API Endpoints
@app.route('/')
def home():
    return "Flask server is running!"

@app.route('/predictions', methods=['GET'])
def get_predictions():
    serializable_predictions = {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in dashboard_predictions.items()
    }
    logger.info(f"Returning predictions: {serializable_predictions}")
    return jsonify(serializable_predictions)

@app.route('/stop/ec2', methods=['POST'])
def api_stop_ec2():
    logger.info(f"Received request for /stop/ec2 with data: {request.json}")
    result = stop_ec2()
    logger.info(f"Stop EC2 result: {result}")
    return jsonify({"message": result})

@app.route('/stop/rds', methods=['POST'])
def api_stop_rds():
    logger.info(f"Received request for /stop/rds with data: {request.json}")
    result = stop_rds()
    logger.info(f"Stop RDS result: {result}")
    return jsonify({"message": result})

@app.route('/stop/ecs', methods=['POST'])
def api_stop_ecs():
    logger.info(f"Received request for /stop/ecs with data: {request.json}")
    result = stop_ecs()
    logger.info(f"Stop ECS result: {result}")
    return jsonify({"message": result})

@app.route('/scale/ec2', methods=['POST'])
def api_scale_ec2():
    logger.info(f"Received request for /scale/ec2 with data: {request.json}")
    decision = request.json.get('decision', 'no change')
    asg_name = 'ec2-scaling'
    result = scale_ec2(decision, asg_name)
    logger.info(f"Scale EC2 result: {result}")
    return jsonify({"message": result})

@app.route('/scale/rds', methods=['POST'])
def api_scale_rds():
    logger.info(f"Received request for /scale/rds with data: {request.json}")
    decision = request.json.get('decision', 'no change')
    result = scale_rds(decision)
    logger.info(f"Scale RDS result: {result}")
    return jsonify({"message": result})

@app.route('/scale/ecs', methods=['POST'])
def api_scale_ecs():
    logger.info(f"Received request for /scale/ecs with data: {request.json}")
    decision = request.json.get('decision', 'no change')
    cluster_name = 'my-ecs-cluster'
    task_definition = 'my-task-definition'
    result = scale_ecs(decision, cluster_name, task_definition)
    logger.info(f"Scale ECS result: {result}")
    return jsonify({"message": result})

# Monitoring Thread
def monitor_and_scale():
    global dashboard_predictions
    models_loaded = load_models()
    
    def predict_usage(model, input_data):
        if model is None or not input_data or None in input_data:
            logger.warning("No model or invalid input data for prediction")
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
            logger.debug(f"Prediction: {prediction.flatten()[0]}")
            return float(prediction.flatten()[0])
        except Exception as e:
            logger.error(f"Error making LSTM prediction: {e}")
            return 0.0

    if not models_loaded:
        logger.error("Monitoring aborted due to model loading failure")
        return

    rl_decision_map = {0: "scale up", 1: "scale down", 2: "no change"}
    cycle_interval = 300  # 5 minutes
    instance_id = 'i-0275ce6aa1f61ca9f'
    asg_name = 'ec2-scaling'
    rds_instance_id = 'database-2'
    ecs_cluster = 'my-ecs-cluster'
    task_definition = 'my-task-definition'

    while True:
        logger.info("Starting monitoring cycle...")
        try:
            ec2_metrics = fetch_ec2_metrics(instance_id)
            rds_metrics = fetch_rds_metrics(rds_instance_id)
            ecs_metrics = fetch_ecs_metrics(ecs_cluster)
            logger.info(f"EC2 Metrics: {ec2_metrics}")
            logger.info(f"RDS Metrics: {rds_metrics}")
            logger.info(f"ECS Metrics: {ecs_metrics}")
            ec2_values = [v if v is not None else 0.0 for v in ec2_metrics.values()]
            rds_values = [v if v is not None else 0.0 for v in rds_metrics.values()]
            ecs_values = [v if v is not None else 0.0 for v in ecs_metrics.values()]
            ec2_pred = predict_usage(ec2_lstm_model, ec2_values)
            rds_pred = predict_usage(rds_lstm_model, rds_values)
            ecs_pred = predict_usage(ecs_lstm_model, ecs_values)
            rl_input = torch.tensor([ec2_pred, rds_pred, ecs_pred], dtype=torch.float32).unsqueeze(0)
            q_ec2, q_rds, q_ecs = rl_model(rl_input)
            ec2_decision = rl_decision_map[torch.argmax(q_ec2, dim=1).item()]
            rds_decision = rl_decision_map[torch.argmax(q_rds, dim=1).item()]
            ecs_decision = rl_decision_map[torch.argmax(q_ecs, dim=1).item()]
            logger.info(f"Decisions - EC2: {ec2_decision}, RDS: {rds_decision}, ECS: {ecs_decision}")
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
    app.run(host='0.0.0.0', port=5002, debug=False)