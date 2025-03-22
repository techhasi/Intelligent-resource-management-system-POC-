# import boto3
# import time
# import json

# ecs_client = boto3.client('ecs', region_name='ap-southeast-1')
# iam_client = boto3.client('iam', region_name='ap-southeast-1')

# def create_ecs_cluster(cluster_name='my-ecs-cluster', task_definition_name='my-task-definition'):
#     try:
#         # Step 1: Create IAM Role for Fargate Task Execution
#         role_name = 'ecsTaskExecutionRole'
#         assume_role_policy = {
#             "Version": "2012-10-17",
#             "Statement": [{"Effect": "Allow", "Principal": {"Service": "ecs-tasks.amazonaws.com"}, "Action": "sts:AssumeRole"}]
#         }
#         try:
#             iam_client.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(assume_role_policy))
#             print(f"Created IAM role: {role_name}")
#         except iam_client.exceptions.EntityAlreadyExistsException:
#             print(f"IAM role {role_name} already exists, proceeding...")

#         policy_arn = 'arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy'
#         iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
#         print(f"Attached policy {policy_arn} to role {role_name}")

#         time.sleep(10)  # Wait for IAM propagation

#         # # Step 2: Create ECS Cluster
#         # try:
#         #     ecs_client.create_cluster(clusterName=cluster_name)
#         #     print(f"Created ECS cluster: {cluster_name}")
#         # except ecs_client.exceptions.ClusterAlreadyExistsException:
#         #     print(f"ECS cluster {cluster_name} already exists, proceeding...")

#         # Step 3: Register Task Definition for Fargate
#         task_definition = {
#             "family": task_definition_name,
#             "networkMode": "awsvpc",  # Required for Fargate
#             "requiresCompatibilities": ["FARGATE"],
#             "cpu": "256",  # 0.25 vCPU
#             "memory": "512",  # 0.5 GB
#             "executionRoleArn": f"arn:aws:iam::515966523443:role/{role_name}",
#             "containerDefinitions": [
#                 {
#                     "name": "nginx",
#                     "image": "nginx:latest",
#                     "memory": 256,
#                     "cpu": 256,
#                     "essential": True,
#                     "portMappings": [
#                         {"containerPort": 80, "hostPort": 80, "protocol": "tcp"}
#                     ]
#                 }
#             ]
#         }
#         ecs_client.register_task_definition(**task_definition)
#         print(f"Registered task definition: {task_definition_name}")

#         # Step 4: Run Task with Fargate
#         response = ecs_client.run_task(
#             cluster=cluster_name,
#             taskDefinition=task_definition_name,
#             count=1,
#             launchType='FARGATE',
#             networkConfiguration={
#                 'awsvpcConfiguration': {
#                     'subnets': ['subnet-022cc8297953122fd'],  # Your subnet
#                     'securityGroups': ['sg-0e09152973f9c89ae'],  # Your security group
#                     'assignPublicIp': 'ENABLED'  # Required for Fargate to pull image
#                 }
#             }
#         )
#         task_arn = response['tasks'][0]['taskArn']
#         print(f"Started task with ARN: {task_arn}")

#         # Wait for task to be running (optional, for dashboard purposes)
#         waiter = ecs_client.get_waiter('tasks_running')
#         waiter.wait(cluster=cluster_name, tasks=[task_arn])
#         print(f"Task {task_arn} is running")

#         return True

#     except Exception as e:
#         print(f"Error creating ECS cluster: {str(e)}")
#         return False

# if __name__ == '__main__':
#     create_ecs_cluster()
    
    
    
    
    
    
    #------------------------------------------------
    #Create a tash to increase CPU and memory usage
    #------------------------------------------------
    
import boto3
import time
import json

ecs_client = boto3.client('ecs', region_name='ap-southeast-1')
iam_client = boto3.client('iam', region_name='ap-southeast-1')

def create_ecs_cluster(cluster_name='my-ecs-cluster', task_definition_name='my-fluctuate-task'):
    try:
        # Step 1: Create IAM Role for Fargate Task Execution
        role_name = 'ecsTaskExecutionRole'
        assume_role_policy = {
            "Version": "2012-10-17",
            "Statement": [{"Effect": "Allow", "Principal": {"Service": "ecs-tasks.amazonaws.com"}, "Action": "sts:AssumeRole"}]
        }
        try:
            iam_client.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(assume_role_policy))
            print(f"Created IAM role: {role_name}")
        except iam_client.exceptions.EntityAlreadyExistsException:
            print(f"IAM role {role_name} already exists, proceeding...")

        policy_arn = 'arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy'
        iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        print(f"Attached policy {policy_arn} to role {role_name}")

        time.sleep(10)  # Wait for IAM propagation

        # Step 2: Register Task Definition for Fargate with Fluctuating Workload
        task_definition = {
            "family": task_definition_name,
            "networkMode": "awsvpc",  # Required for Fargate
            "requiresCompatibilities": ["FARGATE"],
            "cpu": "256",  # 0.25 vCPU
            "memory": "512",  # 0.5 GB
            "executionRoleArn": f"arn:aws:iam::515966523443:role/{role_name}",
            "containerDefinitions": [
                {
                    "name": "fluctuate-container",
                    "image": "busybox:latest",  # Lightweight image with shell utilities
                    "memory": 512,
                    "cpu": 256,
                    "essential": True,
                    "command": [
                        "/bin/sh", "-c",
                        # Shell script with fluctuating CPU and memory usage
                        """
                        while true; do
                            # CPU burst: Run dd for 10 seconds, then sleep
                            dd if=/dev/zero of=/dev/null bs=1M count=1000 & pid=$!
                            sleep 10
                            kill $pid
                            sleep $((RANDOM % 10 + 5))  # Random sleep 5-15s

                            # Memory burst: Allocate memory, hold for 8s, then free
                            for i in $(seq 1 50); do echo 'Fluctuate' > /tmp/mem_$i; done
                            sleep 8
                            rm -f /tmp/mem_*  # Free memory
                            sleep $((RANDOM % 10 + 5))  # Random sleep 5-15s
                        done
                        """
                    ]
                }
            ]
        }
        ecs_client.register_task_definition(**task_definition)
        print(f"Registered task definition: {task_definition_name}")

        # Step 3: Run Task with Fargate
        response = ecs_client.run_task(
            cluster=cluster_name,
            taskDefinition=task_definition_name,
            count=1,
            launchType='FARGATE',
            networkConfiguration={
                'awsvpcConfiguration': {
                    'subnets': ['subnet-022cc8297953122fd'],  # Your subnet
                    'securityGroups': ['sg-0e09152973f9c89ae'],  # Your security group
                    'assignPublicIp': 'ENABLED'  # Required for Fargate to pull image
                }
            }
        )
        task_arn = response['tasks'][0]['taskArn']
        print(f"Started task with ARN: {task_arn}")

        # Wait for task to be running (optional)
        waiter = ecs_client.get_waiter('tasks_running')
        waiter.wait(cluster=cluster_name, tasks=[task_arn])
        print(f"Task {task_arn} is running")

        return True

    except Exception as e:
        print(f"Error creating ECS cluster: {str(e)}")
        return False

if __name__ == '__main__':
    create_ecs_cluster()