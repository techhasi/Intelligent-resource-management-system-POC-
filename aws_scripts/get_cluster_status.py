import boto3

ecs_client = boto3.client('ecs', region_name='ap-southeast-1')

def get_cluster_status(cluster_name='my-ecs-cluster'):
    try:
        # Describe the cluster
        response = ecs_client.describe_clusters(clusters=[cluster_name])
        cluster = response['clusters'][0]
        print(f"Cluster: {cluster_name}")
        print(f"Status: {cluster['status']}")
        print(f"Running Tasks: {cluster['runningTasksCount']}")
        print(f"Pending Tasks: {cluster['pendingTasksCount']}")

        # List running tasks
        tasks = ecs_client.list_tasks(cluster=cluster_name, desiredStatus='RUNNING')['taskArns']
        if tasks:
            task_details = ecs_client.describe_tasks(cluster=cluster_name, tasks=tasks)['tasks']
            print("\nRunning Tasks:")
            for task in task_details:
                print(f"Task ARN: {task['taskArn']}")
                print(f"Status: {task['lastStatus']}")
                print(f"Container: {task['containers'][0]['name']} (Image: {task['containers'][0]['image']})")
                print(f"Started At: {task.get('startedAt', 'N/A')}")
                print("---")
        else:
            print("No running tasks found.")

    except Exception as e:
        print(f"Error fetching cluster status: {str(e)}")

if __name__ == '__main__':
    get_cluster_status()