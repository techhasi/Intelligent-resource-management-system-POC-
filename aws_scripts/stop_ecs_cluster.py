import boto3
import time

# Initialize ECS client
ecs_client = boto3.client('ecs', region_name='ap-southeast-1')

def stop_ecs_cluster(cluster_name='my-ecs-cluster', delete_cluster=False):
    try:
        # Step 1: List all running tasks in the cluster
        response = ecs_client.list_tasks(
            cluster=cluster_name,
            desiredStatus='RUNNING'
        )
        task_arns = response['taskArns']
        print(f"Found {len(task_arns)} running tasks in cluster {cluster_name}")

        # Step 2: Stop each running task
        for task_arn in task_arns:
            ecs_client.stop_task(
                cluster=cluster_name,
                task=task_arn,
                reason='Stopping cluster to avoid charges'
            )
            print(f"Stopped task: {task_arn}")

        # Step 3: Wait for tasks to stop (optional, for confirmation)
        if task_arns:
            waiter = ecs_client.get_waiter('tasks_stopped')
            waiter.wait(cluster=cluster_name, tasks=task_arns)
            print(f"All tasks in cluster {cluster_name} have stopped")

        # Step 4: Optionally delete the cluster
        if delete_cluster:
            ecs_client.delete_cluster(cluster=cluster_name)
            print(f"Deleted cluster: {cluster_name}")
        else:
            print(f"Cluster {cluster_name} remains active but has no running tasks (no charges)")

        return True

    except Exception as e:
        print(f"Error stopping ECS cluster {cluster_name}: {str(e)}")
        return False

if __name__ == '__main__':
    # Set delete_cluster=True if you want to delete the cluster entirely
    stop_ecs_cluster(cluster_name='my-ecs-cluster', delete_cluster=False)