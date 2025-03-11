import boto3

# Initialize EC2 client
ec2_client = boto3.client('ec2', region_name='ap-southeast-1')

def stop_all_ec2_instances():
    try:
        # Step 1: List all running instances
        response = ec2_client.describe_instances(
            Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
        )
        instances = []
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                instances.append(instance['InstanceId'])

        print(f"Found {len(instances)} running EC2 instances")

        # Step 2: Stop all running instances
        if instances:
            ec2_client.stop_instances(InstanceIds=instances)
            print(f"Stopping instances: {', '.join(instances)}")

            # Step 3: Wait for instances to stop (optional, for confirmation)
            waiter = ec2_client.get_waiter('instance_stopped')
            waiter.wait(InstanceIds=instances)
            print("All EC2 instances have stopped")
        else:
            print("No running EC2 instances found")

        return True

    except Exception as e:
        print(f"Error stopping EC2 instances: {str(e)}")
        return False

if __name__ == '__main__':
    stop_all_ec2_instances()