import boto3

# Initialize RDS client
rds_client = boto3.client('rds', region_name='ap-southeast-1')

def stop_all_rds_instances():
    try:
        # Step 1: List all running RDS instances
        response = rds_client.describe_db_instances()
        instances = []
        for db_instance in response['DBInstances']:
            if db_instance['DBInstanceStatus'] in ['available', 'starting']:
                instances.append(db_instance['DBInstanceIdentifier'])

        print(f"Found {len(instances)} running RDS instances")

        # Step 2: Stop all running instances
        for db_instance_id in instances:
            try:
                rds_client.stop_db_instance(DBInstanceIdentifier=db_instance_id)
                print(f"Stopping RDS instance: {db_instance_id}")
            except rds_client.exceptions.InvalidDBInstanceStateFault:
                print(f"RDS instance {db_instance_id} cannot be stopped (e.g., Aurora cluster)")
                continue

        # Step 3: Wait for instances to stop (optional, for confirmation)
        if instances:
            for db_instance_id in instances:
                waiter = rds_client.get_waiter('db_instance_stopped')
                try:
                    waiter.wait(DBInstanceIdentifier=db_instance_id)
                    print(f"RDS instance {db_instance_id} has stopped")
                except Exception as wait_error:
                    print(f"Error waiting for {db_instance_id} to stop: {str(wait_error)}")
        else:
            print("No running RDS instances found")

        return True

    except Exception as e:
        print(f"Error stopping RDS instances: {str(e)}")
        return False

if __name__ == '__main__':
    stop_all_rds_instances()