# import pandas as pd

# # Load your dataset
# data = pd.read_csv('cloud-computing-performance-metrics.csv')

# # Separate the header and data
# header = data.columns
# data = data.iloc[1:]  # Exclude the header row

# # Determine the split size
# total_rows = len(data)
# split_size = total_rows // 3

# # Split the data and create copies
# ec2_data = data.iloc[:split_size].copy()  # First split for EC2
# rds_data = data.iloc[split_size:2 * split_size].copy()  # Second split for RDS
# ecs_data = data.iloc[2 * split_size:].copy()  # Remaining data for ECS

# # Add and process EC2-specific metrics
# ec2_data.loc[:, 'EC2_CPUUtilization'] = ec2_data['cpu_usage']
# ec2_data.loc[:, 'EC2_MemoryUtilization'] = ec2_data['memory_usage']
# ec2_data.loc[:, 'EC2_DiskWriteOps'] = ec2_data['num_executed_instructions']  # Simulated
# ec2_data.loc[:, 'EC2_NetworkIn'] = ec2_data['network_traffic']  # Simulated

# # Add and process RDS-specific metrics
# rds_data.loc[:, 'RDS_CPUUtilization'] = rds_data['cpu_usage']
# rds_data.loc[:, 'RDS_FreeableMemory'] = 100 - rds_data['memory_usage']  # Simulated freeable memory
# rds_data.loc[:, 'RDS_DatabaseConnections'] = rds_data['task_status'].apply(lambda x: 1 if x == 'running' else 0)  # Simulated
# rds_data.loc[:, 'RDS_WriteIOPS'] = rds_data['num_executed_instructions']  # Simulated

# # Add and process ECS-specific metrics
# ecs_data.loc[:, 'ECS_CPUUtilization'] = ecs_data['cpu_usage']
# ecs_data.loc[:, 'ECS_MemoryUtilization'] = ecs_data['memory_usage']
# ecs_data.loc[:, 'ECS_RunningTaskCount'] = ecs_data['task_status'].apply(lambda x: 1 if x == 'running' else 0)  # Simulated

# unwanted_columns = [
#     'vm_id', 'power_consumption', 'num_executed_instructions', 
#     'execution_time', 'energy_efficiency', 'task_type', 
#     'task_priority', 'task_status', 'cpu_usage', 'memory_usage', 'network_traffic'
# ]

# ec2_data.drop(columns=unwanted_columns, inplace=True)
# rds_data.drop(columns=unwanted_columns, inplace=True)
# ecs_data.drop(columns=unwanted_columns, inplace=True)

# # Save the processed datasets
# ec2_data.to_csv('ec2_data.csv', index=False)
# rds_data.to_csv('rds_data.csv', index=False)
# ecs_data.to_csv('ecs_data.csv', index=False)

# # Verify output
# print("EC2 Data Columns:", ec2_data.columns)
# print("RDS Data Columns:", rds_data.columns)
# print("ECS Data Columns:", ecs_data.columns)

# # Optional: Print the sizes of the splits
# print(f"EC2 Data Points: {len(ec2_data)}")
# print(f"RDS Data Points: {len(rds_data)}")
# print(f"ECS Data Points: {len(ecs_data)}")

import pandas as pd

# Load your dataset
data = pd.read_csv('cloud-computing-performance-metrics.csv')

# Separate the header and data
header = data.columns
data = data.iloc[1:]  # Exclude the header row

# Set the maximum number of datapoints per service
MAX_DATAPOINTS = 100000

# Determine the split size (use min to ensure we don't exceed MAX_DATAPOINTS)
total_rows = len(data)
split_size = min(total_rows // 3, MAX_DATAPOINTS)

# Split the data and create copies, limiting each to MAX_DATAPOINTS
ec2_data = data.iloc[:split_size].copy()  # First split for EC2
rds_data = data.iloc[split_size:2 * split_size].copy()  # Second split for RDS
ecs_data = data.iloc[2 * split_size:3 * split_size].copy()  # Third split for ECS

# Add and process EC2-specific metrics
ec2_data.loc[:, 'EC2_CPUUtilization'] = ec2_data['cpu_usage']
ec2_data.loc[:, 'EC2_MemoryUtilization'] = ec2_data['memory_usage']
ec2_data.loc[:, 'EC2_DiskWriteOps'] = ec2_data['num_executed_instructions']  # Simulated
ec2_data.loc[:, 'EC2_NetworkIn'] = ec2_data['network_traffic']  # Simulated

# Add and process RDS-specific metrics
rds_data.loc[:, 'RDS_CPUUtilization'] = rds_data['cpu_usage']
rds_data.loc[:, 'RDS_FreeableMemory'] = 100 - rds_data['memory_usage']  # Simulated freeable memory
rds_data.loc[:, 'RDS_DatabaseConnections'] = rds_data['task_status'].apply(lambda x: 1 if x == 'running' else 0)  # Simulated
rds_data.loc[:, 'RDS_WriteIOPS'] = rds_data['num_executed_instructions']  # Simulated

# Add and process ECS-specific metrics
ecs_data.loc[:, 'ECS_CPUUtilization'] = ecs_data['cpu_usage']
ecs_data.loc[:, 'ECS_MemoryUtilization'] = ecs_data['memory_usage']
ecs_data.loc[:, 'ECS_RunningTaskCount'] = ecs_data['task_status'].apply(lambda x: 1 if x == 'running' else 0)  # Simulated

unwanted_columns = [
    'vm_id', 'power_consumption', 'num_executed_instructions', 
    'execution_time', 'energy_efficiency', 'task_type', 
    'task_priority', 'task_status', 'cpu_usage', 'memory_usage', 'network_traffic'
]

ec2_data.drop(columns=unwanted_columns, inplace=True)
rds_data.drop(columns=unwanted_columns, inplace=True)
ecs_data.drop(columns=unwanted_columns, inplace=True)

# Save the processed datasets
ec2_data.to_csv('reduced_ec2_data.csv', index=False)
rds_data.to_csv('reduced_rds_data.csv', index=False)
ecs_data.to_csv('reduced_ecs_data.csv', index=False)

# Print information about the datasets
print("\nDataset Information:")
print(f"EC2 Data Points: {len(ec2_data):,}")
print(f"RDS Data Points: {len(rds_data):,}")
print(f"ECS Data Points: {len(ecs_data):,}")

print("\nColumns in each dataset:")
print("EC2 Data Columns:", ec2_data.columns.tolist())
print("RDS Data Columns:", rds_data.columns.tolist())
print("ECS Data Columns:", ecs_data.columns.tolist())

# Optional: Print warning if original data was truncated
if total_rows > MAX_DATAPOINTS * 3:
    print(f"\nWarning: Original dataset contained {total_rows:,} rows.")
    print(f"Each service dataset has been limited to {MAX_DATAPOINTS:,} rows.")
    print(f"Total rows used: {len(ec2_data) + len(rds_data) + len(ecs_data):,}")
    print(f"Rows not used: {total_rows - (len(ec2_data) + len(rds_data) + len(ecs_data)):,}")