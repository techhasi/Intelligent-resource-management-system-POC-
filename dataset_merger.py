import pandas as pd
import sqlite3
from datetime import datetime

# Read the CSV files and convert timestamps
ec2_df = pd.read_csv('ec2_data.csv')
rds_df = pd.read_csv('rds_data.csv')
ecs_df = pd.read_csv('ecs_data.csv')

# Convert timestamps
ec2_df['timestamp'] = pd.to_datetime(ec2_df['timestamp'])
rds_df['timestamp'] = pd.to_datetime(rds_df['timestamp'])
ecs_df['timestamp'] = pd.to_datetime(ecs_df['timestamp'])

# Create database connection
conn = sqlite3.connect('merged_metrics.db')

# Store dataframes in SQLite
ec2_df.to_sql('ec2_data', conn, if_exists='replace', index=False)
rds_df.to_sql('rds_data', conn, if_exists='replace', index=False)
ecs_df.to_sql('ecs_data', conn, if_exists='replace', index=False)

# Query with correct column names
query = """
WITH all_timestamps AS (
    SELECT DISTINCT timestamp 
    FROM (
        SELECT timestamp FROM ec2_data
        UNION ALL
        SELECT timestamp FROM rds_data
        UNION ALL
        SELECT timestamp FROM ecs_data
    )
)
SELECT 
    at.timestamp,
    ec2.EC2_CPUUtilization,
    ec2.EC2_MemoryUtilization,
    ec2.EC2_DiskWriteOps,
    ec2.EC2_NetworkIn,
    rds.RDS_CPUUtilization,
    rds.RDS_FreeableMemory,
    rds.RDS_DatabaseConnections,
    rds.RDS_WriteIOPS,
    ecs.ECS_CPUUtilization,
    ecs.ECS_MemoryUtilization,
    ecs.ECS_RunningTaskCount
FROM all_timestamps at
LEFT JOIN ec2_data ec2 ON at.timestamp = ec2.timestamp
LEFT JOIN rds_data rds ON at.timestamp = rds.timestamp
LEFT JOIN ecs_data ecs ON at.timestamp = ecs.timestamp
ORDER BY at.timestamp;
"""

# Execute query and load results
df = pd.read_sql_query(query, conn)

# Remove any duplicate rows
df = df.drop_duplicates()

# Close database connection
conn.close()

# Save the merged dataset
df.to_csv('merged_cloud_metrics.csv', index=False)

print("\nMerged dataset saved as 'merged_cloud_metrics.csv'")
print("\nColumns in the dataset:")
print(df.columns.tolist())
print("\nSample of merged data:")
print(df.head())