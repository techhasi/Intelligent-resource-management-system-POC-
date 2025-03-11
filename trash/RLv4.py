# %% [markdown]
# ## **Deep Q-Network for Resource Scaling**
# **A DQN agent is used to decide whether to scale up, scale down, or take no action.**

# %% [markdown]
# 1. Import libraries

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from sklearn.metrics import mean_absolute_error, mean_squared_error

# %% [markdown]
# 2. Define RL environment

# %%
#Define the RL Environment using CSV datasets with a fixed reward function.
class ResourceScalingEnvCSV:
    def __init__(self, ec2_file="reduced_ec2_data.csv", 
                 rds_file="reduced_rds_data.csv", 
                 ecs_file="reduced_ecs_data.csv"):
        # Load CSV files and select the appropriate CPU utilization column,
        # converting values to floats and filling missing values with 0.
        self.ec2_data = pd.read_csv(ec2_file)["EC2_CPUUtilization"].apply(pd.to_numeric, errors='coerce').fillna(0).values
        self.rds_data = pd.read_csv(rds_file)["RDS_CPUUtilization"].apply(pd.to_numeric, errors='coerce').fillna(0).values
        self.ecs_data = pd.read_csv(ecs_file)["ECS_CPUUtilization"].apply(pd.to_numeric, errors='coerce').fillna(0).values

        # Use the shortest dataset length for a consistent episode length.
        self.length = min(len(self.ec2_data), len(self.rds_data), len(self.ecs_data))
        self.state_size = 3  # one value per service
        self.action_size = 3  # 0: scale up, 1: scale down, 2: no action
        # Set target utilization lower than the typical base value (e.g., 0.4)
        self.target = np.array([0.4, 0.4, 0.4])
        self.index = 0
        self.effect = np.zeros(self.state_size)
    
    def reset(self):
        self.index = 0
        self.effect = np.zeros(self.state_size)
        base_state = np.array([self.ec2_data[self.index], 
                               self.rds_data[self.index], 
                               self.ecs_data[self.index]])
        state = np.clip(base_state + self.effect, 0, 1)
        return state
    
    def step(self, action):
        # Modify the effect based on action:
        # Scale Up (action 0): reduces effective utilization (subtract 0.1)
        # Scale Down (action 1): increases effective utilization (add 0.1)
        # No Action (action 2): leaves effect unchanged.
        if action == 0:
            self.effect -= 0.1
        elif action == 1:
            self.effect += 0.1

        self.index += 1
        done = False
        if self.index >= self.length:
            done = True
            self.index = self.length - 1  # Clamp to last index
        
        base_state = np.array([self.ec2_data[self.index], 
                               self.rds_data[self.index], 
                               self.ecs_data[self.index]])
        state = np.clip(base_state + self.effect, 0, 1)
        # Compute reward as negative average absolute error from the target, scaled by 3.
        reward = -3 * np.mean(np.abs(state - self.target))
        return state, reward, done

# %% [markdown]
# 3. Defining DQN

# %%
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# %% [markdown]
# 4. Define the DQN Agent with a target network and gradient clipping

# %%
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.98          # discount factor
        self.epsilon = 1.0         # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.998
        self.learning_rate = 0.0003  # lowered learning rate
        self.model = DQN(state_size, action_size)
        # Create a target network and initialize it with the same weights.
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state_tensor)
        return torch.argmax(action_values).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                # Use the target network for a more stable target calculation.
                target = reward + self.gamma * torch.max(self.target_model(next_state_tensor)).item()
            predicted = self.model(state_tensor)[0][action]
            loss = self.criterion(predicted, torch.tensor(target).float())
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients to stabilize training.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# %% [markdown]
# 5. Main Training Loop 

# %%
def train_rl_agent(episodes=2000):
    env = ResourceScalingEnvCSV()
    agent = DQNAgent(env.state_size, env.action_size)
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        done = False
        while not done and step < env.length:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1
        agent.replay(64)
        # Update target network every 25 episodes.
        if (episode + 1) % 25 == 0:
            agent.update_target_model()
        avg_reward = total_reward / step  # average reward per step
        if (episode+1) % 50 == 0:
            print(f"Episode {episode+1}/{episodes}, Avg Reward per Step: {avg_reward:.4f}, Epsilon: {agent.epsilon:.4f}")
    torch.save(agent.model.state_dict(), "dqn_scaling_model.pth")
    print("DQN model saved!")

# %% [markdown]
# 6. Evaluate the Trained RL Agent

# %%
def evaluate_rl_agent(episodes=100):
    env = ResourceScalingEnvCSV()
    agent = DQNAgent(env.state_size, env.action_size)
    agent.model.load_state_dict(torch.load("dqn_scaling_model.pth"))
    agent.model.eval()
    
    total_rewards = []
    action_counts = {0: 0, 1: 0, 2: 0}
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        done = False
        while not done and step < env.length:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            action_counts[action] += 1
            state = next_state
            step += 1
        avg_reward = total_reward / step
        total_rewards.append(avg_reward)
    
    overall_avg = np.mean(total_rewards)
    action_distribution = {k: v / sum(action_counts.values()) for k, v in action_counts.items()}
    
    mae = mean_absolute_error(total_rewards, np.zeros_like(total_rewards))
    mse = mean_squared_error(total_rewards, np.zeros_like(total_rewards))
    rmse = np.sqrt(mse)
    
    print(f"Average Reward per Step over evaluation episodes: {overall_avg:.4f}")
    print(f"Action Distribution: {action_distribution}")
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
    
    return overall_avg, action_distribution, mae, mse, rmse



# %% [markdown]
# 7. Run training and evaluation

# %%

if __name__ == "__main__":
    train_rl_agent()
    evaluate_rl_agent()


