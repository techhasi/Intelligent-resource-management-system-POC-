# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define RL Environment
class ResourceScalingEnv:
    def __init__(self):
        self.state_size = 3  # EC2, RDS, ECS predicted usage
        self.action_size = 3  # Scale up, scale down, no action
        self.state = np.zeros(self.state_size)
        self.reward = 0
    
    def reset(self):
        self.state = np.random.rand(self.state_size)  # Start with random usage
        return self.state
    
    def step(self, action):
        # Simulate impact of action
        if action == 0:  # Scale Up
            self.state += np.random.uniform(0.01, 0.05, self.state_size)
            self.reward = -abs(self.state.sum() - 0.8) if action == 0 else abs(self.state.sum() - 0.6) if action == 1 else -abs(self.state.sum() - 0.7)  # More usage, more cost
        elif action == 1:  # Scale Down
            self.state -= np.random.uniform(0.01, 0.05, self.state_size)
            self.reward = self.state.sum()  # Less cost, but risk of under-scaling
        else:  # No Action
            self.reward = -abs(self.state.sum() - 0.5)  # Penalty for over/under allocation
        
        self.state = np.clip(self.state, 0, 1)  # Ensure valid range
        return self.state, self.reward

# Define Deep Q-Network
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

# Train DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.98  # Discount factor
        self.epsilon = 1.0  # Exploration-exploitation balance
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.998
        self.learning_rate = 0.0005
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice([0, 1, 2])
        state_tensor = torch.FloatTensor(state).float()
        with torch.no_grad():
            action_values = self.model(state_tensor)
        return torch.argmax(action_values).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward + self.gamma * torch.max(self.model(torch.FloatTensor(next_state).float())).item()
            predicted_target = self.model(torch.FloatTensor(state))[action]
            loss = self.criterion(predicted_target, torch.tensor(target).float())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Main Training Loop
def train_rl_agent(episodes=2000):
    env = ResourceScalingEnv()
    agent = DQNAgent(env.state_size, env.action_size)
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(20):  # Simulate 10 steps per episode
            action = agent.act(state)
            next_state, reward = env.step(action)
            agent.remember(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        agent.replay(64)
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}")
    torch.save(agent.model.state_dict(), "dqn_scaling_model.pth")
    print("DQN model saved!")

# Evaluate RL Model
def evaluate_rl_agent(episodes=100):
    env = ResourceScalingEnv()
    agent = DQNAgent(env.state_size, env.action_size)
    agent.model.load_state_dict(torch.load("dqn_scaling_model.pth"))
    agent.model.eval()
    
    total_rewards = []
    action_counts = {0: 0, 1: 0, 2: 0}
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(10):
            action = agent.act(state)
            next_state, reward = env.step(action)
            total_reward += reward
            action_counts[action] += 1
            state = next_state
        total_rewards.append(total_reward)
    
    avg_reward = np.mean(total_rewards)
    action_distribution = {k: v / sum(action_counts.values()) for k, v in action_counts.items()}
    
    # Calculate accuracy metrics
    mae = mean_absolute_error(total_rewards, np.zeros_like(total_rewards))
    mse = mean_squared_error(total_rewards, np.zeros_like(total_rewards))
    rmse = np.sqrt(mse)
    
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"Action Distribution: {action_distribution}")
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
    
    return avg_reward, action_distribution, mae, mse, rmse

# Run training
train_rl_agent()

# Evaluate model
evaluate_rl_agent()



