import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import random

env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
lr = 0.001
gamma = 0.99
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 0.995
batch_size = 64
memory_size = 10000
target_update_freq = 10
episodes = 1000

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def size(self):
        return len(self.buffer)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        shared = self.shared_layer(x)
        advantage = self.advantage(shared)
        value = self.value(shared)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
    
class D3QNAgent:
    def __init__(self, state_dim, action_dim):
        self.policy_net = DuelingDQN(state_dim, action_dim)
        self.target_net = DuelingDQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(memory_size)
        self.epsilon = epsilon_start
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, action_dim - 1)
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()
    
    def train(self):
        if self.memory.size() < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # print(states.dim(), actions.dim())

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
        
        target_q_values = rewards + gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


agent = D3QNAgent(state_dim, action_dim)
episode_rewards = []

for episode in range(episodes):
    state = env.reset()[0]
    total_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.memory.add(state, action, reward, next_state, done)
        
        agent.train()

        if done:
            break

        state = next_state
        total_reward += reward

    if episode % target_update_freq == 0:
        agent.update_target_net()
    
    agent.epsilon = max(epsilon_final, agent.epsilon * epsilon_decay)

    episode_rewards.append(total_reward)
    
    if episode % 10 == 9:
        print(f"Episode: {episode + 1:5}, Reward: {total_reward:3}")

plt.figure(figsize=(15,5))
plt.plot(episode_rewards)
plt.show()

env.close()
