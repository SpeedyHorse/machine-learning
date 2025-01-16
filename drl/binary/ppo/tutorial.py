import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("CartPole-v1")

lr = 0.002
gamma = 0.99
epsilon = 0.2
k_epochs = 4
episodes = 500
batch_size = 64

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        shared = self.shared_layer(x)
        policy = self.actor(shared)
        value = self.critic(shared)
        return policy, value

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=lr) 

def compute_gae(rewards, values, next_value, gamma=0.99, lambda_=0.95):
    values = torch.cat((values, torch.tensor([next_value], dtype=torch.float32)))
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lambda_ * gae
        advantages.insert(0, gae)
    return torch.tensor(advantages, dtype=torch.float32)

def ppo():
    for episode in range(episodes):
        state  =env.reset()[0]
        log_probs, values, rewards, states, actions = [], [], [], [], []
        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            policy, value = model(state_tensor)
            dist = Categorical(policy)
            action = dist.sample()

            states.append(state_tensor)
            actions.append(action)
            log_probs.append(dist.log_prob(action))
            values.append(value)

            next_state, reward, done, _, _ = env.step(action.item())
            rewards.append(reward)
            state = next_state

            if done:
                next_value = 0 if done else model(torch.tensor(next_state, dtype=torch.float32))[1]
                advantages = compute_gae(rewards, torch.tensor(values), next_value)
                returns = advantages + torch.tensor(values)
                break

        states = torch.stack(states).detach()
        actions = torch.stack(actions).detach()
        old_log_probs = torch.stack(log_probs).detach()
        values = torch.tensor(values).detach()

        for _ in range(k_epochs):
            policy, value = model(states)
            dist = Categorical(policy)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
            actor_loss = - torch.min(surr1, surr2).mean()

            critic_loss = nn.functional.mse_loss(value.squeeze(), returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if episode % 50 == 0:
            print(f"Episode: {episode}, Total Reward: {sum(rewards)}")
    
    print("Training Complete")

ppo()

state = env.reset()[0]
done = False

while not done:
    env.render()
    state_tensor = torch.tensor(state, dtype=torch.float32)
    policy, _ = model(state_tensor)
    action = policy.argmax().item()
    state, _, done, _, _ = env.step(action)

env.close()