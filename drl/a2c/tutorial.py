import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("CartPole-v1")

lr = 0.002
gamma = 0.99
episodes = 500

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

def a2c():
    all_rewards = []
    for episode in range(episodes):
        state = env.reset()[0]
        log_probs = []
        values = []
        rewards = []

        while True:
            # select action
            state_tensor = torch.tensor(state, dtype=torch.float32)
            policy, value = model(state_tensor)
            action_dist = torch.distributions.Categorical(policy)
            action = action_dist.sample()

            # store
            log_probs.append(action_dist.log_prob(action))
            values.append(value)

            next_state, reward, done, _, _ = env.step(action.item())

            rewards.append(reward)
            state = next_state

            if done:
                break
        
        all_rewards.append(np.sum(rewards))
        
        # optimize
        cumulative_reward = 0
        discounted_rewards = []
        for r in reversed(rewards):
            cumulative_reward = r + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        values = torch.cat(values).squeeze()
        advantages = discounted_rewards - values.detach()

        actor_loss = - (torch.stack(log_probs) * advantages).sum()
        critic_loss = nn.functional.mse_loss(values, discounted_rewards)
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 50 == 0:
            print(f"Episode: {episode}, Total Reward: {sum(rewards)}")
    
    plt.figure(figsize=(15,5))
    plt.plot(all_rewards)
    means = [all_rewards[0]]
    for i in range(1, len(all_rewards)):
        means.append(np.mean(all_rewards[0:i]))
    plt.plot(means, color="red")
    plt.show()
    
    print("Training Complete")

a2c()

state = env.reset()[0]

done = False

while not done:
    env.render()
    state_tensor = torch.tensor(state, dtype=torch.float32)
    policy, _ = model(state_tensor)
    action = policy.argmax().item()
    state, _, done, _, _ = env.step(action)

env.close()