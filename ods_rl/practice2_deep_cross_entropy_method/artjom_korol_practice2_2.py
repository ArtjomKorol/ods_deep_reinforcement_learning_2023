import gym
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

env = gym.make('MountainCarContinuous-v0')

state_dim = env.observation_space.shape[0]
action_n = env.action_space.shape[0]
# total_reward = 0

# for i in range(10):
#     action = env.action_space.sample()
#     print(i, action)
#     state, reward, done, _ = env.step(action)
#     total_reward += reward
#     time.sleep(0.05)
#     env.render()
    
#     if done:
#         break
    
# print(reward)
    

class CEM(nn.Module):
    def __init__(self, state_dim, action_n):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n
        self.network = nn.Sequential(nn.Linear(self.state_dim, 16),
                                     nn.ReLU(),
                                     nn.Linear(16, self.action_n)
                                     )
        
        self.loss = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.01)
        self.eps = 0
    
    def forward(self, input):
        x = self.network(input)
        return torch.clamp(x, min = -1, max = 1)
        
    def get_action(self, state, iteration_n):
        state = torch.FloatTensor(state)
        
        with torch.no_grad():
            action = self.forward(state)
        
        noise = torch.Tensor([np.random.uniform(-1, 1)]) * self.eps
        action = action + noise
        
        return torch.clip(action, min = -1, max = 1)
    
    def fit(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                elite_states.append(state)
                elite_actions.append(action)
                
        elite_states = torch.FloatTensor(elite_states)
        elite_actions = torch.FloatTensor(elite_actions)
        pred_actions = self.forward(elite_states)
        
        loss = self.loss(torch.ravel(pred_actions), elite_actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
def get_trajectory(env, agent, n_iters, max_len=1000, visualize=False):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    state = env.reset()

    for _ in range(max_len):
        trajectory['states'].append(state)
        
        action = agent.get_action(state, iteration_n)
        trajectory['actions'].append(action)
        
        state, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)
    
        if visualize:
            time.sleep(0.05)
            env.render()

        if done:
            break
    
    return trajectory


reward_threshold = 0
iteration_n = 10
trajectory_n = 250
trajectory_len = 3000

agent = CEM(state_dim, action_n)

iter_reward = []

for iteration in range(iteration_n):
    
    agent.eps = 1 / ((iteration + 1) ** 0.1)

    #policy evaluation
    trajectories = [get_trajectory(env, agent, iteration_n - iteration, max_len = trajectory_len) for _ in range(trajectory_n)]
    total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
    print('iteration:', iteration, 'mean total reward:', np.mean(total_rewards))

    #policy improvement
    elite_trajectories = []
    for trajectory in trajectories:
        total_reward = np.sum(trajectory['rewards'])
        if total_reward > reward_threshold:
            elite_trajectories.append(trajectory)
            
    if len(elite_trajectories) > 0:
        print('elite trajectories: ', len(elite_trajectories))
        agent.fit(elite_trajectories)

    iter_reward.append(np.mean(total_rewards))
    
trajectory = get_trajectory(env, agent, 0, max_len=100, visualize=True)
print('total reward:', sum(trajectory['rewards']))


plt.plot(np.arange(10), iter_reward)
plt.xlabel("Iteration", fontsize = 10)
plt.ylabel("Mean total reward", fontsize = 10)
plt.title(f"Reward progress \n"
          f"Sample only trajectories with rewards > {reward_threshold} \n"
          f"Max trajectory length: {trajectory_len}", fontsize = 10)
plt.legend(loc = "best")
plt.savefig('cart_task')