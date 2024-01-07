import gym
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

env = gym.make('LunarLander-v2', continuous = False)
state_dim = 8
action_n = 4

class CEM(nn.Module):
    def __init__(self, state_dim, action_n):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n
        
        self.network = nn.Sequential(nn.Linear(self.state_dim, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, self.action_n)
                                     )
        
        
        
        self.softmax = nn.Softmax()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.0225)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, _input):
        return self.network(_input)
        
        
    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.forward(state)
        probs = self.softmax(logits).data.numpy()
        action = np.random.choice(self.action_n, p = probs)
        return action
    
    def fit(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                elite_states.append(state)
                elite_actions.append(action)
                
        elite_states = torch.FloatTensor(elite_states)
        elite_actions = torch.LongTensor(elite_actions)
        pred_actions = self.forward(elite_states)
        
        loss = self.loss(pred_actions, elite_actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()   

        
def get_trajectory(env, agent, max_len=1000, visualize=False):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    state = env.reset()

    for _ in range(max_len):
        trajectory['states'].append(state)
        
        action = agent.get_action(state)
        trajectory['actions'].append(action)
        
        state, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)
    
        if visualize:
            time.sleep(0.05)
            env.render()

        if done:
            break
    
    return trajectory

# agent = CEM(state_dim, action_n, neurons)
q_param = 0.5
iteration_n = 50
trajectory_n = 100
traj_max_len = 300

all_rewards = []
params_range = [25, 50, 75, 100, 125, 150, 200, 250, 300]

for trajectory_n in params_range:
    
    agent = CEM(state_dim, action_n)
    
    print("trajectory_n", trajectory_n)
    rewards_array = []
    
    for iteration in range(iteration_n):

        #policy evaluation
        trajectories = [get_trajectory(env, agent, max_len = traj_max_len) for _ in range(trajectory_n)]
        total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
        print('iteration:', iteration, 'mean total reward:', np.mean(total_rewards))
        rewards_array.append(np.mean(total_rewards))
        
        #policy improvement
        quantile = np.quantile(total_rewards, q_param)
        elite_trajectories = []
        for trajectory in trajectories:
            total_reward = np.sum(trajectory['rewards'])
            if total_reward > quantile:
                elite_trajectories.append(trajectory)
                
        if len(elite_trajectories) > 0:
            agent.fit(elite_trajectories)
            
    all_rewards.append(rewards_array)
    
    trajectory = get_trajectory(env, agent, max_len=1000, visualize=False)
    print('total reward:', sum(trajectory['rewards']))
    print('model:')
    
for rewards, neurons_num in zip(all_rewards, params_range):
    plt.plot(np.arange(0, iteration_n), rewards, label = np.round(neurons_num, 4))
    
plt.xlabel("Iteration", fontsize = 10)
plt.ylabel("Mean total reward", fontsize = 10)
plt.title(f"Reward dependence on number of trajectories, learning_rate: {0.0225} \n"
          f"Quantile: {q_param} \n"
          f"Max trajectory length: {traj_max_len}", fontsize = 10)
plt.legend(loc = "best")
plt.savefig('trajectory_len')