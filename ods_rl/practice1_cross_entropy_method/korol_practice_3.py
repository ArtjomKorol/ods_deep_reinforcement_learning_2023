import gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 7)

env = gym.make("Taxi-v3")
    
class CrossEntropyAgent():
    def __init__(self, state_n, action_n):
        self.state_n = state_n
        self.action_n = action_n
        self.model = np.ones((state_n, action_n)) / action_n
    
    def get_action(self, state):
        action = np.random.choice(np.arange(self.action_n), p = self.model[state])
        return int(action)
    
    def fit(self, elite_trajectories):
        new_model = np.zeros((state_n, action_n))
        
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1
        
        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()
            
        self.model = new_model
        
        return None
    
       
    
def get_state(obs):
    return int(obs)

def get_trajectory(env, agent, max_len=1000, visualize=False):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    obs = env.reset()
    state = get_state(obs)
    
    for _ in range(max_len):
        trajectory['states'].append(state)
        
        action = agent.get_action(state)
        trajectory['actions'].append(action)
        
        obs, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)
        
        state = get_state(obs)
        
        if visualize:
            time.sleep(0.1)
            env.render()
        
        if done:
            break
        
    return trajectory

action_n = 6
state_n = 500

q_param = 0.975
trajectory_n = 100
iteration_n = 100
max_len = 1000
m_param = 15


total_mean_rewards = []
param_range = [10, 20, 30, 40, 50]

for m_param in param_range:
    print('M: ', m_param)
    
    agent = CrossEntropyAgent(state_n, action_n)
    
    iter_reward = []
    
    for iteration in range(iteration_n):
        
        trajectories_m = []
        mean_rewards_arr = []
        
        # evaluation
        for _ in range(m_param):
            trajectories = [get_trajectory(env, agent, max_len) for _ in range(trajectory_n)]
            trajectories_m.append(trajectories)
            
        for trajectories in trajectories_m:
            total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
            mean_rewards_arr.append(np.mean(total_rewards))
            
        print('iteration:', iteration,'mean reward:', np.mean(mean_rewards_arr), 'from ', m_param, ' samples')
        iter_reward.append(np.mean(mean_rewards_arr))
        
        # improvement
        quantile = np.quantile(mean_rewards_arr, q_param)
        elite_trajectories = []
        for ix, reward in enumerate(mean_rewards_arr):
            if reward > quantile:
                elite_trajectories.extend(trajectories_m[ix])
                
        agent.fit(elite_trajectories)
        
    total_mean_rewards.append(iter_reward)

for rewards, trajectory_len in zip(total_mean_rewards, param_range):
    plt.plot(np.arange(0, iteration_n), rewards, label = np.round(trajectory_len, 2))
    
plt.xlabel("Iteration", fontsize = 14)
plt.ylabel("Mean total reward", fontsize = 14)
plt.title(f"Rewards dependence on M param, q_param: {q_param}, trajectory_len: {trajectory_n}, max_len: {max_len}", fontsize = 16)
plt.legend(loc = "best")
plt.savefig('stochastic_rewards')

# trajectory = get_trajectory(env, agent, max_len = 100, visualize = False)
# print('total reward', np.sum(trajectory['rewards']))

