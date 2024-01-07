import gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 7)

env = gym.make("Taxi-v3")
    
class CrossEntropyAgent():
    def __init__(self, state_n, action_n, smooth_lambda):
        self.state_n = state_n
        self.action_n = action_n
        self.smooth_lambda = smooth_lambda
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
            
        self.model = self.smooth_lambda * new_model + self.model * (1 - self.smooth_lambda)
        
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

q_param = 0.8
trajectory_n = 2000
iteration_n = 40
max_len = 1000

# optimal params
# q_param = 0.8
# trajectory_n = 2000
# iteration_n = 20
# max_len = 1000

total_mean_rewards = []
param_range = np.arange(0.1, 1, 0.1)
for smooth_lambda in param_range:
    print('labmda: ', smooth_lambda)
    
    agent = CrossEntropyAgent(state_n, action_n, smooth_lambda)
    
    mean_rewards_arr = []
    
    for iteration in range(iteration_n):
        
        # evaluation
        trajectories = [get_trajectory(env, agent, max_len) for _ in range(trajectory_n)]
        total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
        print('iteration:', iteration,'mean total reward:', np.mean(total_rewards))
        mean_rewards_arr.append(np.mean(total_rewards))
        
        # improvement
        quantile = np.quantile(total_rewards, q_param)
        elite_trajectories = [trajectory for trajectory in trajectories if np.sum(trajectory['rewards']) > quantile]
        
        agent.fit(elite_trajectories)
        
    total_mean_rewards.append(mean_rewards_arr)

for rewards, trajectory_len in zip(total_mean_rewards, param_range):
    plt.plot(np.arange(0, iteration_n), rewards, label = np.round(trajectory_len, 2))
    
plt.xlabel("Iteration", fontsize = 14)
plt.ylabel("Mean total reward", fontsize = 14)
plt.title(f"Rewards dependence on policy smoothing labmda, q_param: {q_param}, trajectory_len: {trajectory_n}, max_len: {max_len}", fontsize = 16)
plt.legend(loc = "best")
plt.savefig('policy_smoothing_rewards')

# trajectory = get_trajectory(env, agent, max_len = 100, visualize = False)
# print('total reward', np.sum(trajectory['rewards']))

