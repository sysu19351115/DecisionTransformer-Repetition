import gymnasium as gym
import gymnasium
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
import numpy as np
from tqdm import tqdm
import pickle
import collections


env_name = "Pusher-v4"
env = gym.make(env_name,render_mode = 'human')
checkpoint_path = 'pusher-ppo/PPO_Pusher-v4_4d805_00000_0_2023-10-26_19-31-47/checkpoint_001360'

"""
config = PPOConfig()
(
    config
    .rollouts(num_rollout_workers=20)
    .resources(num_gpus=0)
    .environment(env="Pusher-v4")
    .framework("torch")
    .training(
        lr=0.001,
        gamma=0.999,
        train_batch_size=1000,
        vf_loss_coeff=0.1,
        entropy_coeff=0.1,
        clip_param=0.2,
        sgd_minibatch_size=128,
        lambda_=0.95,
        grad_clip=0.5
    )
)

config.fcnet_hiddens = [256, 256]
config.fcnet_activation = "relu"
config.clip_rewards = True
config.observation_filter = "MeanStdFilter"

algo=config.build()

algo.restore(checkpoint_path)
"""

algo = Algorithm.from_checkpoint(checkpoint_path)
# algo = Algorithm.from_checkpoint("/workspaces/Pusher-v4/checkpoint_000140")

trajectories = []

for i in tqdm(range(100)):
    #print(f"--------------start episode {i}--------------------------")
    current_trace = {}
    observations = []
    next_observations = []
    actions = []
    rewards = []
    terminals = []

    episode_reward = 0
    terminated = truncated = False

    obs, info = env.reset()

    while not terminated and not truncated:
        observations.append(obs)

        action = algo.compute_single_action(obs,explore=False)
        actions.append(action)
        #print(action)

        obs, reward, terminated, truncated, info = env.step(action)
        next_observations.append(obs)
        rewards.append(reward)
        terminals.append(terminated)
        
        print(info)
        episode_reward += reward
    #print("reward_sum:",episode_reward)

    observations = np.array(observations)
    next_observations = np.array(next_observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    terminals = np.array(terminals)
    current_trace['observations'] = observations
    current_trace['next_observations'] = next_observations
    current_trace['actions'] = actions
    current_trace['rewards'] = rewards
    current_trace['terminals'] = terminals
    trajectories.append(current_trace)

    #if i%10000 == 0:
        #with open(f'pusher-medium-v2-{i}trace.pkl', 'wb') as f:
            #pickle.dump(trajectories, f)

#with open(f'pusher-medium-v2.pkl', 'wb') as f:
    #pickle.dump(trajectories, f)

