import gym
import numpy as np
import torch
import wandb
from gym.wrappers import RecordVideo#,Monitor

import argparse
import pickle
import random
import sys

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from tqdm import tqdm


if __name__ == '__main__':
    env_name = 'hopper'
    device = 'cuda'
    mode = 'normal'
    dataset = 'medium'

    # load dataset
    dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    

    #make env
    if env_name == 'hopper':
        env = gym.make('Hopper-v4')
        max_ep_len = 1000
        env_targets = [3200, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
    elif env_name == 'pusher':
        env = gym.make('Pusher-v4')
        max_ep_len = 100
        env_targets = [-20, -30]
        scale = 10.
    else:
        raise NotImplementedError

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    target_rew = env_targets[0]

    #init model
    model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=20,
            max_ep_len=max_ep_len,
            hidden_size=128,
            n_layer=3,
            n_head=1,
            n_inner=4*128,
            activation_function='relu',
            n_positions=1024,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )
    model.load_state_dict(torch.load(f'models/{dataset}/{env_name}-model.pth'))

    env = RecordVideo(env,'video')
    
    Average_returns = 0
    Max_returns = 0
    for i in tqdm(range(100)):
        episode_return, episode_length = evaluate_episode_rtg(env,state_dim,act_dim,model,max_ep_len=max_ep_len,scale=scale,target_return=target_rew/scale,mode=mode,state_mean=state_mean,state_std=state_std,device=device,is_train=False)
        print(f'episode_return: {episode_return}, episode_length: {episode_length}')
        Average_returns += episode_return
        if episode_return > Max_returns:
            Max_returns = episode_return

    print(f'Average_returns:{Average_returns/100} | Max_returns:{Max_returns}')
    env.close()

