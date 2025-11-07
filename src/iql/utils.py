import torch
import torch.nn as nn
from typing import Iterable
import pandas as pd
import numpy as np
from tqdm import tqdm
from .batch import Batch


def fanin_init(m: nn.Module):
    """Initialize Linear layers uniformly in ±1/sqrt(fan_in)."""
    if isinstance(m, nn.Linear):
        bound = 1.0 / (m.weight.size(1) ** 0.5)
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.uniform_(m.bias, -bound, bound)


def build_mlp(in_dim: int, hidden: Iterable[int], out_dim: int, act=nn.ReLU) -> nn.Sequential:
    """Build MLP: [Linear -> Activation] * N then Linear -> out_dim."""
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), act()]
        prev = h
    layers += [nn.Linear(prev, out_dim)]
    net = nn.Sequential(*layers)
    net.apply(fanin_init)
    return net

def preprocess_kuairand(data_loader, env, max_transitions_per_user: int = None):
    """
    Convert KuaiRand logged data into offline RL transitions.

    Args:
        data_loader: KuaiRandDataLoader instance
        env: KuaiRandEnv instance
        max_transitions_per_user: Maximum transitions per user (None = all)

    Returns:
        Batch object with (obs, act, rew, next_obs, done)
    """
    from .batch import Batch

    all_obs = []
    all_acts = []
    all_rews = []
    all_next_obs = []
    all_dones = []

    n_users = data_loader.get_n_users()
    n_videos = data_loader.get_n_videos()

    print(f"Preprocessing {n_users} users into offline dataset...")

    for user_idx in tqdm(range(n_users), desc="Processing users"):
        # Get user's interaction history
        history = data_loader.get_user_history(user_idx)
        video_seq = history['video_sequence']

        if len(video_seq) < 2:
            continue

        # Limit transitions per user if specified
        max_len = len(video_seq) - 1
        if max_transitions_per_user is not None:
            max_len = min(max_len, max_transitions_per_user)

        # Set environment to this user and extract transitions
        env.current_user_idx = user_idx

        for step in range(max_len):
            # Current state
            env.current_step = step
            env.episode_history = []
            for i in range(step):
                env.episode_history.append({
                    'video_idx': video_seq[i],
                    'reward': history['clicks'][i] * 0.5 + history['watch_ratios'][i] * 0.5
                })
            state = env._get_state()

            # Action taken (video recommended)
            action_discrete = video_seq[step]
            # Convert to continuous action [-1, 1] for IQL policy
            action_continuous = (action_discrete / (n_videos - 1)) * 2.0 - 1.0

            # Reward: 50/50 click + watch
            reward = history['clicks'][step] * 0.5 + history['watch_ratios'][step] * 0.5

            # Next state
            env.current_step = step + 1
            env.episode_history.append({
                'video_idx': action_discrete,
                'reward': reward
            })
            next_state = env._get_state()

            # Done flag (last transition for this user)
            done = float(step == len(video_seq) - 2)

            all_obs.append(state)
            all_acts.append(np.array([action_continuous], dtype=np.float32))
            all_rews.append(reward)
            all_next_obs.append(next_state)
            all_dones.append(done)

    print(f"\nDone! Created {len(all_obs)} offline transitions from {n_users} users")

    # Convert to tensors
    obs_tensor = torch.tensor(np.array(all_obs), dtype=torch.float32)
    act_tensor = torch.tensor(np.array(all_acts), dtype=torch.float32)
    rew_tensor = torch.tensor(np.array(all_rews), dtype=torch.float32)
    next_obs_tensor = torch.tensor(np.array(all_next_obs), dtype=torch.float32)
    done_tensor = torch.tensor(np.array(all_dones), dtype=torch.float32)

    return Batch(
        obs=obs_tensor,
        act=act_tensor,
        rew=rew_tensor,
        next_obs=next_obs_tensor,
        done=done_tensor
    )
