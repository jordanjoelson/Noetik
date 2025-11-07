import numpy as np
import sys
import os
import torch

from src.data_loader import KuaiRandDataLoader
from src.environment import KuaiRandEnv
from agent import IQLAgent
from src.iql.config import IQLConfig

print("="*60)
print("KuaiRand IQL Agent Demo")
print("="*60)

print("\n[1/3] Loading dataset...")
loader = KuaiRandDataLoader()

try:
    loader.load_processed_data()
except FileNotFoundError:
    print("Processing dataset for first time...")
    loader.load_data(random_only=True)
    loader.save_processed_data()

print(f"Dataset: {loader.get_n_users():,} users, {loader.get_n_videos():,} videos")

print("\n[2/3] Creating environment...")
env = KuaiRandEnv(data_loader=loader)
print(f"State space: {env.observation_space.shape}")
print(f"Action space: {env.action_space.n:,} videos")

print("\n[3/3] Creating IQL agent...")
agent_config = IQLConfig(
    obs_dim=128,
    act_dim=1,
    hidden_sizes=(256, 256)
)
agent = IQLAgent(agent_config)
print(f"IQL Agent ready!")

print("\n" + "="*60)
print("Sample Episode with IQL Agent")
print("="*60)

state, info = env.reset()
print(f"\nUser {info['user_idx']}")

episode_reward = 0
for step in range(10):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    
    with torch.no_grad():
        action_continuous = agent.act(state_tensor, deterministic=True)
    
    scaled = (action_continuous + 1) / 2  # [0,1]
    action_discrete = (scaled * (env.action_space.n - 1)).round().long().item()
    
    next_state, reward, terminated, truncated, info = env.step(action_discrete)

    episode_reward += reward

    print(f"  Step {step+1}: video={action_discrete}, reward={reward:.3f}, "
          f"click={info['interaction']['is_click']}, "
          f"watch={info['interaction']['watch_ratio']:.2f}")

    state = next_state
    if terminated or truncated:
        break

print(f"\nTotal reward: {episode_reward:.3f}")