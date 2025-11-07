import numpy as np
import torch
from typing import Dict, List
import random

from src.environment import KuaiRandEnv
from agent import IQLAgent
from src.iql.config import IQLConfig
from src.iql.batch import Batch 

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = []
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer"""
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample batch of transitions"""
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

class IQLTrainer:
    def __init__(self, env: KuaiRandEnv, agent_config: IQLConfig, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.env = env
        self.device = device
        self.agent = IQLAgent(agent_config, device=device)
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
    def continuous_to_discrete(self, continuous_act: torch.Tensor, n_actions: int) -> int:
        """Map continuous action from [-1,1] to discrete [0, n_actions-1]"""
        # continuous_act is in [-1, 1], convert to [0, 1]
        scaled = (continuous_act + 1) / 2  
        # Scale to [0, n_actions-1] and round to integer
        discrete = (scaled * (n_actions - 1)).round().long()
        return discrete.clamp(0, n_actions - 1).item()
    
    def collect_rollouts(self, n_episodes: int = 1000):
        """Collect experience and train the agent"""
        metrics = {
            'episode_rewards': [],
            'episode_clicks': [],
            'avg_q_values': []
        }
        
        for episode in range(n_episodes):
            state, info = self.env.reset()
            episode_reward = 0
            episode_clicks = 0
            
            for step in range(self.env.max_episode_length):
                # Convert state to tensor and get action
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                with torch.no_grad():
                    action_continuous = self.agent.act(state_tensor, deterministic=False)
                
                # Convert to discrete action for environment
                action_discrete = self.continuous_to_discrete(
                    action_continuous, self.env.action_space.n
                )
                
                # Environment step
                next_state, reward, terminated, truncated, info = self.env.step(action_discrete)
                episode_reward += reward
                episode_clicks += info['interaction']['is_click']
                
                # Store transition (convert to tensors)
                # Ensure action is stored as 1D array
                action_to_store = action_continuous.squeeze().numpy()
                if action_to_store.ndim == 0:  # If scalar, make it 1D
                    action_to_store = np.array([action_to_store])
                
                self.replay_buffer.push(
                    state, 
                    action_to_store,  # Store as 1D array
                    reward, 
                    next_state, 
                    terminated or truncated
                )
                
                state = next_state
                
                # Training step
                if len(self.replay_buffer) >= self.agent.cfg.batch_size:
                    batch = self.sample_batch()
                    if batch is not None:  # Only train if we got a valid batch
                        logs = self.agent.update(batch)
                        
                        if 'avg_q_values' not in metrics:
                            metrics['avg_q_values'] = []
                        metrics['avg_q_values'].append(logs['q1'])
                
                if terminated or truncated:
                    break
            
            metrics['episode_rewards'].append(episode_reward)
            metrics['episode_clicks'].append(episode_clicks)
            
            if episode % 100 == 0:
                print(f"Episode {episode}: Reward={episode_reward:.3f}, Clicks={episode_clicks}")
        
        return metrics
    
    def sample_batch(self) -> Batch:
        """Sample batch from replay buffer and convert to tensors"""
        batch_data = self.replay_buffer.sample(self.agent.cfg.batch_size)
        if batch_data is None:
            return None
            
        states, actions, rewards, next_states, dones = batch_data
        
        actions = actions.reshape(-1, 1)
        
        return Batch(
            obs=torch.FloatTensor(states).to(self.device),
            act=torch.FloatTensor(actions).to(self.device),
            rew=torch.FloatTensor(rewards).to(self.device),
            next_obs=torch.FloatTensor(next_states).to(self.device),
            done=torch.FloatTensor(dones).to(self.device)
        )