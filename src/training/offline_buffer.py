"""
Offline Replay Buffer for IQL Training

Loads entire dataset at initialization and samples batches randomly.
No new data is added during training (pure offline RL).
"""

import torch
import numpy as np
from typing import Optional
from src.iql.batch import Batch


class OfflineReplayBuffer:
    """
    Replay buffer for offline RL that holds a fixed dataset.

    Unlike online buffers, this never adds new transitions - it loads
    the complete offline dataset at initialization and samples from it.
    """

    def __init__(self, offline_data: Batch, device: str = "cpu"):
        """
        Initialize with complete offline dataset.

        Args:
            offline_data: Batch containing all transitions (obs, act, rew, next_obs, done)
            device: Device to store tensors on ("cpu" or "cuda")
        """
        self.device = device

        # Store data on specified device
        self.obs = offline_data.obs.to(device)
        self.act = offline_data.act.to(device)
        self.rew = offline_data.rew.to(device)
        self.next_obs = offline_data.next_obs.to(device)
        self.done = offline_data.done.to(device)

        self.size = len(self.obs)

        print(f"OfflineReplayBuffer initialized with {self.size:,} transitions")
        print(f"  Observation shape: {self.obs.shape}")
        print(f"  Action shape: {self.act.shape}")
        print(f"  Device: {device}")

        # Compute dataset statistics for monitoring
        self._compute_statistics()

    def _compute_statistics(self):
        """Compute and store dataset statistics."""
        self.stats = {
            'mean_reward': float(self.rew.mean()),
            'std_reward': float(self.rew.std()),
            'min_reward': float(self.rew.min()),
            'max_reward': float(self.rew.max()),
            'mean_done': float(self.done.mean()),
            'n_episodes': int(self.done.sum()),
        }

        print(f"\nDataset Statistics:")
        print(f"  Mean reward: {self.stats['mean_reward']:.4f} ± {self.stats['std_reward']:.4f}")
        print(f"  Reward range: [{self.stats['min_reward']:.4f}, {self.stats['max_reward']:.4f}]")
        print(f"  Episodes: {self.stats['n_episodes']}")
        print(f"  Avg episode length: {self.size / max(1, self.stats['n_episodes']):.1f}")

    def sample(self, batch_size: int) -> Batch:
        """
        Sample random batch from offline dataset.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Batch containing sampled transitions
        """
        if batch_size > self.size:
            raise ValueError(f"Batch size {batch_size} exceeds buffer size {self.size}")

        # Random sampling with replacement
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        return Batch(
            obs=self.obs[indices],
            act=self.act[indices],
            rew=self.rew[indices],
            next_obs=self.next_obs[indices],
            done=self.done[indices]
        )

    def __len__(self) -> int:
        """Return size of offline dataset."""
        return self.size

    def get_full_dataset(self) -> Batch:
        """
        Get the entire dataset as a single batch.

        Useful for computing statistics or validation metrics.
        """
        return Batch(
            obs=self.obs,
            act=self.act,
            rew=self.rew,
            next_obs=self.next_obs,
            done=self.done
        )

    def get_statistics(self) -> dict:
        """Return dataset statistics."""
        return self.stats.copy()


def save_offline_buffer(buffer: OfflineReplayBuffer, path: str):
    """
    Save offline buffer to disk.

    Args:
        buffer: OfflineReplayBuffer to save
        path: Path to save to (e.g., "offline_data.pt")
    """
    data = {
        'obs': buffer.obs.cpu(),
        'act': buffer.act.cpu(),
        'rew': buffer.rew.cpu(),
        'next_obs': buffer.next_obs.cpu(),
        'done': buffer.done.cpu(),
        'stats': buffer.stats,
        'size': buffer.size
    }
    torch.save(data, path)
    print(f"Saved offline buffer ({buffer.size:,} transitions) to {path}")


def load_offline_buffer(path: str, device: str = "cpu") -> OfflineReplayBuffer:
    """
    Load offline buffer from disk.

    Args:
        path: Path to load from
        device: Device to load tensors onto

    Returns:
        OfflineReplayBuffer instance
    """
    data = torch.load(path, map_location=device, weights_only=False)

    batch = Batch(
        obs=data['obs'],
        act=data['act'],
        rew=data['rew'],
        next_obs=data['next_obs'],
        done=data['done']
    )

    buffer = OfflineReplayBuffer(batch, device=device)
    print(f"Loaded offline buffer ({buffer.size:,} transitions) from {path}")

    return buffer
