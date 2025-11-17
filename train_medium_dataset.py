"""
Train IQL on Medium Dataset

Optimized for 1M interactions with balanced hyperparameters.
Expected training time: 4-6 hours
Expected correlation: 0.35-0.45
"""

import torch
import numpy as np
from pathlib import Path
import pickle

from src.data_loader import KuaiRandDataLoader
from src.environment import KuaiRandEnv
from src.iql.config import IQLConfig
from src.iql.utils import preprocess_kuairand
from src.iql.batch import Batch
from src.training.offline_buffer import OfflineReplayBuffer
from src.training.offline_trainer import OfflineIQLTrainer
from agent import IQLAgent


def load_medium_train_data():
    """Load medium-sized training data."""
    print("\n[1/5] Loading medium training data...")

    train_path = Path("./processed_data_medium/train_data.pkl")

    if not train_path.exists():
        print("ERROR: Medium dataset not found!")
        print("Please run: python create_medium_dataset.py")
        return None

    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)

    # Create data loader with medium data
    loader = KuaiRandDataLoader()
    loader.user_to_idx = train_data['user_to_idx']
    loader.video_to_idx = train_data['video_to_idx']
    loader.idx_to_user = train_data['idx_to_user']
    loader.idx_to_video = train_data['idx_to_video']
    loader.interaction_lookup = train_data['interaction_lookup']
    loader.user_histories = train_data['user_histories']
    loader.video_stats = train_data['video_stats']
    loader.user_stats = train_data['user_stats']

    print(f"Medium training data loaded:")
    print(f"  Users: {len(loader.user_to_idx):,}")
    print(f"  Videos: {len(loader.video_to_idx):,}")
    print(f"  Interactions: {len(loader.interaction_lookup):,}")
    print(f"  Statistics:")
    print(f"    Click rate:  {train_data['stats']['click_rate']:.2%}")
    print(f"    Watch ratio: {train_data['stats']['watch_ratio_mean']:.2%}")
    print(f"    Mean reward: {train_data['stats']['reward_mean']:.4f}")

    return loader, train_data['stats']


def main():
    print("="*70)
    print("Medium Dataset IQL Training")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Training configuration
    n_training_steps = 100_000  # Reduced from 200K (medium data size)
    eval_frequency = 5_000
    eval_episodes = 20
    log_frequency = 500
    save_frequency = 20_000

    # Load medium training data
    loader, train_stats = load_medium_train_data()
    if loader is None:
        return

    # Check for cached buffer
    buffer_path = Path("./processed_data_medium/offline_buffer.pt")

    if buffer_path.exists():
        print("\n[2/5] Loading cached preprocessed buffer...")
        data = torch.load(buffer_path, weights_only=False)

        batch = Batch(
            obs=data['obs'],
            act=data['act'],
            rew=data['rew'],
            next_obs=data['next_obs'],
            done=data['done']
        )
        buffer = OfflineReplayBuffer(batch, device=device)

        reward_mean = data['reward_mean']
        reward_std = data['reward_std']

        print(f"Loaded {len(buffer):,} transitions from cache")

    else:
        print("\n[2/5] Preprocessing medium dataset into RL transitions...")

        env = KuaiRandEnv(data_loader=loader)

        # Process with reasonable limits
        offline_data = preprocess_kuairand(
            loader, env,
            max_transitions_per_user=100  # More than quick test, less than full
        )

        print(f"\nCreated {len(offline_data.obs):,} transitions")

        # Normalize rewards
        reward_mean = float(offline_data.rew.mean())
        reward_std = float(offline_data.rew.std())

        print(f"\nReward statistics (before normalization):")
        print(f"  Mean: {reward_mean:.4f}")
        print(f"  Std:  {reward_std:.4f}")

        normalized_rewards = (offline_data.rew - reward_mean) / (reward_std + 1e-8)

        print(f"\nNormalized reward statistics:")
        print(f"  Mean: {normalized_rewards.mean():.4f}")
        print(f"  Std:  {normalized_rewards.std():.4f}")

        # Create buffer
        normalized_batch = Batch(
            obs=offline_data.obs,
            act=offline_data.act,
            rew=normalized_rewards,
            next_obs=offline_data.next_obs,
            done=offline_data.done
        )

        buffer = OfflineReplayBuffer(normalized_batch, device=device)

        # Cache for future use
        data = {
            'obs': buffer.obs.cpu(),
            'act': buffer.act.cpu(),
            'rew': buffer.rew.cpu(),
            'next_obs': buffer.next_obs.cpu(),
            'done': buffer.done.cpu(),
            'reward_mean': reward_mean,
            'reward_std': reward_std
        }
        torch.save(data, buffer_path)
        print(f"\nCached buffer to {buffer_path}")

    # Create agent
    print("\n[3/5] Creating IQL Agent...")

    config = IQLConfig(
        obs_dim=128,
        act_dim=1,
        hidden_sizes=(256, 256),  # Standard size for medium data
        lr_policy=1e-4,
        lr_q=3e-4,
        lr_v=3e-4,
        batch_size=256,
        discount=0.99,
        tau=0.7,           # Standard IQL tau
        temperature=10.0,  # Standard IQL temperature
        weight_decay=1e-5
    )

    agent = IQLAgent(config)
    agent.to(device)

    print(f"\nConfiguration:")
    print(f"  State dim: {config.obs_dim}")
    print(f"  Hidden layers: {config.hidden_sizes}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Tau: {config.tau}")
    print(f"  Temperature: {config.temperature}")

    # Evaluation environment
    print("\n[4/5] Setting up evaluation...")
    eval_loader = KuaiRandDataLoader()
    eval_loader.load_processed_data()  # Use full dataset for eval
    eval_env = KuaiRandEnv(data_loader=eval_loader)

    # Train
    print("\n[5/5] Training...")
    print(f"  Steps: {n_training_steps:,}")
    print(f"  Expected time: ~4-6 hours")
    print(f"  Target correlation: >0.35")

    trainer = OfflineIQLTrainer(
        agent=agent,
        buffer=buffer,
        env=eval_env,
        config=config,
        device=device
    )

    metrics = trainer.train(
        n_steps=n_training_steps,
        eval_frequency=eval_frequency,
        eval_episodes=eval_episodes,
        log_frequency=log_frequency,
        save_frequency=save_frequency,
        save_dir="./checkpoints_medium"
    )

    # Save with normalization stats
    final_checkpoint_path = Path("./checkpoints_medium/final_model.pt")
    checkpoint = torch.load(final_checkpoint_path, weights_only=False)
    checkpoint['reward_normalization'] = {
        'mean': reward_mean,
        'std': reward_std
    }
    torch.save(checkpoint, final_checkpoint_path)

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nModel saved to: {final_checkpoint_path}")
    print("\nNext step:")
    print("  python evaluate_medium_model.py")
    print("="*70)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
