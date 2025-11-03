"""
Offline IQL Training with Reward Normalization

Trains IQL agent on full KuaiRand dataset with normalized rewards.
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
from src.training.offline_buffer import OfflineReplayBuffer, save_offline_buffer
from src.training.offline_trainer import OfflineIQLTrainer
from agent import IQLAgent


def load_train_data():
    """Load training data with proper split."""
    print("\n[1/5] Loading training data...")

    train_path = Path("./processed_data/train_data.pkl")

    if not train_path.exists():
        print("ERROR: Train data not found!")
        print("Please run: python create_train_test_split.py")
        return None, None

    # Load train split
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)

    # Create temporary data loader with train data only
    loader = KuaiRandDataLoader()
    loader.user_to_idx = train_data['user_to_idx']
    loader.video_to_idx = train_data['video_to_idx']
    loader.idx_to_user = train_data['idx_to_user']
    loader.idx_to_video = train_data['idx_to_video']
    loader.interaction_lookup = train_data['interaction_lookup']
    loader.user_histories = train_data['user_histories']
    loader.video_stats = train_data['video_stats']
    loader.user_stats = train_data['user_stats']

    print(f"Training data loaded:")
    print(f"  Users: {len(loader.user_to_idx):,}")
    print(f"  Videos: {len(loader.video_to_idx):,}")
    print(f"  Interactions: {len(loader.interaction_lookup):,}")
    print(f"  Train statistics:")
    print(f"    Click rate:  {train_data['stats']['click_rate']:.2%}")
    print(f"    Watch ratio: {train_data['stats']['watch_ratio_mean']:.2%}")
    print(f"    Mean reward: {train_data['stats']['reward_mean']:.4f}")

    return loader, train_data['stats']


def main():
    print("="*70)
    print("Improved Offline IQL Training - Interpretable Policy")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # IMPROVED CONFIGURATION
    n_training_steps = 200_000  # More training steps for better convergence
    eval_frequency = 10_000
    eval_episodes = 20
    log_frequency = 500
    save_frequency = 20_000

    # Load training data
    loader, train_stats = load_train_data()
    if loader is None:
        return

    # Create offline buffer with normalization
    print("\n[2/5] Creating offline buffer with reward normalization...")

    buffer_path = Path("./processed_data/offline_buffer_normalized.pt")

    if buffer_path.exists():
        print(f"Loading preprocessed buffer from {buffer_path}...")
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

        print(f"Reward normalization stats:")
        print(f"  Original mean: {reward_mean:.4f}")
        print(f"  Original std:  {reward_std:.4f}")

    else:
        print("Preprocessing training data...")

        # Create environment for state encoding
        env_temp = KuaiRandEnv(data_loader=loader)

        # Preprocess
        offline_data = preprocess_kuairand(
            data_loader=loader,
            env=env_temp,
            max_transitions_per_user=None
        )

        # Compute reward statistics BEFORE normalization
        reward_mean = float(offline_data.rew.mean())
        reward_std = float(offline_data.rew.std())

        print(f"\nReward statistics (before normalization):")
        print(f"  Mean: {reward_mean:.4f}")
        print(f"  Std:  {reward_std:.4f}")
        print(f"  Min:  {offline_data.rew.min():.4f}")
        print(f"  Max:  {offline_data.rew.max():.4f}")

        # NORMALIZE REWARDS: (r - mean) / std
        print("\nNormalizing rewards...")
        normalized_rewards = (offline_data.rew - reward_mean) / (reward_std + 1e-8)

        print(f"Normalized reward statistics:")
        print(f"  Mean: {normalized_rewards.mean():.4f}")
        print(f"  Std:  {normalized_rewards.std():.4f}")
        print(f"  Min:  {normalized_rewards.min():.4f}")
        print(f"  Max:  {normalized_rewards.max():.4f}")

        # Create normalized batch
        normalized_batch = Batch(
            obs=offline_data.obs,
            act=offline_data.act,
            rew=normalized_rewards,
            next_obs=offline_data.next_obs,
            done=offline_data.done
        )

        buffer = OfflineReplayBuffer(normalized_batch, device=device)

        # Save with normalization stats
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
        print(f"Saved normalized buffer to {buffer_path}")

    # Create agent with IMPROVED configuration
    print("\n[3/5] Creating IQL Agent with improved config...")

    config = IQLConfig(
        obs_dim=64,
        act_dim=1,
        hidden_sizes=(256, 256),
        lr_policy=1e-4,
        lr_q=3e-4,
        lr_v=3e-4,
        batch_size=256,
        discount=0.99,
        tau=0.7,
        temperature=10.0,
        weight_decay=1e-5
    )

    agent = IQLAgent(config)
    agent.policy.to(device)
    agent.value.to(device)
    agent.q1.to(device)
    agent.q2.to(device)

    print(f"\nIQL Agent configuration:")
    print(f"  State dim: {config.obs_dim}")
    print(f"  Action dim: {config.act_dim}")
    print(f"  Hidden layers: {config.hidden_sizes}")
    print(f"  Learning rates: policy={config.lr_policy}, Q={config.lr_q}, V={config.lr_v}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Tau: {config.tau}")

    # Create evaluation environment
    print("\n[4/5] Setting up evaluation...")

    # Load full data for evaluation
    eval_loader = KuaiRandDataLoader()
    try:
        eval_loader.load_processed_data()
    except FileNotFoundError:
        print("Loading full dataset (11.7M interactions)...")
        eval_loader.load_data(random_only=False)
        eval_loader.save_processed_data()

    eval_env = KuaiRandEnv(data_loader=eval_loader)
    print(f"Evaluation environment ready")

    # Create trainer
    trainer = OfflineIQLTrainer(
        agent=agent,
        buffer=buffer,
        env=eval_env,
        config=config,
        device=device
    )

    # Train
    print("\n[5/5] Training with improved settings...")
    print(f"  Training steps: {n_training_steps:,}")
    print(f"  Evaluation: every {eval_frequency:,} steps")
    print(f"  Checkpoints: every {save_frequency:,} steps")

    metrics = trainer.train(
        n_steps=n_training_steps,
        eval_frequency=eval_frequency,
        eval_episodes=eval_episodes,
        log_frequency=log_frequency,
        save_frequency=save_frequency,
        save_dir="./checkpoints_improved"
    )

    # Save normalization stats with final model
    final_checkpoint_path = Path("./checkpoints_improved/final_model.pt")
    checkpoint = torch.load(final_checkpoint_path, weights_only=False)
    checkpoint['reward_normalization'] = {
        'mean': reward_mean,
        'std': reward_std
    }
    torch.save(checkpoint, final_checkpoint_path)

    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)

    summary = trainer.get_training_summary()
    print(f"\nFinal Statistics:")
    print(f"  Total steps: {summary['global_step']:,}")
    print(f"  Final Q-loss: {summary['recent_q_loss']:.4f}")
    print(f"  Final V-loss: {summary['recent_v_loss']:.4f}")
    print(f"  Final Policy-loss: {summary['recent_policy_loss']:.4f}")
    print(f"  Final Mean Q-value: {summary['recent_mean_q']:.3f}")

    if summary['best_eval_reward']:
        print(f"\nEvaluation Results:")
        print(f"  Best reward: {summary['best_eval_reward']:.4f}")
        print(f"  Latest reward: {summary['latest_eval_reward']:.4f}")

    print("\nIMPORTANT: Q-values and rewards are normalized!")
    print(f"To convert back to original scale:")
    print(f"  original_reward = normalized_reward * {reward_std:.4f} + {reward_mean:.4f}")

    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("1. Run test set evaluation:")
    print("   python evaluate_improved_model.py")
    print("\n2. Compare with baseline (should see big improvement):")
    print("   - Q ↔ Reward correlation: target > 0.3")
    print("   - Q ↔ Click correlation: target > 0.2")
    print("   - Q ↔ Watch correlation: target > 0.2")
    print("="*70 + "\n")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    main()
