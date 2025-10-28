"""
Offline IQL Training Script for KuaiRand

Trains IQL agent on logged KuaiRand interaction data without environment interaction (from the logged dataset).
"""

import torch
import numpy as np
from pathlib import Path

from src.data_loader import KuaiRandDataLoader
from src.environment import KuaiRandEnv
from src.iql.config import IQLConfig
from src.iql.utils import preprocess_kuairand
from src.training.offline_buffer import OfflineReplayBuffer, save_offline_buffer, load_offline_buffer
from src.training.offline_trainer import OfflineIQLTrainer
from agent import IQLAgent


def main():
    print("="*70)
    print("Offline IQL Training for KuaiRand Video Recommendations")
    print("="*70)

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Paths
    processed_buffer_path = "./processed_data/offline_buffer.pt"

    # Training hyperparameters
    n_training_steps = 100_000  # Number of gradient steps
    eval_frequency = 5_000      # Evaluate every N steps
    eval_episodes = 20          # Episodes per evaluation
    log_frequency = 500         # Log every N steps
    save_frequency = 10_000     # Save checkpoint every N steps

    # Step 1: Load or create offline dataset
    print("\n" + "="*70)
    print("Step 1: Loading Dataset")
    print("="*70)

    if Path(processed_buffer_path).exists():
        print(f"\nLoading preprocessed offline buffer from {processed_buffer_path}...")
        buffer = load_offline_buffer(processed_buffer_path, device=device)

    else:
        print("\nPreprocessed buffer not found. Creating from raw data...")

        # Load KuaiRand data
        print("\n[1/3] Loading KuaiRand dataset...")
        data_loader = KuaiRandDataLoader()

        try:
            data_loader.load_processed_data()
        except FileNotFoundError:
            print("Processing dataset for first time (this may take 30-60 seconds)...")
            data_loader.load_data(random_only=True)  # Use random_only=False for full dataset
            data_loader.save_processed_data()

        print(f"Dataset loaded: {data_loader.get_n_users():,} users, {data_loader.get_n_videos():,} videos")

        # Create environment (used only for state encoding)
        print("\n[2/3] Creating environment for state encoding...")
        env_temp = KuaiRandEnv(data_loader=data_loader)

        # Preprocess logged data into offline transitions
        print("\n[3/3] Preprocessing logged interactions into RL transitions...")
        print("(This converts user histories into state-action-reward-next_state tuples)")
        offline_data = preprocess_kuairand(
            data_loader=data_loader,
            env=env_temp,
            max_transitions_per_user=None  # Use all transitions (set to int to limit)
        )

        # Create offline buffer
        buffer = OfflineReplayBuffer(offline_data, device=device)

        # Save for future runs
        save_offline_buffer(buffer, processed_buffer_path)

    # Step 2: Create agent
    print("\n" + "="*70)
    print("Step 2: Creating IQL Agent")
    print("="*70)

    config = IQLConfig(
        obs_dim=64,              # State dimension from environment
        act_dim=1,               # 1D continuous action
        hidden_sizes=(256, 256),  # Network architecture
        lr_policy=3e-4,          # Policy learning rate
        lr_q=3e-4,               # Q-network learning rate
        lr_v=3e-4,               # Value network learning rate
        batch_size=256,          # Batch size for training
        discount=0.99,           # Discount factor
        tau=0.7,                 # IQL expectile parameter
        temperature=3.0          # Advantage weighting temperature
    )

    agent = IQLAgent(config)
    agent.policy.to(device)
    agent.value.to(device)
    agent.q1.to(device)
    agent.q2.to(device)

    print(f"\nIQL Agent created:")
    print(f"  Observation dim: {config.obs_dim}")
    print(f"  Action dim: {config.act_dim}")
    print(f"  Hidden layers: {config.hidden_sizes}")
    print(f"  Learning rates: π={config.lr_policy}, Q={config.lr_q}, V={config.lr_v}")
    print(f"  IQL tau: {config.tau}")
    print(f"  Temperature: {config.temperature}")

    # Step 3: Create environment for evaluation
    print("\n" + "="*70)
    print("Step 3: Setting up Evaluation Environment")
    print("="*70)

    data_loader = KuaiRandDataLoader()
    try:
        data_loader.load_processed_data()
    except FileNotFoundError:
        print("Loading data for evaluation...")
        data_loader.load_data(random_only=True)
        data_loader.save_processed_data()

    eval_env = KuaiRandEnv(data_loader=data_loader)
    print(f"Evaluation environment ready")
    print(f"  State space: {eval_env.observation_space.shape}")
    print(f"  Action space: {eval_env.action_space.n:,} videos")

    # Step 4: Create trainer
    print("\n" + "="*70)
    print("Step 4: Creating Offline Trainer")
    print("="*70)

    trainer = OfflineIQLTrainer(
        agent=agent,
        buffer=buffer,
        env=eval_env,
        config=config,
        device=device
    )

    # Step 5: Train
    print("\n" + "="*70)
    print("Step 5: Training")
    print("="*70)

    metrics = trainer.train(
        n_steps=n_training_steps,
        eval_frequency=eval_frequency,
        eval_episodes=eval_episodes,
        log_frequency=log_frequency,
        save_frequency=save_frequency,
        save_dir="./checkpoints"
    )

    # Step 6: Final summary
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)

    summary = trainer.get_training_summary()
    print(f"\nFinal Statistics:")
    print(f"  Total training steps: {summary['global_step']:,}")
    print(f"  Final Q-loss: {summary['recent_q_loss']:.4f}")
    print(f"  Final V-loss: {summary['recent_v_loss']:.4f}")
    print(f"  Final Policy-loss: {summary['recent_policy_loss']:.4f}")
    print(f"  Final Mean Q-value: {summary['recent_mean_q']:.3f}")

    if summary['best_eval_reward']:
        print(f"\nEvaluation Results:")
        print(f"  Best reward: {summary['best_eval_reward']:.4f}")
        print(f"  Latest reward: {summary['latest_eval_reward']:.4f}")

    print("\n" + "="*70)
    print("Done! Model saved to ./checkpoints/final_model.pt")
    print("="*70)


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    main()
