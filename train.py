import numpy as np

from data_loader import KuaiRandDataLoader
from environment import KuaiRandEnv
from src.iql.config import IQLConfig
from src.training.trainer import IQLTrainer

def main():
    print("=== KuaiRand IQL Training ===")
    
    print("Loading data...")
    loader = KuaiRandDataLoader()
    try:
        loader.load_processed_data()
    except FileNotFoundError:
        print("Processing data from CSV...")
        loader.load_data(random_only=True)
        loader.save_processed_data()
    
    print("Creating environment...")
    env = KuaiRandEnv(data_loader=loader, max_episode_length=10)
    
    agent_config = IQLConfig(
        obs_dim=64,
        act_dim=1, 
        hidden_sizes=(256, 256),
        discount=0.99,
        tau=0.7,
        temperature=3.0,
        lr_policy=3e-4,
        lr_q=3e-4,
        lr_v=3e-4,
        batch_size=256
    )
    
    print("Creating trainer...")
    trainer = IQLTrainer(env, agent_config)
    
    print("Starting training...")
    metrics = trainer.collect_rollouts(n_episodes=1000)
    
    print("\nTraining completed!")
    print(f"Final average reward: {np.mean(metrics['episode_rewards'][-100:]):.3f}")
    print(f"Final average clicks: {np.mean(metrics['episode_clicks'][-100:]):.1f}")

if __name__ == "__main__":
    main()