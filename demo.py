"""
Demo of KuaiRand RL Environment

Demonstrates data loading, environment, and agent interaction
"""

from src.data_loader import KuaiRandDataLoader
from src.environment import KuaiRandEnv
from src.agent import RandomAgent

print("="*60)
print("KuaiRand RL Environment Demo")
print("="*60)

# Load data
print("\n[1/3] Loading dataset...")
loader = KuaiRandDataLoader()

try:
    loader.load_processed_data()
except FileNotFoundError:
    print("Processing dataset for first time (this may take 30-60 seconds)...")
    loader.load_data(random_only=True)
    loader.save_processed_data()

print(f"Dataset: {loader.get_n_users():,} users, {loader.get_n_videos():,} videos")

# Create environment
print("\n[2/3] Creating environment...")
env = KuaiRandEnv(data_loader=loader)
print(f"State space: {env.observation_space.shape}")
print(f"Action space: {env.action_space.n:,} videos")

# Create agent
print("\n[3/3] Creating agent...")
agent = RandomAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n
)
print(f"Agent ready")

# Run sample episode
print("\n" + "="*60)
print("Sample Episode")
print("="*60)

state, info = env.reset()
print(f"\nUser {info['user_idx']}")

episode_reward = 0
for step in range(10):
    action = agent.select_action(state)
    next_state, reward, terminated, truncated, info = env.step(action)

    episode_reward += reward

    print(f"  Step {step+1}: video={action}, reward={reward:.3f}, "
          f"click={info['interaction']['is_click']}, "
          f"watch={info['interaction']['watch_ratio']:.2f}")

    state = next_state
    if terminated or truncated:
        break

print(f"\nTotal reward: {episode_reward:.3f}")
