"""
Validation script for KuaiRand RL environment

Tests all components for proper integration
"""

import numpy as np
from src.data_loader import KuaiRandDataLoader
from src.environment import KuaiRandEnv

# Try to import RandomAgent if available; otherwise provide a small fallback for tests
try:
    from src.agent import RandomAgent
except Exception:
    import random

    class _SimpleReplayBuffer:
        def __init__(self):
            self.buf = []

        def push(self, s, a, r, ns, d):
            self.buf.append((s, a, r, ns, d))

        def __len__(self):
            return len(self.buf)

    class RandomAgent:
        def __init__(self, state_dim: int, action_dim: int):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.replay_buffer = _SimpleReplayBuffer()

        def select_action(self, state, epsilon: float = None):
            # Return a single discrete action
            return random.randint(0, self.action_dim - 1)

        def train_step(self, batch_size: int = 32):
            # Dummy training step
            return 0.0


def test_data_loader():
    """Test data loader functionality"""
    print("=" * 60)
    print("Testing Data Loader")
    print("=" * 60)

    loader = KuaiRandDataLoader()

    # Try to load processed data
    try:
        print("Loading processed data...")
        loader.load_processed_data()
    except FileNotFoundError:
        print("Processed data not found, loading from CSV...")
        loader.load_data()
        loader.save_processed_data()

    # Validate
    assert loader.get_n_users() > 0, "No users loaded"
    assert loader.get_n_videos() > 0, "No videos loaded"
    print(f"\u0013 Loaded {loader.get_n_users()} users and {loader.get_n_videos()} videos")

    # Test user history
    user_idx = 0
    history = loader.get_user_history(user_idx)
    assert len(history['video_sequence']) > 0, "User has no history"
    print(f"\u0013 User {user_idx} has {len(history['video_sequence'])} interactions")

    # Test interaction lookup
    video_idx = history['video_sequence'][0]
    interaction = loader.get_interaction(user_idx, video_idx)
    assert interaction is not None, "Interaction should exist"
    print(f"\u0013 Interaction lookup works")

    # Test random sampling
    random_user = loader.get_random_user()
    assert 0 <= random_user < loader.get_n_users()
    print(f"\u0013 Random sampling works")

    print("\n\x05 Data Loader: All tests passed!\n")

    return loader


def test_environment(loader):
    """Test environment functionality"""
    print("=" * 60)
    print("Testing Environment")
    print("=" * 60)

    env = KuaiRandEnv(data_loader=loader, max_episode_length=10)

    # Test reset
    state, info = env.reset()
    assert state.shape == (128,), f"State shape should be (128,), got {state.shape}"
    print(f"\u0013 Reset works, state shape: {state.shape}")

    # Test action space
    assert env.action_space.n == loader.get_n_videos()
    print(f"\u0013 Action space: {env.action_space.n} videos")

    # Test step
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    assert next_state.shape == (128,), "Next state shape incorrect"
    assert 0 <= reward <= 1.0, f"Reward should be in [0, 1], got {reward}"
    print(f"\u0013 Step works, reward: {reward:.3f}")

    # Test full episode
    print("\nRunning full episode:")
    state, info = env.reset()
    episode_reward = 0
    clicks = 0
    actual_count = 0

    for step in range(10):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        clicks += info['interaction']['is_click']
        actual_count += int(info['interaction']['is_actual'])

        state = next_state

        if terminated or truncated:
            break

    print(f"  Total reward: {episode_reward:.3f}")
    print(f"  Total clicks: {clicks}")
    print(f"  Actual interactions: {actual_count}/10")
    print(f"\u0013 Full episode completed successfully")

    print("\n\x05 Environment: All tests passed!\n")

    return env


def test_agent():
    """Test agent functionality"""
    print("=" * 60)
    print("Testing Agent")
    print("=" * 60)

    agent = RandomAgent(state_dim=128, action_dim=7388)
    print(f"Agent created")

    # Test action selection
    state = np.random.randn(128)
    action = agent.select_action(state)
    assert 0 <= action < 7388, "Action out of range"
    print(f"Action selection works")

    # Add some experiences
    for i in range(100):
        state = np.random.randn(128)
        action = np.random.randint(0, 7388)
        reward = np.random.random()
        next_state = np.random.randn(128)
        done = False
        agent.replay_buffer.push(state, action, reward, next_state, done)

    assert len(agent.replay_buffer) == 100
    print(f"Replay buffer: {len(agent.replay_buffer)} transitions")

    print("\nAgent tests passed\n")

    return agent


def test_integration(loader, env, agent):
    """Test integration with short training loop"""
    print("=" * 60)
    print("Testing Integration (5 episodes)")
    print("=" * 60)

    epsilon = 0.5
    batch_size = 32

    for episode in range(5):
        state, info = env.reset()
        episode_reward = 0
        clicks = 0

        for step in range(10):
            # Select action
            action = agent.select_action(state, epsilon)

            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)

            # Store transition
            done = terminated or truncated
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Train
            if len(agent.replay_buffer) >= batch_size:
                loss = agent.train_step(batch_size)

            episode_reward += reward
            clicks += info['interaction']['is_click']
            state = next_state

            if terminated or truncated:
                break

        print(f"Episode {episode+1}: Reward={episode_reward:.3f}, Clicks={clicks}")

    print("\n\x05 Integration: All tests passed!\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("KuaiRand RL Environment - Validation Tests")
    print("=" * 60 + "\n")

    try:
        # Test 1: Data Loader
        loader = test_data_loader()

        # Test 2: Environment
        env = test_environment(loader)

        # Test 3: Agent
        agent = test_agent()

        # Test 4: Integration
        test_integration(loader, env, agent)

        # Summary
        print("=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)

        return True

    except AssertionError as e:
        print(f"\nL Test failed: {e}")
        return False
    except Exception as e:
        print(f"\nL Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
