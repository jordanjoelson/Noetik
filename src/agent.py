import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Experience replay buffer
    Stores transitions (state, action, reward, next_state, done)
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)


class RandomAgent:
    """
    Random baseline agent
    Selects actions uniformly at random
    """

    def __init__(self, state_dim: int = 64, action_dim: int = 7388):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = ReplayBuffer(capacity=10000)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select random action"""
        return np.random.randint(0, self.action_dim)

    def train_step(self, batch_size: int = 64) -> float:
        """Training step"""
        return 0.0

    def update_target_network(self):
        """Update target network"""
        pass


if __name__ == "__main__":
    print("Testing Random Agent")

    agent = RandomAgent(state_dim=64, action_dim=7388)
    print(f"State dim: {agent.state_dim}")
    print(f"Action dim: {agent.action_dim}")

    # Test action selection
    state = np.random.randn(64)
    action = agent.select_action(state)
    assert 0 <= action < agent.action_dim
    print(f"Action: {action}")

    # Test replay buffer
    for i in range(100):
        state = np.random.randn(64)
        action = np.random.randint(0, agent.action_dim)
        reward = np.random.random()
        next_state = np.random.randn(64)
        done = False
        agent.replay_buffer.push(state, action, reward, next_state, done)

    assert len(agent.replay_buffer) == 100
    print(f"Replay buffer size: {len(agent.replay_buffer)}")
    print("Tests passed")
