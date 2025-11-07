import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional
from src.data_loader import KuaiRandDataLoader


class KuaiRandEnv(gym.Env):
    """
    Gymnasium-compatible RL environment for video recommendation

    State: 128-dimensional vector (user embedding + history + context)
    Action: Discrete (video index to recommend)
    Reward: Composite signal (click + watch_ratio + like)
    Episode: 10 steps
    """

    def __init__(self, data_loader: KuaiRandDataLoader, max_episode_length: int = 10):
        super().__init__()

        self.data_loader = data_loader
        self.max_episode_length = max_episode_length

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(data_loader.get_n_videos())
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(128,), dtype=np.float32
        )

        # Episode state
        self.current_user_idx = None
        self.current_step = 0
        self.episode_history = []

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment and start new episode

        Returns:
            state: 64-dimensional state vector
            info: Additional information
        """
        super().reset(seed=seed)

        # Sample random user
        self.current_user_idx = self.data_loader.get_random_user()
        self.current_step = 0
        self.episode_history = []

        # Get initial state
        state = self._get_state()

        info = {
            'user_idx': self.current_user_idx,
            'step': self.current_step
        }

        return state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take action (recommend video) and observe reward

        Args:
            action: Video index to recommend

        Returns:
            next_state: Next state vector
            reward: Reward signal
            terminated: Episode finished naturally
            truncated: Episode cut off by time limit
            info: Additional information
        """
        # Get reward from interaction (actual or estimated)
        reward, interaction_data = self._simulate_user_interaction(self.current_user_idx, action)

        # Update episode history
        self.episode_history.append({
            'video_idx': action,
            'reward': reward,
            'interaction': interaction_data
        })

        self.current_step += 1

        # Check if episode is done
        terminated = False
        truncated = self.current_step >= self.max_episode_length

        # Get next state
        next_state = self._get_state()

        info = {
            'user_idx': self.current_user_idx,
            'step': self.current_step,
            'video_idx': action,
            'interaction': interaction_data
        }

        return next_state, reward, terminated, truncated, info

    def _get_state(self) -> np.ndarray:
        """
        Encode current state as 128-dimensional vector

        State components:
        - [0:32]: User embedding
        - [32:96]: History embedding
        - [96:128]: Context features
        """
        state = np.zeros(128, dtype=np.float32)

        # User embedding
        user_emb = self._encode_user(self.current_user_idx)
        state[0:32] = user_emb

        # History embedding
        history_emb = self._encode_history(self.current_user_idx)
        state[32:96] = history_emb

        # Context features
        context_emb = self._encode_context()
        state[96:128] = context_emb

        return state

    def _encode_user(self, user_idx: int) -> np.ndarray:
        """
        Encode user as 32-dimensional vector

        Uses user statistics from data loader
        """
        user_emb = np.zeros(32, dtype=np.float32)

        # Simple hash encoding
        user_emb[user_idx % 32] = 1.0

        # User statistics
        user_stats = self.data_loader.user_stats.get(user_idx, {})
        user_emb[0] = user_stats.get('click_rate', 0.0)
        user_emb[1] = user_stats.get('like_rate', 0.0)
        user_emb[2] = user_stats.get('avg_watch_ratio', 0.0)
        user_emb[3] = min(1.0, user_stats.get('count', 0) / 1000.0)

        return user_emb

    def _encode_history(self, user_idx: int) -> np.ndarray:
        """
        Encode user history as 64-dimensional vector

        Includes both long-term history and episode history
        """
        history_emb = np.zeros(64, dtype=np.float32)

        # Get user's long-term history
        user_history = self.data_loader.get_user_history(user_idx)

        if len(user_history['video_sequence']) > 0:
            # Recent videos (last 10)
            recent_videos = user_history['video_sequence'][-10:]
            recent_clicks = user_history['clicks'][-10:]
            recent_watch_ratios = user_history['watch_ratios'][-10:]
            recent_likes = user_history['likes'][-10:]

            # Aggregate features
            history_emb[0] = len(recent_videos) / 50.0  # Normalized count
            history_emb[1] = np.mean(recent_clicks) if recent_clicks else 0.0
            history_emb[2] = np.mean(recent_watch_ratios) if recent_watch_ratios else 0.0
            history_emb[3] = np.std(recent_watch_ratios) if len(recent_watch_ratios) > 1 else 0.0
            history_emb[4] = np.mean(recent_likes) if recent_likes else 0.0

            # Video diversity
            if len(recent_videos) > 0:
                history_emb[5] = len(set(recent_videos)) / len(recent_videos)

        # Episode history features
        if len(self.episode_history) > 0:
            episode_rewards = [h['reward'] for h in self.episode_history]
            history_emb[6] = np.mean(episode_rewards)
            history_emb[7] = np.max(episode_rewards)
            history_emb[8] = len(self.episode_history) / self.max_episode_length

        return history_emb

    def _encode_context(self) -> np.ndarray:
        """
        Encode episode context as 32-dimensional vector
        """
        context_emb = np.zeros(32, dtype=np.float32)

        # Episode progress
        context_emb[0] = self.current_step / self.max_episode_length

        # Steps remaining
        context_emb[1] = (self.max_episode_length - self.current_step) / self.max_episode_length

        return context_emb

    def _simulate_user_interaction(self, user_idx: int, video_idx: int) -> Tuple[float, Dict]:
        """
        Simulate user interaction with video

        First tries to get actual logged interaction.
        If not available, estimates reward based on statistics.

        Returns:
            reward: Composite reward signal
            interaction_data: Interaction details
        """
        # Try to get actual interaction
        interaction = self.data_loader.get_interaction(user_idx, video_idx)

        if interaction is not None:
            # Use actual logged interaction
            is_click = interaction['is_click']
            watch_ratio = interaction['watch_ratio']
            is_like = interaction['is_like']
        else:
            # Estimate interaction based on statistics
            video_stats = self.data_loader.video_stats.get(video_idx, {})
            user_stats = self.data_loader.user_stats.get(user_idx, {})

            # Weighted average of video and user statistics
            video_click_rate = video_stats.get('click_rate', 0.1)
            user_click_rate = user_stats.get('click_rate', 0.1)
            estimated_click_prob = 0.6 * video_click_rate + 0.4 * user_click_rate

            # Sample click
            is_click = 1 if np.random.random() < estimated_click_prob else 0

            # Estimate watch ratio (can happen even without click)
            video_watch = video_stats.get('avg_watch_ratio', 0.3)
            user_watch = user_stats.get('avg_watch_ratio', 0.3)

            if is_click:
                # Higher watch ratio when clicked (from dataset: 44.4% avg)
                estimated_watch = 0.6 * video_watch + 0.4 * user_watch
                watch_ratio = min(1.0, max(0.0, estimated_watch + np.random.normal(0, 0.1)))

                video_like_rate = video_stats.get('like_rate', 0.01)
                user_like_rate = user_stats.get('like_rate', 0.01)
                estimated_like_prob = 0.6 * video_like_rate + 0.4 * user_like_rate
                is_like = 1 if np.random.random() < estimated_like_prob else 0
            else:
                # Lower watch ratio when not clicked (from dataset: 9% avg)
                # Scale down the watch ratio for non-clicks
                estimated_watch = 0.2 * (0.6 * video_watch + 0.4 * user_watch)
                watch_ratio = min(1.0, max(0.0, estimated_watch + np.random.normal(0, 0.05)))
                is_like = 0

        # Calculate composite reward
        reward = self._calculate_reward(is_click, watch_ratio, is_like)

        interaction_data = {
            'is_click': is_click,
            'watch_ratio': watch_ratio,
            'is_like': is_like,
            'is_actual': interaction is not None
        }

        return reward, interaction_data

    def _calculate_reward(self, is_click: int, watch_ratio: float, is_like: int) -> float:
        """Calculate reward: 0.5 * click + 0.5 * watch_ratio"""
        reward = 0.5 * is_click + 0.5 * watch_ratio
        return float(reward)


if __name__ == "__main__":
    # Test environment
    print("Loading data...")
    loader = KuaiRandDataLoader()

    # Try to load processed data, otherwise load from CSV
    try:
        loader.load_processed_data()
    except FileNotFoundError:
        print("Processed data not found, loading from CSV...")
        loader.load_data()
        loader.save_processed_data()

    print("\nCreating environment...")
    env = KuaiRandEnv(data_loader=loader)

    print("\n=== Environment Tests ===")

    # Test 1: Reset
    state, info = env.reset()
    assert state.shape == (64,), f"State shape should be (64,), got {state.shape}"
    print(f"Reset works, state shape: {state.shape}")

    # Test 2: Action space
    assert env.action_space.n == loader.get_n_videos()
    print(f"Action space: {env.action_space.n} videos")

    # Test 3: Step
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    assert next_state.shape == (64,), "Next state shape incorrect"
    assert 0 <= reward <= 1.0, f"Reward should be in [0, 1], got {reward}"
    print(f"Step works, reward: {reward:.3f}")

    # Test 4: Full episode
    print("\n=== Running Full Episode ===")
    state, info = env.reset()
    episode_reward = 0

    for step in range(10):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        print(f"Step {step+1}: action={action}, reward={reward:.3f}, "
              f"click={info['interaction']['is_click']}, "
              f"actual={info['interaction']['is_actual']}")

        state = next_state

        if terminated or truncated:
            break

    print(f"\nEpisode reward: {episode_reward:.3f}")
