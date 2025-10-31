import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional
from data_loader import KuaiRandDataLoader

class KuaiRandEnv(gym.Env):
    """
    Gymnasium-compatible RL environment for video recommendation

    State: 128-dimensional vector (user + content + temporal + context + history)
    Action: Discrete (video index to recommend)
    Reward: Composite signal (click + watch_ratio + like)
    Episode: 10 steps
    """

    def __init__(self, data_loader: KuaiRandDataLoader, max_episode_length: int = 10):
        super().__init__()

        self.data_loader = data_loader
        self.max_episode_length = max_episode_length

        self.action_space = gym.spaces.Discrete(data_loader.get_n_videos())
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(128,), dtype=np.float32
        )

        self.current_user_idx = None
        self.current_step = 0
        self.episode_history = []

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment and start new episode

        Returns:
            state: 128-dimensional state vector
            info: Additional information
        """
        super().reset(seed=seed)

        self.current_user_idx = self.data_loader.get_random_user()
        self.current_step = 0
        self.episode_history = []

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
        reward, interaction_data = self._simulate_user_interaction(self.current_user_idx, action)

        self.episode_history.append({
            'video_idx': action,
            'reward': reward,
            'interaction': interaction_data
        })

        self.current_step += 1

        terminated = False
        truncated = self.current_step >= self.max_episode_length

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
        - [32:64]: Content preferences
        - [64:96]: Temporal patterns
        - [96:112]: Context features
        - [112:128]: History embedding
        """
        state = np.zeros(128, dtype=np.float32)

        # 1. User embedding (0-31)
        user_emb = self._encode_user_embedding(self.current_user_idx)
        state[0:32] = user_emb

        # 2. Content preferences (32-63)
        content_emb = self._encode_content_preferences()
        state[32:64] = content_emb

        # 3. Temporal patterns (64-95)
        temporal_emb = self._encode_temporal_patterns()
        state[64:96] = temporal_emb

        # 4. Context features (96-111)
        context_emb = self._encode_context_features()
        state[96:112] = context_emb

        # 5. History embedding (112-127)
        history_emb = self._encode_history_embedding()
        state[112:128] = history_emb

        return state

    def _encode_user_embedding(self, user_idx: int) -> np.ndarray:
        """32-dim: User characteristics, preferences, and behavior patterns"""
        user_emb = np.zeros(32, dtype=np.float32)
        user_stats = self.data_loader.user_stats.get(user_idx, {})
        user_history = self.data_loader.get_user_history(user_idx)
        
        # Basic engagement metrics (0-7)
        user_emb[0] = user_stats.get('click_rate', 0.0)
        user_emb[1] = user_stats.get('like_rate', 0.0)
        user_emb[2] = user_stats.get('avg_watch_ratio', 0.0)
        user_emb[3] = min(1.0, user_stats.get('count', 0) / 200.0)
        
        # Engagement depth (8-15) - FIXED: Check if arrays exist and have data
        if user_history and user_history.get('watch_ratios') and len(user_history['watch_ratios']) > 0:
            recent_watches = user_history['watch_ratios'][-20:]
            user_emb[8] = np.mean(recent_watches)
            user_emb[9] = np.mean([1 for r in recent_watches if r > 0.8]) if recent_watches else 0.0
            user_emb[10] = max(recent_watches) if recent_watches else 0.0
            user_emb[11] = np.std(recent_watches) if len(recent_watches) > 1 else 0.0
        
        # Consistency patterns (16-23) - FIXED: Check if arrays exist and have data
        if user_history and user_history.get('clicks') and len(user_history['clicks']) > 10:
            recent_clicks = user_history['clicks'][-10:]
            user_emb[16] = np.mean(recent_clicks)
            user_emb[17] = np.std(recent_clicks) if len(recent_clicks) > 1 else 0.0
            user_emb[18] = 1 if recent_clicks[-1] == 1 else 0
        
        # Like behavior (24-31) - FIXED: Check if arrays exist and have data
        if user_history and user_history.get('likes') and len(user_history['likes']) > 0:
            recent_likes = user_history['likes'][-20:]
            user_emb[24] = np.mean(recent_likes) if recent_likes else 0.0
            user_emb[25] = sum(user_history['likes']) / len(user_history['likes'])
        
        return user_emb
    def _encode_content_preferences(self) -> np.ndarray:
        """32-dim: What types of content the user engages with"""
        content_emb = np.zeros(32, dtype=np.float32)
        user_history = self.data_loader.get_user_history(self.current_user_idx)
        
        # FIXED: Comprehensive check for empty history
        if not user_history or not user_history.get('video_sequence') or len(user_history['video_sequence']) == 0:
            return content_emb
        
        recent_videos = user_history['video_sequence'][-30:]
        recent_clicks = user_history['clicks'][-30:] if user_history.get('clicks') else []
        recent_watches = user_history['watch_ratios'][-30:] if user_history.get('watch_ratios') else []
        
        # Content diversity (0-7)
        unique_videos = len(set(recent_videos))
        content_emb[0] = unique_videos / len(recent_videos)
        content_emb[1] = len(set(recent_videos)) / 100.0
        
        # Video popularity patterns (8-15)
        if recent_clicks and len(recent_clicks) > 0:
            successful_videos = [v for v, c in zip(recent_videos, recent_clicks) if c == 1]
            if successful_videos:
                video_popularities = [self.data_loader.video_stats.get(v, {}).get('count', 0) 
                                    for v in successful_videos]
                if video_popularities and len(video_popularities) > 0:
                    content_emb[8] = min(1.0, np.mean(video_popularities) / 500.0)
                    content_emb[9] = min(1.0, max(video_popularities) / 500.0)
                    content_emb[10] = np.std(video_popularities) / 100.0 if len(video_popularities) > 1 else 0.0
        
        # Engagement quality (16-23)
        if recent_watches and len(recent_watches) > 0:
            content_emb[16] = np.mean(recent_watches)
            content_emb[17] = np.mean([1 for w in recent_watches if w > 0.8]) if recent_watches else 0.0
            content_emb[18] = np.std(recent_watches) if len(recent_watches) > 1 else 0.0
        
        # Click patterns (24-31)
        if recent_clicks and len(recent_clicks) > 0:
            content_emb[24] = np.mean(recent_clicks)
            content_emb[25] = sum(recent_clicks) / 30.0
        
        return content_emb

    def _encode_temporal_patterns(self) -> np.ndarray:
        """32-dim: Recent behavior trends and temporal dynamics"""
        temporal_emb = np.zeros(32, dtype=np.float32)
        
        if len(self.episode_history) > 0:
            recent_rewards = [h['reward'] for h in self.episode_history[-8:]]
            recent_clicks = [h['interaction']['is_click'] for h in self.episode_history[-8:]]
            recent_watches = [h['interaction']['watch_ratio'] for h in self.episode_history[-8:]]
            
            # Current session performance (0-7)
            temporal_emb[0] = np.mean(recent_rewards) if recent_rewards else 0.0    
            temporal_emb[1] = sum(recent_clicks) / len(recent_clicks) if recent_clicks else 0.0  
            temporal_emb[2] = np.mean(recent_watches) if recent_watches else 0.0    
            
            # Engagement trends (8-15)
            if len(recent_rewards) >= 3:
                temporal_emb[8] = np.mean(recent_rewards[-3:]) - np.mean(recent_rewards[:-3])  
                temporal_emb[9] = 1 if recent_rewards[-1] > recent_rewards[0] else -1        
            
            # Patterns and streaks (16-23)
            if len(recent_clicks) >= 4:
                temporal_emb[16] = sum(recent_clicks[-2:])                          
                temporal_emb[17] = 1 if all(recent_clicks[-2:]) else 0             
                temporal_emb[18] = 1 if not any(recent_clicks[-2:]) else 0          
            
            # Watch pattern analysis (24-31)
            if recent_watches:
                temporal_emb[24] = sum(1 for w in recent_watches if w > 0.7) / len(recent_watches)  
                temporal_emb[25] = max(recent_watches)                              
        
        # Session timing (28-31)
        temporal_emb[28] = self.current_step / self.max_episode_length  
        temporal_emb[29] = (self.max_episode_length - self.current_step) / self.max_episode_length  
        
        return temporal_emb

    def _encode_context_features(self) -> np.ndarray:
        """16-dim: Current session context and environment"""
        context_emb = np.zeros(16, dtype=np.float32)
        
        # Session progress and timing (0-7)
        context_emb[0] = self.current_step / self.max_episode_length                 
        context_emb[1] = (self.max_episode_length - self.current_step) / self.max_episode_length  
        context_emb[2] = 1.0 if self.current_step == 0 else 0.0                     
        context_emb[3] = 1.0 if self.current_step >= self.max_episode_length - 1 else 0.0  
        
        # Current session performance (8-15)
        if len(self.episode_history) > 0:
            episode_rewards = [h['reward'] for h in self.episode_history]
            episode_clicks = [h['interaction']['is_click'] for h in self.episode_history]
            
            context_emb[8] = np.mean(episode_rewards)                               
            context_emb[9] = sum(episode_clicks) / len(episode_clicks)              
            context_emb[10] = len(self.episode_history) / self.max_episode_length  
            context_emb[11] = sum(episode_clicks)                                  
        
        return context_emb

    def _encode_history_embedding(self) -> np.ndarray:
        """16-dim: Recent interaction history and patterns"""
        history_emb = np.zeros(16, dtype=np.float32)
        
        if len(self.episode_history) > 0:
            # Recent interactions (0-7)
            last_interaction = self.episode_history[-1]
            history_emb[0] = last_interaction['reward']                             
            history_emb[1] = last_interaction['interaction']['is_click']            
            history_emb[2] = last_interaction['interaction']['watch_ratio']         
            history_emb[3] = last_interaction['interaction']['is_like']             
            
            # Recent patterns (8-15)
            recent_rewards = [h['reward'] for h in self.episode_history[-3:]]       
            recent_clicks = [h['interaction']['is_click'] for h in self.episode_history[-3:]]
            
            if recent_rewards:
                history_emb[8] = np.mean(recent_rewards)                            
                history_emb[9] = sum(recent_clicks) / len(recent_clicks)            
                history_emb[10] = 1 if recent_clicks[-1] == 1 else 0                
        
        return history_emb

    def _simulate_user_interaction(self, user_idx: int, video_idx: int) -> Tuple[float, Dict]:
        """
        Simulate user interaction with video

        First tries to get actual logged interaction.
        If not available, estimates reward based on statistics.

        Returns:
            reward: Composite reward signal
            interaction_data: Interaction details
        """
        interaction = self.data_loader.get_interaction(user_idx, video_idx)

        if interaction is not None:
            is_click = interaction['is_click']
            watch_ratio = interaction['watch_ratio']
            is_like = interaction['is_like']
        else:
            video_stats = self.data_loader.video_stats.get(video_idx, {})
            user_stats = self.data_loader.user_stats.get(user_idx, {})

            video_click_rate = video_stats.get('click_rate', 0.1)
            user_click_rate = user_stats.get('click_rate', 0.1)
            estimated_click_prob = 0.6 * video_click_rate + 0.4 * user_click_rate

            is_click = 1 if np.random.random() < estimated_click_prob else 0

            if is_click:
                video_watch = video_stats.get('avg_watch_ratio', 0.3)
                user_watch = user_stats.get('avg_watch_ratio', 0.3)
                estimated_watch = 0.6 * video_watch + 0.4 * user_watch
                watch_ratio = min(1.0, max(0.0, estimated_watch + np.random.normal(0, 0.1)))

                video_like_rate = video_stats.get('like_rate', 0.01)
                user_like_rate = user_stats.get('like_rate', 0.01)
                estimated_like_prob = 0.6 * video_like_rate + 0.4 * user_like_rate
                is_like = 1 if np.random.random() < estimated_like_prob else 0
            else:
                watch_ratio = 0.0
                is_like = 0

        reward = self._calculate_reward(is_click, watch_ratio, is_like)

        interaction_data = {
            'is_click': is_click,
            'watch_ratio': watch_ratio,
            'is_like': is_like,
            'is_actual': interaction is not None
        }

        return reward, interaction_data

    def _calculate_reward(self, is_click: int, watch_ratio: float, is_like: int) -> float:
        # Conservative weights that sum to <= 1.0 even with bonuses
        click_reward = 0.25 * is_click
        watch_reward = 0.35 * watch_ratio  
        like_reward = 0.25 * is_like
        
        quality_bonus = 0.0
        if is_click:
            if watch_ratio > 0.8:
                quality_bonus += 0.1
            elif watch_ratio > 0.5:
                quality_bonus += 0.05
            
            if is_like:
                quality_bonus += 0.05
        
        total_reward = click_reward + watch_reward + like_reward + quality_bonus
        
        return min(max(total_reward, 0.0), 1.0)


if __name__ == "__main__":
    print("Loading data...")
    loader = KuaiRandDataLoader()

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
    assert state.shape == (128,), f"State shape should be (128,), got {state.shape}"
    print(f"Reset works, state shape: {state.shape}")

    # Test 2: Action space
    assert env.action_space.n == loader.get_n_videos()
    print(f"Action space: {env.action_space.n} videos")

    # Test 3: Step
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    assert next_state.shape == (128,), "Next state shape incorrect"
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