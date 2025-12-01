import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class KuaiRandDataLoader:
    """
    Load and process KuaiRand dataset
    """

    def __init__(self, data_path: str = "./KuaiRand-1K/data"):
        self.data_path = Path(data_path)

        # Data containers
        self.interactions = None
        self.users = None
        self.videos_basic = None

        # Quick lookup structures
        self.user_to_idx = {}
        self.video_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_video = {}
        self.interaction_lookup = {}
        self.user_histories = {}

        # Statistics for reward estimation
        self.video_stats = {}
        self.user_stats = {}
        
    def load_data(self, sample_users: Optional[int] = None, random_only: bool = False):
        """
        Load KuaiRand dataset

        Args:
            sample_users: If provided, only load this many users
            random_only: If True, only load random logs (~43K interactions, ~7K videos)
                        If False, load all logs (~11.7M interactions, ~4.3M videos)
        """
        # Load interaction logs
        if random_only:
            print("Loading random subset only (smaller dataset)...")
            random_log = pd.read_csv(
                self.data_path / 'log_random_4_22_to_5_08_1k.csv',
                na_values=['null', 'NULL', 'None']
            )
            self.interactions = random_log
        else:
            random_log = pd.read_csv(
                self.data_path / 'log_random_4_22_to_5_08_1k.csv',
                na_values=['null', 'NULL', 'None']
            )
            standard_log1 = pd.read_csv(
                self.data_path / 'log_standard_4_08_to_4_21_1k.csv',
                na_values=['null', 'NULL', 'None']
            )
            standard_log2 = pd.read_csv(
                self.data_path / 'log_standard_4_22_to_5_08_1k.csv',
                na_values=['null', 'NULL', 'None']
            )

            # Combine interactions
            self.interactions = pd.concat(
                [random_log, standard_log1, standard_log2],
                ignore_index=True
            )
        
        # Sort by user and time
        self.interactions = self.interactions.sort_values(['user_id', 'time_ms'])
        
        # Sample users if requested
        if sample_users:
            user_sample = self.interactions['user_id'].unique()[:sample_users]
            self.interactions = self.interactions[
                self.interactions['user_id'].isin(user_sample)
            ]
        
        # Load user and video features
        self.users = pd.read_csv(
            self.data_path / 'user_features_1k.csv',
            na_values=['null', 'NULL', 'None']
        )
        self.videos_basic = pd.read_csv(
            self.data_path / 'video_features_basic_1k.csv',
            na_values=['null', 'NULL', 'None']
        )
        
        # Handle missing values
        self.interactions['is_click'] = self.interactions['is_click'].fillna(0)
        self.interactions['is_like'] = self.interactions['is_like'].fillna(0)
        self.interactions['play_time_ms'] = self.interactions['play_time_ms'].fillna(0)
        self.interactions['duration_ms'] = self.interactions['duration_ms'].fillna(1)  # Avoid division by zero

        # Create mappings
        unique_users = sorted(self.interactions['user_id'].unique())
        unique_videos = sorted(self.interactions['video_id'].unique())

        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.video_to_idx = {vid: idx for idx, vid in enumerate(unique_videos)}
        self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
        self.idx_to_video = {idx: vid for vid, idx in self.video_to_idx.items()}

        # Create interaction lookup dictionary for fast access
        print("Building interaction lookup...")
        self._create_interaction_lookup()

        # Build user histories
        print("Building user histories...")
        self._build_user_histories()

        # Calculate statistics for reward estimation
        print("Calculating statistics...")
        self._calculate_statistics()

        print(f"\nDone! Loaded: {len(self.interactions):,} interactions")
        print(f"Users: {len(unique_users):,}")
        print(f"Videos: {len(unique_videos):,}")
        
    def _create_interaction_lookup(self):
        """Create dictionary for fast interaction lookup using indices"""
        # Map IDs to indices
        self.interactions['user_idx'] = self.interactions['user_id'].map(self.user_to_idx)
        self.interactions['video_idx'] = self.interactions['video_id'].map(self.video_to_idx)

        # Calculate watch ratios
        self.interactions['watch_ratio'] = (
            self.interactions['play_time_ms'] /
            self.interactions['duration_ms'].clip(lower=1)
        ).clip(0, 1)

        # Convert to records
        records = self.interactions[['user_idx', 'video_idx', 'is_click', 'is_like',
                                     'watch_ratio', 'play_time_ms', 'duration_ms', 'time_ms']].to_dict('records')

        # Build lookup
        for rec in records:
            key = (rec['user_idx'], rec['video_idx'])
            self.interaction_lookup[key] = {
                'is_click': int(rec['is_click']),
                'is_like': int(rec['is_like']),
                'watch_ratio': float(rec['watch_ratio']),
                'play_time_ms': rec['play_time_ms'],
                'duration_ms': rec['duration_ms'],
                'timestamp': rec['time_ms']
            }

    def _build_user_histories(self):
        """Build sorted histories for each user using groupby"""
        # Group by user and build histories
        for user_idx, group in self.interactions.groupby('user_idx'):
            group = group.sort_values('time_ms')
            self.user_histories[user_idx] = {
                'video_sequence': group['video_idx'].tolist(),
                'timestamps': group['time_ms'].tolist(),
                'clicks': group['is_click'].astype(int).tolist(),
                'watch_ratios': group['watch_ratio'].tolist(),
                'likes': group['is_like'].astype(int).tolist()
            }

    def _calculate_statistics(self):
        """Calculate video and user statistics using groupby"""
        # Video statistics using groupby
        video_groups = self.interactions.groupby('video_idx').agg({
            'is_click': 'mean',
            'is_like': 'mean',
            'watch_ratio': 'mean',
            'video_id': 'count'
        }).rename(columns={'video_id': 'count'})

        for video_idx in range(len(self.video_to_idx)):
            if video_idx in video_groups.index:
                row = video_groups.loc[video_idx]
                self.video_stats[video_idx] = {
                    'click_rate': float(row['is_click']),
                    'like_rate': float(row['is_like']),
                    'avg_watch_ratio': float(row['watch_ratio']),
                    'count': int(row['count'])
                }
            else:
                self.video_stats[video_idx] = {
                    'click_rate': 0.0,
                    'like_rate': 0.0,
                    'avg_watch_ratio': 0.0,
                    'count': 0
                }

        # User statistics using groupby
        user_groups = self.interactions.groupby('user_idx').agg({
            'is_click': 'mean',
            'is_like': 'mean',
            'watch_ratio': 'mean',
            'user_id': 'count'
        }).rename(columns={'user_id': 'count'})

        for user_idx in range(len(self.user_to_idx)):
            if user_idx in user_groups.index:
                row = user_groups.loc[user_idx]
                self.user_stats[user_idx] = {
                    'click_rate': float(row['is_click']),
                    'like_rate': float(row['is_like']),
                    'avg_watch_ratio': float(row['watch_ratio']),
                    'count': int(row['count'])
                }
            else:
                self.user_stats[user_idx] = {
                    'click_rate': 0.0,
                    'like_rate': 0.0,
                    'avg_watch_ratio': 0.0,
                    'count': 0
                }
    
    def get_user_history(self, user_idx: int) -> Dict:
        """
        Get user's complete viewing history

        Args:
            user_idx: User index (0-indexed)

        Returns:
            Dictionary with video_sequence, timestamps, clicks, watch_ratios, likes
        """
        return self.user_histories.get(user_idx, {
            'video_sequence': [],
            'timestamps': [],
            'clicks': [],
            'watch_ratios': [],
            'likes': []
        })

    def get_interaction(self, user_idx: int, video_idx: int) -> Optional[Dict]:
        """
        Get specific user-video interaction if it exists

        Args:
            user_idx: User index (0-indexed)
            video_idx: Video index (0-indexed)

        Returns:
            Dictionary with interaction data, or None if not found
        """
        return self.interaction_lookup.get((user_idx, video_idx))

    def get_n_users(self) -> int:
        """Return number of users"""
        return len(self.user_to_idx)

    def get_n_videos(self) -> int:
        """Return number of videos"""
        return len(self.video_to_idx)

    def get_random_user(self) -> int:
        """Sample random user index"""
        return np.random.randint(0, len(self.user_to_idx))

    def get_random_videos(self, n: int) -> List[int]:
        """Sample n random video indices"""
        return np.random.choice(len(self.video_to_idx), size=n, replace=False).tolist()

    def save_processed_data(self, path: str = "./processed_data"):
        """Save processed data structures for fast loading"""
        save_path = Path(path)
        save_path.mkdir(exist_ok=True)

        data = {
            'user_to_idx': self.user_to_idx,
            'video_to_idx': self.video_to_idx,
            'idx_to_user': self.idx_to_user,
            'idx_to_video': self.idx_to_video,
            'interaction_lookup': self.interaction_lookup,
            'user_histories': self.user_histories,
            'video_stats': self.video_stats,
            'user_stats': self.user_stats
        }

        with open(save_path / 'processed_data.pkl', 'wb') as f:
            pickle.dump(data, f)

        print(f"Processed data saved to {save_path / 'processed_data.pkl'}")

    def load_processed_data(self, path: str = "./processed_data"):
        """Load processed data from cache"""
        load_path = Path(path) / 'processed_data.pkl'

        if not load_path.exists():
            raise FileNotFoundError(f"Processed data not found at {load_path}")

        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        self.user_to_idx = data['user_to_idx']
        self.video_to_idx = data['video_to_idx']
        self.idx_to_user = data['idx_to_user']
        self.idx_to_video = data['idx_to_video']
        self.interaction_lookup = data['interaction_lookup']
        self.user_histories = data['user_histories']
        self.video_stats = data['video_stats']
        self.user_stats = data['user_stats']

        print(f"Processed data loaded from {load_path}")
        print(f"Users: {len(self.user_to_idx):,}")
        print(f"Videos: {len(self.video_to_idx):,}")
        print(f"Interactions: {len(self.interaction_lookup):,}")


if __name__ == "__main__":
    loader = KuaiRandDataLoader()

    print("Loading data...")
    loader.load_data()

    print("\n=== Validation Tests ===")

    # Test 1: Check data loaded
    assert loader.get_n_users() > 0, "No users loaded"
    assert loader.get_n_videos() > 0, "No videos loaded"
    print(f"Users: {loader.get_n_users()}")
    print(f"Videos: {loader.get_n_videos()}")

    # Test 2: Check user history
    user_idx = 0
    history = loader.get_user_history(user_idx)
    assert len(history['video_sequence']) > 0, "User has no history"
    print(f"User {user_idx} history length: {len(history['video_sequence'])}")

    # Test 3: Check interaction lookup
    video_idx = history['video_sequence'][0]
    interaction = loader.get_interaction(user_idx, video_idx)
    assert interaction is not None, "Interaction should exist"
    print(f"Interaction lookup works")

    # Test 4: Random sampling
    random_user = loader.get_random_user()
    assert 0 <= random_user < loader.get_n_users()
    print(f"Random user: {random_user}")

    random_videos = loader.get_random_videos(5)
    assert len(random_videos) == 5
    print(f"Random videos: {random_videos[:3]}...")

    # Test 5: Save and load
    print("\nSaving processed data...")
    loader.save_processed_data()

    print("\nLoading processed data...")
    loader2 = KuaiRandDataLoader()
    loader2.load_processed_data()
    assert loader2.get_n_users() == loader.get_n_users()
    print("Save/load works correctly")

    print("\n=== All tests passed! ===)")