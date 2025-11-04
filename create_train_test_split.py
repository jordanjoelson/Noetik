"""
Create Train/Test Split for Offline RL Evaluation

Splits KuaiRand dataset into 80% train / 20% test.
"""

import pickle
import numpy as np
from pathlib import Path
from src.data_loader import KuaiRandDataLoader


def create_train_test_split(
    train_ratio: float = 0.8,
    save_dir: str = "./processed_data",
    random_seed: int = 42
):
    """Split user interaction histories into train and test sets."""
    np.random.seed(random_seed)

    print("="*70)
    print("Creating Train/Test Split for Offline RL Evaluation")
    print("="*70)

    # Load data
    print("\n[1/4] Loading KuaiRand dataset...")
    loader = KuaiRandDataLoader()
    try:
        loader.load_processed_data()
    except FileNotFoundError:
        print("Processing FULL dataset for first time (11.7M interactions)...")
        print("This will take 5-10 minutes...")
        loader.load_data(random_only=False)  # Use all logs!
        loader.save_processed_data()

    print(f"Loaded: {loader.get_n_users():,} users, {loader.get_n_videos():,} videos")
    print(f"Total interactions: {len(loader.interaction_lookup):,}")

    # Split interactions
    print(f"\n[2/4] Splitting interactions (train={train_ratio:.0%}, test={1-train_ratio:.0%})...")

    train_interactions = {}
    test_interactions = {}
    train_user_histories = {}
    test_user_histories = {}
    train_count = 0
    test_count = 0

    for user_idx in range(loader.get_n_users()):
        history = loader.get_user_history(user_idx)
        n_interactions = len(history['video_sequence'])

        if n_interactions < 2:
            continue

        # Temporal split
        split_idx = int(n_interactions * train_ratio)
        split_idx = max(1, split_idx)
        split_idx = min(n_interactions - 1, split_idx)

        # Split history
        train_user_histories[user_idx] = {
            'video_sequence': history['video_sequence'][:split_idx],
            'timestamps': history['timestamps'][:split_idx],
            'clicks': history['clicks'][:split_idx],
            'watch_ratios': history['watch_ratios'][:split_idx],
            'likes': history['likes'][:split_idx]
        }

        test_user_histories[user_idx] = {
            'video_sequence': history['video_sequence'][split_idx:],
            'timestamps': history['timestamps'][split_idx:],
            'clicks': history['clicks'][split_idx:],
            'watch_ratios': history['watch_ratios'][split_idx:],
            'likes': history['likes'][split_idx:]
        }

        # Split interaction lookup
        for i in range(split_idx):
            video_idx = history['video_sequence'][i]
            key = (user_idx, video_idx)
            if key in loader.interaction_lookup:
                train_interactions[key] = loader.interaction_lookup[key]
                train_count += 1

        for i in range(split_idx, n_interactions):
            video_idx = history['video_sequence'][i]
            key = (user_idx, video_idx)
            if key in loader.interaction_lookup:
                test_interactions[key] = loader.interaction_lookup[key]
                test_count += 1

    print(f"\nSplit complete:")
    print(f"  Train: {train_count:,} ({train_count/(train_count+test_count)*100:.1f}%)")
    print(f"  Test:  {test_count:,} ({test_count/(train_count+test_count)*100:.1f}%)")

    # Compute statistics
    print("\n[3/4] Computing statistics...")

    def compute_stats(interactions_dict):
        clicks = []
        watch_ratios = []
        likes = []
        for interaction in interactions_dict.values():
            clicks.append(interaction['is_click'])
            watch_ratios.append(interaction['watch_ratio'])
            likes.append(interaction['is_like'])
        clicks = np.array(clicks)
        watch_ratios = np.array(watch_ratios)
        likes = np.array(likes)
        return {
            'n_interactions': len(clicks),
            'click_rate': clicks.mean(),
            'watch_ratio_mean': watch_ratios.mean(),
            'watch_ratio_std': watch_ratios.std(),
            'like_rate': likes.mean(),
            'reward_mean': 0.3 * clicks.mean() + 0.4 * watch_ratios.mean() + 0.3 * likes.mean()
        }

    train_stats = compute_stats(train_interactions)
    test_stats = compute_stats(test_interactions)

    print("\nTrain Set:", train_stats)
    print("Test Set:", test_stats)

    # Save splits
    print(f"\n[4/4] Saving to {save_dir}/...")
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    train_data = {
        'user_to_idx': loader.user_to_idx,
        'video_to_idx': loader.video_to_idx,
        'idx_to_user': loader.idx_to_user,
        'idx_to_video': loader.idx_to_video,
        'interaction_lookup': train_interactions,
        'user_histories': train_user_histories,
        'video_stats': loader.video_stats,
        'user_stats': loader.user_stats,
        'split': 'train',
        'stats': train_stats
    }

    test_data = {
        'user_to_idx': loader.user_to_idx,
        'video_to_idx': loader.video_to_idx,
        'idx_to_user': loader.idx_to_user,
        'idx_to_video': loader.idx_to_video,
        'interaction_lookup': test_interactions,
        'user_histories': test_user_histories,
        'video_stats': loader.video_stats,
        'user_stats': loader.user_stats,
        'split': 'test',
        'stats': test_stats
    }

    with open(save_path / 'train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open(save_path / 'test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)

    print(f"\nSaved train set: {save_path / 'train_data.pkl'}")
    print(f"Saved test set:  {save_path / 'test_data.pkl'}")
    print("\n" + "="*70)


if __name__ == "__main__":
    create_train_test_split()
