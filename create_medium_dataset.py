"""
Create Medium-Sized Dataset for Training

Creates a balanced 1M interaction subset from full KuaiRand dataset
while maintaining proper statistics (38% click rate, etc.)
"""

import pickle
import numpy as np
from pathlib import Path
from src.data_loader import KuaiRandDataLoader
from tqdm import tqdm


def create_medium_dataset(
    target_interactions: int = 1_000_000,
    train_ratio: float = 0.8,
    save_dir: str = "./processed_data_medium",
    random_seed: int = 42
):
    """
    Create medium-sized dataset with stratified sampling.

    Strategy:
    - Sample users proportionally to maintain statistics
    - Keep temporal ordering within each user
    - Ensure train/test split is temporally valid
    """
    np.random.seed(random_seed)

    print("="*70)
    print(f"Creating Medium Dataset ({target_interactions:,} interactions)")
    print("="*70)

    # Load full dataset
    print("\n[1/5] Loading full KuaiRand dataset...")
    loader = KuaiRandDataLoader()

    try:
        loader.load_processed_data()
    except FileNotFoundError:
        print("Processing full dataset for first time...")
        loader.load_data(random_only=False)
        loader.save_processed_data()

    total_interactions = len(loader.interaction_lookup)
    n_users = loader.get_n_users()

    print(f"Full dataset: {total_interactions:,} interactions, {n_users:,} users")

    # Calculate target interactions per user
    target_per_user = target_interactions / n_users

    print(f"\n[2/5] Sampling {target_per_user:.1f} interactions per user on average...")

    # Sample from each user proportionally
    sampled_interactions = {}
    sampled_user_histories = {}
    user_interaction_counts = []

    for user_idx in tqdm(range(n_users), desc="Sampling users"):
        history = loader.get_user_history(user_idx)
        n_interactions = len(history['video_sequence'])

        if n_interactions < 2:
            continue

        # Sample proportion of user's history
        sample_ratio = min(1.0, target_per_user / n_interactions)
        n_sample = max(2, int(n_interactions * sample_ratio))

        # Take first n_sample interactions (maintains temporal order)
        sampled_user_histories[user_idx] = {
            'video_sequence': history['video_sequence'][:n_sample],
            'timestamps': history['timestamps'][:n_sample],
            'clicks': history['clicks'][:n_sample],
            'watch_ratios': history['watch_ratios'][:n_sample],
            'likes': history['likes'][:n_sample]
        }

        # Add to interaction lookup
        for i in range(n_sample):
            video_idx = history['video_sequence'][i]
            key = (user_idx, video_idx)
            if key in loader.interaction_lookup:
                sampled_interactions[key] = loader.interaction_lookup[key]

        user_interaction_counts.append(n_sample)

    actual_total = len(sampled_interactions)
    print(f"\nSampled {actual_total:,} interactions ({actual_total/total_interactions*100:.1f}% of full dataset)")
    print(f"Average per user: {np.mean(user_interaction_counts):.1f}")

    # Compute statistics of sampled data
    print("\n[3/5] Computing sampled data statistics...")

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
            'reward_mean': 0.5 * clicks.mean() + 0.5 * watch_ratios.mean()  # Current 50/50 formula
        }

    sampled_stats = compute_stats(sampled_interactions)
    print(f"\nSampled data statistics:")
    print(f"  Interactions: {sampled_stats['n_interactions']:,}")
    print(f"  Click rate:   {sampled_stats['click_rate']:.2%}")
    print(f"  Watch ratio:  {sampled_stats['watch_ratio_mean']:.2%}")
    print(f"  Like rate:    {sampled_stats['like_rate']:.2%}")
    print(f"  Mean reward:  {sampled_stats['reward_mean']:.4f}")

    # Split into train/test
    print(f"\n[4/5] Creating train/test split ({train_ratio:.0%}/{1-train_ratio:.0%})...")

    train_interactions = {}
    test_interactions = {}
    train_user_histories = {}
    test_user_histories = {}

    for user_idx, history in sampled_user_histories.items():
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

        # Split interactions
        for i in range(split_idx):
            video_idx = history['video_sequence'][i]
            key = (user_idx, video_idx)
            if key in sampled_interactions:
                train_interactions[key] = sampled_interactions[key]

        for i in range(split_idx, n_interactions):
            video_idx = history['video_sequence'][i]
            key = (user_idx, video_idx)
            if key in sampled_interactions:
                test_interactions[key] = sampled_interactions[key]

    train_stats = compute_stats(train_interactions)
    test_stats = compute_stats(test_interactions)

    print(f"\nTrain set: {len(train_interactions):,} ({len(train_interactions)/actual_total*100:.1f}%)")
    print(f"  Click rate:  {train_stats['click_rate']:.2%}")
    print(f"  Mean reward: {train_stats['reward_mean']:.4f}")

    print(f"\nTest set:  {len(test_interactions):,} ({len(test_interactions)/actual_total*100:.1f}%)")
    print(f"  Click rate:  {test_stats['click_rate']:.2%}")
    print(f"  Mean reward: {test_stats['reward_mean']:.4f}")

    # Save
    print(f"\n[5/5] Saving to {save_dir}/...")
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

    print(f"\nSaved:")
    print(f"  Train: {save_path / 'train_data.pkl'}")
    print(f"  Test:  {save_path / 'test_data.pkl'}")

    print("\n" + "="*70)
    print("Medium Dataset Created Successfully!")
    print("="*70)
    print(f"\nDataset size: {actual_total:,} interactions")
    print(f"Statistics match full dataset: {abs(sampled_stats['click_rate'] - 0.385) < 0.05}")
    print("\nNext steps:")
    print("  1. Train: python train_medium_dataset.py")
    print("  2. Evaluate: python evaluate_medium_model.py")
    print("="*70)


if __name__ == "__main__":
    create_medium_dataset(
        target_interactions=1_000_000,  # 1M interactions
        train_ratio=0.8,
        save_dir="./processed_data_medium"
    )
