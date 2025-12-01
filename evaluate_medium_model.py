"""
Evaluate Medium Dataset Model

Evaluates model trained on 1M interaction dataset.
Expected correlation: 0.35-0.45
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr

from src.iql.config import IQLConfig
from agent import IQLAgent


def evaluate_medium_model(
    checkpoint_path: str = "./checkpoints_medium/final_model.pt",
    test_data_path: str = "./processed_data_medium/test_data.pkl",
    device: str = "cpu"
):
    """Evaluate medium dataset model."""

    print("="*70)
    print("Medium Dataset Model Evaluation")
    print("="*70)

    # Load test data
    print(f"\n[1/3] Loading test set...")
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)

    test_interactions = test_data['interaction_lookup']
    n_videos = len(test_data['video_to_idx'])

    print(f"Test set: {len(test_interactions):,} interactions")
    print(f"  Click rate:  {test_data['stats']['click_rate']:.2%}")
    print(f"  Watch ratio: {test_data['stats']['watch_ratio_mean']:.2%}")
    print(f"  Mean reward: {test_data['stats']['reward_mean']:.4f}")

    # Load checkpoint
    print(f"\n[2/3] Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get normalization stats
    reward_mean = checkpoint['reward_normalization']['mean']
    reward_std = checkpoint['reward_normalization']['std']
    print(f"\nReward normalization:")
    print(f"  Mean: {reward_mean:.4f}")
    print(f"  Std:  {reward_std:.4f}")

    # Load model
    config = checkpoint['config']
    agent = IQLAgent(config)
    agent.policy.load_state_dict(checkpoint['agent_state']['policy'])
    agent.q1.load_state_dict(checkpoint['agent_state']['q1'])
    agent.q2.load_state_dict(checkpoint['agent_state']['q2'])
    agent.value.load_state_dict(checkpoint['agent_state']['value'])

    agent.to(device)
    agent.eval()

    print(f"Model loaded (trained for {checkpoint['global_step']:,} steps)")

    # Evaluate
    print(f"\n[3/3] Evaluating on test interactions...")

    q_values = []
    gt_rewards = []
    gt_clicks = []
    gt_watch = []

    # Sample for speed
    test_samples = list(test_interactions.items())
    if len(test_samples) > 10000:
        np.random.shuffle(test_samples)
        test_samples = test_samples[:10000]

    for (user_idx, video_idx), interaction in tqdm(test_samples, desc="Evaluating"):
        # Create 128-dim state
        state = np.zeros(128, dtype=np.float32)
        state[user_idx % 32] = 1.0
        state[32 + (video_idx % 64)] = 0.5
        state[96 + (user_idx % 32)] = 0.25

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # Action
        action_continuous = (video_idx / (n_videos - 1)) * 2.0 - 1.0
        action_tensor = torch.FloatTensor([[action_continuous]]).to(device)

        with torch.no_grad():
            q_val = agent.q1(state_tensor, action_tensor).item()

        # Denormalize
        q_denorm = q_val * reward_std + reward_mean

        # Ground truth (50/50 formula)
        gt_reward = 0.5 * interaction['is_click'] + 0.5 * interaction['watch_ratio']

        q_values.append(q_denorm)
        gt_rewards.append(gt_reward)
        gt_clicks.append(interaction['is_click'])
        gt_watch.append(interaction['watch_ratio'])

    # Convert to arrays
    q_values = np.array(q_values)
    gt_rewards = np.array(gt_rewards)
    gt_clicks = np.array(gt_clicks)
    gt_watch = np.array(gt_watch)

    # Results
    print("\n" + "="*70)
    print("MEDIUM DATASET RESULTS")
    print("="*70)

    print(f"\nSample size: {len(q_values):,}")

    print(f"\nModel predictions:")
    print(f"  Q-values: {q_values.mean():.4f} ± {q_values.std():.4f}")

    print(f"\nGround truth:")
    print(f"  Rewards:     {gt_rewards.mean():.4f} ± {gt_rewards.std():.4f}")
    print(f"  Click rate:  {gt_clicks.mean():.2%}")
    print(f"  Watch ratio: {gt_watch.mean():.2%}")

    # Correlations
    q_reward_corr, _ = pearsonr(q_values, gt_rewards)
    q_click_corr, _ = pearsonr(q_values, gt_clicks)
    q_watch_corr, _ = pearsonr(q_values, gt_watch)

    print(f"\nCORRELATIONS:")
    print(f"  Q ↔ Reward: {q_reward_corr:.3f}")
    print(f"  Q ↔ Click:  {q_click_corr:.3f}")
    print(f"  Q ↔ Watch:  {q_watch_corr:.3f}")

    # Ranking
    sorted_indices = np.argsort(q_values)[::-1]
    top_20_pct = int(len(q_values) * 0.2)

    top_reward = gt_rewards[sorted_indices[:top_20_pct]].mean()
    bottom_reward = gt_rewards[sorted_indices[-top_20_pct:]].mean()
    improvement = ((top_reward / bottom_reward) - 1) * 100

    print(f"\nRANKING QUALITY:")
    print(f"  Top 20% avg reward:    {top_reward:.4f}")
    print(f"  Bottom 20% avg reward: {bottom_reward:.4f}")
    print(f"  Improvement: {improvement:.1f}%")

    # Interpretation
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)

    if q_reward_corr > 0.35:
        print(f"\nEXCELLENT: Correlation {q_reward_corr:.3f} exceeds target (>0.35)")
    elif q_reward_corr > 0.25:
        print(f"\nGOOD: Correlation {q_reward_corr:.3f} shows strong learning")
    elif q_reward_corr > 0.15:
        print(f"\nMODERATE: Correlation {q_reward_corr:.3f} shows partial learning")
    else:
        print(f"\nWEAK: Correlation {q_reward_corr:.3f} below expectations")

    print(f"\nComparison:")
    print(f"  Baseline (untrained): 0.016")
    print(f"  Quick test (43K):     0.318")
    print(f"  Medium (1M):          {q_reward_corr:.3f}")

    if q_reward_corr > 0.318:
        print(f"  Improvement over quick test: {(q_reward_corr/0.318 - 1)*100:.1f}%")

    print("\n" + "="*70)

    # Return results for LaTeX
    return {
        'correlations': {
            'q_reward': q_reward_corr,
            'q_click': q_click_corr,
            'q_watch': q_watch_corr
        },
        'ranking_improvement': improvement,
        'sample_size': len(q_values),
        'training_size': '1M interactions',
        'training_steps': checkpoint['global_step']
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = evaluate_medium_model(device=device)

    print("\n" + "="*70)
    print("FOR YOUR PAPER")
    print("="*70)
    print(f"\nDataset: 1,000,000 interactions")
    print(f"Training steps: {results['training_steps']:,}")
    print(f"Correlation: {results['correlations']['q_reward']:.3f}")
    print(f"Ranking improvement: {results['ranking_improvement']:.1f}%")
    print("="*70)
