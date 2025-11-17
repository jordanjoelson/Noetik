"""
Evaluate IQL Model on Test Set

Denormalizes rewards and computes correlation metrics.
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse

from src.iql.config import IQLConfig
from agent import IQLAgent
from interpret import GradientDescentInterpreter


def evaluate_improved_model(
    checkpoint_path: str,
    test_data_path: str = "./processed_data/test_data.pkl",
    device: str = "cpu"
):
    """Evaluate improved model with proper reward denormalization."""

    print("="*70)
    print("Improved Model Evaluation on Test Set")
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

    # Get normalization stats if available
    reward_mean = 0.0
    reward_std = 1.0
    if 'reward_normalization' in checkpoint:
        reward_mean = checkpoint['reward_normalization']['mean']
        reward_std = checkpoint['reward_normalization']['std']
        print(f"\nReward normalization detected:")
        print(f"  Mean: {reward_mean:.4f}")
        print(f"  Std:  {reward_std:.4f}")

    # Load model
    config = checkpoint['config']
    agent = IQLAgent(config)
    agent.policy.load_state_dict(checkpoint['agent_state']['policy'])
    agent.q1.load_state_dict(checkpoint['agent_state']['q1'])
    agent.q2.load_state_dict(checkpoint['agent_state']['q2'])
    agent.value.load_state_dict(checkpoint['agent_state']['value'])

    agent.policy.to(device)
    agent.q1.to(device)
    agent.q2.to(device)
    agent.value.to(device)
    agent.policy.eval()

    print(f"Model loaded (trained for {checkpoint['global_step']:,} steps)")

    # Evaluate
    print(f"\n[3/3] Evaluating on test interactions...")

    results = {
        'q_values': [],
        'q_values_denorm': [],
        'advantages': [],
        'gt_rewards': [],
        'gt_clicks': [],
        'gt_watch': [],
        'gt_likes': []
    }

    # Sample for speed
    test_samples = list(test_interactions.items())
    if len(test_samples) > 5000:
        np.random.shuffle(test_samples)
        test_samples = test_samples[:5000]

    for (user_idx, video_idx), interaction in tqdm(test_samples, desc="Evaluating"):
        # Create 128-dim state (matches environment output)
        state = np.zeros(128, dtype=np.float32)
        # Simple encoding for evaluation
        state[user_idx % 32] = 1.0  # User portion (0-31)
        state[32 + (video_idx % 64)] = 0.5  # History portion (32-95)
        state[96 + (user_idx % 32)] = 0.25  # Context portion (96-127)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # Action
        action_continuous = (video_idx / (n_videos - 1)) * 2.0 - 1.0
        action_tensor = torch.FloatTensor([[action_continuous]]).to(device)

        with torch.no_grad():
            q1_val = agent.q1(state_tensor, action_tensor)
            q2_val = agent.q2(state_tensor, action_tensor)
            q_val = torch.minimum(q1_val, q2_val)
            v_val = agent.value(state_tensor)
            advantage = q_val - v_val

        # Store normalized Q-values
        q_normalized = q_val.item()

        # Denormalize Q-values to original scale
        q_denormalized = q_normalized * reward_std + reward_mean

        # Ground truth (unnormalized) - use current 50/50 formula
        gt_reward = (
            0.5 * interaction['is_click'] +
            0.5 * interaction['watch_ratio']
        )

        results['q_values'].append(q_normalized)
        results['q_values_denorm'].append(q_denormalized)
        results['advantages'].append(advantage.item())
        results['gt_rewards'].append(gt_reward)
        results['gt_clicks'].append(interaction['is_click'])
        results['gt_watch'].append(interaction['watch_ratio'])
        results['gt_likes'].append(interaction['is_like'])

    # Convert to arrays
    q_values = np.array(results['q_values_denorm'])  # Use denormalized for comparison
    advantages = np.array(results['advantages'])
    gt_rewards = np.array(results['gt_rewards'])
    gt_clicks = np.array(results['gt_clicks'])
    gt_watch = np.array(results['gt_watch'])
    gt_likes = np.array(results['gt_likes'])

    # Analysis
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)

    print(f"\nSample size: {len(q_values):,}")

    print("\nMODEL PREDICTIONS (denormalized):")
    print(f"  Q-values: {q_values.mean():.4f} ± {q_values.std():.4f}")
    print(f"  Range:    [{q_values.min():.4f}, {q_values.max():.4f}]")

    print("\nGROUND TRUTH:")
    print(f"  Rewards:     {gt_rewards.mean():.4f} ± {gt_rewards.std():.4f}")
    print(f"  Click rate:  {gt_clicks.mean():.2%}")
    print(f"  Watch ratio: {gt_watch.mean():.2%}")
    print(f"  Like rate:   {gt_likes.mean():.2%}")

    # Correlations
    print("\nCORRELATION ANALYSIS:")
    q_reward_corr = np.corrcoef(q_values, gt_rewards)[0, 1]
    q_click_corr = np.corrcoef(q_values, gt_clicks)[0, 1]
    q_watch_corr = np.corrcoef(q_values, gt_watch)[0, 1]

    print(f"  Q-value <-> Reward: {q_reward_corr:.3f}")
    print(f"  Q-value <-> Click:  {q_click_corr:.3f}")
    print(f"  Q-value <-> Watch:  {q_watch_corr:.3f}")

    # Ranking
    print("\nRANKING QUALITY:")
    sorted_indices = np.argsort(q_values)[::-1]
    top_k = min(1000, len(sorted_indices) // 5)

    top_reward = gt_rewards[sorted_indices[:top_k]].mean()
    bottom_reward = gt_rewards[sorted_indices[-top_k:]].mean()

    print(f"  Top {top_k} Q-values:")
    print(f"    Reward:      {top_reward:.4f}")
    print(f"    Click rate:  {gt_clicks[sorted_indices[:top_k]].mean():.2%}")
    print(f"    Watch ratio: {gt_watch[sorted_indices[:top_k]].mean():.2%}")

    print(f"  Bottom {top_k} Q-values:")
    print(f"    Reward:      {bottom_reward:.4f}")
    print(f"    Click rate:  {gt_clicks[sorted_indices[-top_k:]].mean():.2%}")
    print(f"    Watch ratio: {gt_watch[sorted_indices[-top_k:]].mean():.2%}")

    improvement = ((top_reward / bottom_reward) - 1) * 100
    print(f"\n  Ranking improvement: {improvement:.1f}%")

    
    print("\nINTERPRETATION ANALYSIS")

    interpreter = GradientDescentInterpreter(agent)

    print(f"\nAnalyzing top {min(10, len(sorted_indices))} recommendations:")
    for i, idx in enumerate(sorted_indices[:10]):
        user_idx, video_idx = test_samples[idx][0]
        interaction = test_samples[idx][1]
        
        state = np.zeros(128, dtype=np.float32)
        state[user_idx % 16] = 1.0
        action_continuous = (video_idx / (n_videos - 1)) * 2.0 - 1.0
        
        importance = interpreter.compute_importance(state, action_continuous)
        explanation = interpreter.generate_gradient_explanation(state, action_continuous)
        confidence = interpreter.analyze_decision_confidence(state, action_continuous)
        
        print(f"\nTop #{i+1}: User {user_idx}, Video {video_idx}")
        print(f"  Q-value: {q_values[idx]:.4f}, GT Reward: {gt_rewards[idx]:.4f}")
        print(f"  Click: {interaction['is_click']}, Watch: {interaction['watch_ratio']:.2f}")
        print(f"  Explanation: {explanation}")
        print(f"  Confidence: {'HIGH' if confidence['is_high_confidence'] else 'MEDIUM' if not confidence['is_ambiguous'] else 'LOW'}")
        print(f"  Top features: {interpreter.get_top_influential_features(state, action_continuous, top_k=2)}")

    print("OVERALL FEATURE IMPORTANCE DISTRIBUTION")

    sample_interpretations = []
    for idx in sorted_indices[:100]:  #top 100 sampled
        user_idx, video_idx = test_samples[idx][0]
        state = np.zeros(64, dtype=np.float32)
        state[user_idx % 16] = 1.0
        action_continuous = (video_idx / (n_videos - 1)) * 2.0 - 1.0
        
        importance = interpreter.compute_importance(state, action_continuous)
        sample_interpretations.append(importance)

    avg_importance = {key: np.mean([imp[key] for imp in sample_interpretations]) 
                     for key in sample_interpretations[0].keys()}

    print("Average feature importance in top recommendations:")
    for feature, importance in sorted(avg_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {importance:.3f}")

    print("COMPARISON: TOP vs BOTTOM RECOMMENDATIONS")

    bottom_interpretations = []
    for idx in sorted_indices[-100:]:  #bottom 100 sampled
        user_idx, video_idx = test_samples[idx][0]
        state = np.zeros(64, dtype=np.float32)
        state[user_idx % 16] = 1.0
        action_continuous = (video_idx / (n_videos - 1)) * 2.0 - 1.0
        
        importance = interpreter.compute_importance(state, action_continuous)
        bottom_interpretations.append(importance)

    avg_importance_bottom = {key: np.mean([imp[key] for imp in bottom_interpretations]) 
                           for key in bottom_interpretations[0].keys()}

    print("Feature importance differences (Top - Bottom):")
    for feature in avg_importance.keys():
        diff = avg_importance[feature] - avg_importance_bottom[feature]
        print(f"  {feature:20}: {diff:+.3f}")

    print("\n" + "="*70)

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    status = []

    if q_reward_corr > 0.3:
        status.append("STRONG: Model learned reward patterns well!")
    elif q_reward_corr > 0.15:
        status.append("MODERATE: Model partially learned reward patterns")
    else:
        status.append("WEAK: Model did not learn reward patterns well")

    if q_click_corr > 0.2:
        status.append("Model learned click predictions")
    elif q_click_corr > 0.1:
        status.append("Model partially learned click predictions")
    else:
        status.append("Model did not learn click predictions")

    if q_watch_corr > 0.2:
        status.append("Model learned watch time predictions")
    elif q_watch_corr > 0.1:
        status.append("Model partially learned watch time predictions")
    else:
        status.append("Model did not learn watch time predictions")

    if improvement > 50:
        status.append("EXCELLENT ranking quality!")
    elif improvement > 25:
        status.append("Good ranking quality")
    else:
        status.append("Ranking quality needs improvement")

    for s in status:
        print(f"\n{s}")

    return {
        'correlations': {
            'q_reward': q_reward_corr,
            'q_click': q_click_corr,
            'q_watch': q_watch_corr
        },
        'ranking_improvement': improvement,
        'top_metrics': {
            'reward': top_reward,
            'click': gt_clicks[sorted_indices[:top_k]].mean(),
            'watch': gt_watch[sorted_indices[:top_k]].mean()
        },
        'interpretation': {
            'avg_feature_importance': avg_importance,
            'feature_importance_diff': {feature: avg_importance[feature] - avg_importance_bottom[feature] 
                                      for feature in avg_importance.keys()}
        }
    }

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./checkpoints_improved/final_model.pt')
    parser.add_argument('--test-data', default='./processed_data/test_data.pkl')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluate_improved_model(
        checkpoint_path=args.checkpoint,
        test_data_path=args.test_data,
        device=device
    )