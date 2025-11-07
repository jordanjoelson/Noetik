import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

from src.iql.networks import ValueNet, QNet, GaussianTanhPolicy
from src.iql.config import IQLConfig
from src.data_loader import KuaiRandDataLoader
from src.environment import KuaiRandEnv


class AttentionAnalyzer:
    """Analyzes and visualizes attention patterns in the networks."""
    
    def __init__(self, value_net: ValueNet, q_net: QNet, policy_net: GaussianTanhPolicy):
        self.value_net = value_net
        self.q_net = q_net
        self.policy_net = policy_net
        
        # Create output directory
        self.output_dir = Path("outputs/attention_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_attention_weights(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract attention weights from all networks for analysis."""
        
        # Register hooks to capture attention weights
        attention_weights = {}
        
        def hook_fn(module, input, output):
            # Capture attention weights from MultiHeadAttention
            if hasattr(module, 'num_heads') and hasattr(module, 'last_attn'):
                attention_weights[module.__class__.__name__] = module.last_attn.detach()
        
        # Register hooks
        hooks = []
        for net in [self.value_net, self.q_net, self.policy_net]:
            for name, module in net.named_modules():
                if 'attention' in name.lower():
                    hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass
        with torch.no_grad():
            _ = self.value_net(obs)
            _ = self.q_net(obs, torch.zeros((obs.shape[0], 1)))
            _ = self.policy_net(obs)
            
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return attention_weights
    
    def analyze_component_importance(self, obs: torch.Tensor) -> Dict[str, float]:
        """Analyze which components (user, history, context) receive most attention."""
        
        weights = self.extract_attention_weights(obs)
        
        # Average attention weights across heads and batches
        component_scores = {
            'user': 0.0,
            'history': 0.0,
            'context': 0.0
        }
        
        for attn_name, attn_weights in weights.items():
            # Average across batches and heads
            avg_weights = attn_weights.mean(dim=(0,1))  # (N, N)
            
            # Sum attention to each component
            component_scores['user'] += avg_weights[:, 0].mean().item()
            component_scores['history'] += avg_weights[:, 1].mean().item()
            component_scores['context'] += avg_weights[:, 2].mean().item()
            
        # Normalize scores
        total = sum(component_scores.values())
        for k in component_scores:
            component_scores[k] /= total
            
        return component_scores
    
    def visualize_attention_patterns(self, obs: torch.Tensor, save_path: str = None):
        """Create heatmap visualizations of attention patterns."""
        
        weights = self.extract_attention_weights(obs)
        
        fig, axes = plt.subplots(1, len(weights), figsize=(15, 5))
        if len(weights) == 1:
            axes = [axes]
            
        for ax, (name, attn) in zip(axes, weights.items()):
            # Average across batches and heads
            avg_attn = attn.mean(dim=(0,1)).cpu().numpy()
            
            # Create heatmap
            sns.heatmap(
                avg_attn,
                ax=ax,
                cmap='viridis',
                xticklabels=['User', 'History', 'Context', 'Action'],
                yticklabels=['User', 'History', 'Context', 'Action']
            )
            ax.set_title(f'{name} Attention')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def test_attention_behavior(self, env: KuaiRandEnv, n_samples: int = 100) -> Dict:
        """Test if attention behaves as expected in different scenarios."""
        
        results = {
            'user_focused': 0,
            'history_focused': 0,
            'context_focused': 0,
            'adaptive_focus': 0
        }
        
        for _ in range(n_samples):
            state, _ = env.reset()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get component importance
            importance = self.analyze_component_importance(state_tensor)
            
            # Check which component gets most attention
            max_component = max(importance.items(), key=lambda x: x[1])[0]
            
            # Update counters
            if max_component == 'user':
                results['user_focused'] += 1
            elif max_component == 'history':
                results['history_focused'] += 1
            elif max_component == 'context':
                results['context_focused'] += 1
                
            # Check if attention adapts to state
            prev_importance = importance
            
            # Take a step and check if attention changes
            action = env.action_space.sample()
            next_state, _, _, _, _ = env.step(action)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            new_importance = self.analyze_component_importance(next_state_tensor)
            
            # If attention distribution changed significantly
            if max(abs(prev_importance[k] - new_importance[k]) for k in prev_importance) > 0.1:
                results['adaptive_focus'] += 1
                
        # Convert to percentages
        for k in results:
            results[k] = (results[k] / n_samples) * 100
            
        return results


def main():
    """Run attention mechanism validation tests."""
    print("\n=== Attention Mechanism Validation ===\n")
    
    # Create environment and networks
    print("Initializing environment and networks...")
    loader = KuaiRandDataLoader()
    try:
        loader.load_processed_data()
    except FileNotFoundError:
        loader.load_data(random_only=True)
        loader.save_processed_data()
        
    env = KuaiRandEnv(data_loader=loader)
    
    config = IQLConfig(
        obs_dim=128,
        act_dim=1,
        hidden_sizes=(256, 256)
    )
    
    value_net = ValueNet(config.obs_dim, config.hidden_sizes)
    q_net = QNet(config.obs_dim, config.act_dim, config.hidden_sizes)
    policy_net = GaussianTanhPolicy(config.obs_dim, config.act_dim, config.hidden_sizes)
    
    # Create analyzer
    analyzer = AttentionAnalyzer(value_net, q_net, policy_net)
    
    # Test 1: Basic attention patterns
    print("\nTest 1: Analyzing basic attention patterns...")
    state, _ = env.reset()
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    
    importance = analyzer.analyze_component_importance(state_tensor)
    print("\nComponent importance:")
    for component, score in importance.items():
        print(f"  {component}: {score:.2%}")
        
    # Test 2: Visualization
    print("\nTest 2: Generating attention visualizations...")
    viz_path = analyzer.output_dir / "attention_patterns.png"
    analyzer.visualize_attention_patterns(state_tensor, str(viz_path))
    print(f"Visualization saved to {viz_path}")
    
    # Test 3: Behavioral analysis
    print("\nTest 3: Testing attention behavior...")
    behavior_results = analyzer.test_attention_behavior(env)
    
    print("\nBehavioral test results:")
    print(f"  States focusing on user info: {behavior_results['user_focused']:.1f}%")
    print(f"  States focusing on history: {behavior_results['history_focused']:.1f}%")
    print(f"  States focusing on context: {behavior_results['context_focused']:.1f}%")
    print(f"  States with adaptive focus: {behavior_results['adaptive_focus']:.1f}%")
    
    # Validation summary
    print("\n=== Validation Summary ===")
    
    # Check component balance
    component_balance = max(importance.values()) - min(importance.values())
    if component_balance < 0.3:
        print("✓ Good component balance: attention distributed across components")
    else:
        print("⚠ High component imbalance: attention might be too focused")
        
    # Check adaptivity
    if behavior_results['adaptive_focus'] > 50:
        print("✓ Good attention adaptivity: focus changes with state")
    else:
        print("⚠ Low adaptivity: attention patterns might be too static")
        
    # Overall assessment
    if (component_balance < 0.3 and behavior_results['adaptive_focus'] > 50):
        print("\nOverall: Attention mechanisms appear to be working correctly!")
    else:
        print("\nOverall: Attention mechanisms may need adjustment.")
        
    print("\nRecommendations:")
    if component_balance >= 0.3:
        print("- Consider adjusting attention head weights")
        print("- Check if any components are being overlooked")
    if behavior_results['adaptive_focus'] <= 50:
        print("- Increase temperature in attention softmax")
        print("- Review attention block connectivity")


if __name__ == "__main__":
    main()