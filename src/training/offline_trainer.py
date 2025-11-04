"""
Offline IQL Trainer

Pure offline reinforcement learning trainer for IQL.
Trains on fixed dataset without environment interaction.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm

from agent import IQLAgent
from src.iql.config import IQLConfig
from src.training.offline_buffer import OfflineReplayBuffer
from src.environment import KuaiRandEnv


class OfflineIQLTrainer:
    """
    Trainer for offline IQL (Implicit Q-Learning).

    Key features:
    - Pure offline training (no environment interaction during training)
    - Batch sampling from fixed dataset
    - Periodic evaluation using environment
    - Model checkpointing
    - Training metrics logging
    """

    def __init__(
        self,
        agent: IQLAgent,
        buffer: OfflineReplayBuffer,
        env: Optional[KuaiRandEnv] = None,
        config: Optional[IQLConfig] = None,
        device: str = "cpu"
    ):
        """
        Initialize offline trainer.

        Args:
            agent: IQLAgent to train
            buffer: OfflineReplayBuffer with complete dataset
            env: Optional environment for evaluation
            config: IQL configuration
            device: Device for training
        """
        self.agent = agent
        self.buffer = buffer
        self.env = env
        self.config = config or IQLConfig()
        self.device = device

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Metrics history
        self.metrics_history = {
            'loss_q': [],
            'loss_v': [],
            'loss_policy': [],
            'q1': [],
            'q2': [],
            'v': [],
            'adv_mean': [],
            'weight_mean': [],
            'eval_reward': [],
            'eval_click_rate': []
        }

        print(f"\nOfflineIQLTrainer initialized")
        print(f"  Agent device: {device}")
        print(f"  Buffer size: {len(buffer):,} transitions")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Environment: {'Available' if env else 'Not provided'}")

    def train(
        self,
        n_steps: int,
        eval_frequency: int = 5000,
        eval_episodes: int = 10,
        log_frequency: int = 100,
        save_frequency: int = 10000,
        save_dir: str = "./checkpoints"
    ) -> Dict[str, List[float]]:
        """
        Train agent on offline dataset.

        Args:
            n_steps: Number of gradient steps to perform
            eval_frequency: Evaluate every N steps
            eval_episodes: Number of episodes for evaluation
            log_frequency: Log metrics every N steps
            save_frequency: Save checkpoint every N steps
            save_dir: Directory to save checkpoints

        Returns:
            Dictionary of training metrics
        """
        print(f"\n{'='*60}")
        print(f"Starting Offline IQL Training")
        print(f"{'='*60}")
        print(f"Training steps: {n_steps:,}")
        print(f"Evaluation: every {eval_frequency:,} steps ({eval_episodes} episodes)")
        print(f"Checkpoints: every {save_frequency:,} steps -> {save_dir}/")
        print(f"{'='*60}\n")

        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Training loop
        pbar = tqdm(range(n_steps), desc="Training")

        for step in pbar:
            self.global_step += 1

            # Sample batch and train
            batch = self.buffer.sample(self.config.batch_size)
            metrics = self.agent.update(batch)

            # Log metrics
            if self.global_step % log_frequency == 0:
                self._log_metrics(metrics)
                pbar.set_postfix({
                    'Q_loss': f"{metrics['loss_q']:.3f}",
                    'V_loss': f"{metrics['loss_v']:.3f}",
                    'π_loss': f"{metrics['loss_policy']:.3f}",
                    'Q': f"{metrics['q1']:.2f}"
                })

            # Evaluate
            if self.env and self.global_step % eval_frequency == 0:
                eval_metrics = self.evaluate(n_episodes=eval_episodes)
                self._log_eval_metrics(eval_metrics)

                print(f"\n[Step {self.global_step:,}] Evaluation:")
                print(f"  Mean reward: {eval_metrics['mean_reward']:.4f} ± {eval_metrics['std_reward']:.4f}")
                print(f"  Mean click rate: {eval_metrics['mean_click_rate']:.2%}")
                print(f"  Mean episode length: {eval_metrics['mean_length']:.1f}\n")

            # Save checkpoint
            if self.global_step % save_frequency == 0:
                checkpoint_path = save_path / f"checkpoint_step_{self.global_step}.pt"
                self.save_checkpoint(checkpoint_path)

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}\n")

        # Final evaluation
        if self.env:
            print("Running final evaluation...")
            final_eval = self.evaluate(n_episodes=50)
            print(f"\nFinal Results:")
            print(f"  Mean reward: {final_eval['mean_reward']:.4f} ± {final_eval['std_reward']:.4f}")
            print(f"  Mean click rate: {final_eval['mean_click_rate']:.2%}")
            print(f"  Best episode reward: {final_eval['max_reward']:.4f}")

        # Save final model
        final_path = save_path / "final_model.pt"
        self.save_checkpoint(final_path)
        print(f"\nFinal model saved to {final_path}")

        return self.metrics_history

    def evaluate(self, n_episodes: int = 10, deterministic: bool = True) -> Dict:
        """
        Evaluate trained policy in environment.

        Args:
            n_episodes: Number of episodes to run
            deterministic: Use deterministic policy (no exploration)

        Returns:
            Dictionary of evaluation metrics
        """
        if not self.env:
            raise ValueError("Environment not provided - cannot evaluate")

        episode_rewards = []
        episode_clicks = []
        episode_lengths = []

        for _ in range(n_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_click_count = 0
            step = 0

            done = False
            while not done:
                # Get action from policy
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action_continuous = self.agent.act(state_tensor, deterministic=deterministic)

                # Convert continuous action to discrete
                action_continuous_np = action_continuous.cpu().numpy()[0, 0]
                scaled = (action_continuous_np + 1) / 2  # [-1,1] -> [0,1]
                action_discrete = int(np.round(scaled * (self.env.action_space.n - 1)))
                action_discrete = np.clip(action_discrete, 0, self.env.action_space.n - 1)

                # Environment step
                next_state, reward, terminated, truncated, info = self.env.step(action_discrete)

                episode_reward += reward
                episode_click_count += info['interaction']['is_click']
                step += 1

                state = next_state
                done = terminated or truncated

            episode_rewards.append(episode_reward)
            episode_clicks.append(episode_click_count)
            episode_lengths.append(step)

        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_click_rate': np.mean(episode_clicks) / np.mean(episode_lengths),
            'mean_length': np.mean(episode_lengths)
        }

    def _log_metrics(self, metrics: Dict):
        """Store training metrics."""
        self.metrics_history['loss_q'].append(metrics['loss_q'])
        self.metrics_history['loss_v'].append(metrics['loss_v'])
        self.metrics_history['loss_policy'].append(metrics['loss_policy'])
        self.metrics_history['q1'].append(metrics['q1'])
        self.metrics_history['q2'].append(metrics['q2'])
        self.metrics_history['v'].append(metrics['v'])
        self.metrics_history['adv_mean'].append(metrics['adv_mean'])
        self.metrics_history['weight_mean'].append(metrics['weight_mean'])

    def _log_eval_metrics(self, eval_metrics: Dict):
        """Store evaluation metrics."""
        self.metrics_history['eval_reward'].append(eval_metrics['mean_reward'])
        self.metrics_history['eval_click_rate'].append(eval_metrics['mean_click_rate'])

    def save_checkpoint(self, path: str):
        """
        Save training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'agent_state': {
                'policy': self.agent.policy.state_dict(),
                'value': self.agent.value.state_dict(),
                'q1': self.agent.q1.state_dict(),
                'q2': self.agent.q2.state_dict(),
                'opt_policy': self.agent.opt_policy.state_dict(),
                'opt_v': self.agent.opt_v.state_dict(),
                'opt_q': self.agent.opt_q.state_dict(),
            },
            'config': self.config,
            'metrics_history': self.metrics_history,
            'buffer_stats': self.buffer.get_statistics()
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']

        # Load agent state
        self.agent.policy.load_state_dict(checkpoint['agent_state']['policy'])
        self.agent.value.load_state_dict(checkpoint['agent_state']['value'])
        self.agent.q1.load_state_dict(checkpoint['agent_state']['q1'])
        self.agent.q2.load_state_dict(checkpoint['agent_state']['q2'])
        self.agent.opt_policy.load_state_dict(checkpoint['agent_state']['opt_policy'])
        self.agent.opt_v.load_state_dict(checkpoint['agent_state']['opt_v'])
        self.agent.opt_q.load_state_dict(checkpoint['agent_state']['opt_q'])

        self.metrics_history = checkpoint['metrics_history']

        print(f"Checkpoint loaded: {path}")
        print(f"  Resumed at step {self.global_step:,}")

    def get_training_summary(self) -> Dict:
        """
        Get summary of training progress.

        Returns:
            Dictionary with training statistics
        """
        if not self.metrics_history['loss_q']:
            return {'message': 'No training data yet'}

        recent_steps = 100
        return {
            'global_step': self.global_step,
            'recent_q_loss': np.mean(self.metrics_history['loss_q'][-recent_steps:]),
            'recent_v_loss': np.mean(self.metrics_history['loss_v'][-recent_steps:]),
            'recent_policy_loss': np.mean(self.metrics_history['loss_policy'][-recent_steps:]),
            'recent_mean_q': np.mean(self.metrics_history['q1'][-recent_steps:]),
            'best_eval_reward': max(self.metrics_history['eval_reward']) if self.metrics_history['eval_reward'] else None,
            'latest_eval_reward': self.metrics_history['eval_reward'][-1] if self.metrics_history['eval_reward'] else None
        }
