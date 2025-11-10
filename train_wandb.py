"""
train_wandb.py

Run a short training loop with optional Weights & Biases logging. Defaults to offline mode
unless `--online` is passed. Checkpointing and attention heatmap logging are supported.
"""

import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    import wandb
except Exception:
    wandb = None

import torch
import numpy as np
from src.data_loader import KuaiRandDataLoader
from src.environment import KuaiRandEnv
from agent import IQLAgent
from src.iql.config import IQLConfig
from test_attention import AttentionAnalyzer


def main(project_name: str = "noetik-iql", run_name: str = None, n_episodes: int = 200, online: bool = False, checkpoint_interval: int = 10, image_interval: int = 10):
    """Main training loop.

    Args:
        project_name: W&B project name
        run_name: optional run name
        n_episodes: number of episodes to run
        online: whether to enable online W&B logging
        checkpoint_interval: save checkpoint every N episodes
        image_interval: save attention image every N episodes
    """

    # Initialize wandb only if online requested and wandb is installed
    run = None
    if online:
        if wandb is None:
            print("wandb package not installed; running in offline/local mode")
        else:
            run = wandb.init(project=project_name, name=run_name)
            try:
                print("W&B Run:", run.url)
            except Exception:
                pass

    print("Loading data...")
    loader = KuaiRandDataLoader()
    try:
        loader.load_processed_data()
    except FileNotFoundError:
        loader.load_data(random_only=True)
        loader.save_processed_data()

    env = KuaiRandEnv(data_loader=loader, max_episode_length=10)

    cfg = IQLConfig(
        obs_dim=128,
        act_dim=1,
        hidden_sizes=(256, 256),
        discount=0.99,
        tau=0.7,
        temperature=3.0,
        lr_policy=3e-4,
        lr_q=3e-4,
        lr_v=3e-4,
        batch_size=256,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = IQLAgent(cfg, device=device)

    analyzer = AttentionAnalyzer(agent.value, agent.q1, agent.policy)

    global_step = 0

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_clicks = 0

        for t in range(env.max_episode_length):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action_cont = agent.act(state_tensor, deterministic=False)

            scaled = (action_cont + 1) / 2
            action_discrete = int((scaled * (env.action_space.n - 1)).round().clamp(0, env.action_space.n - 1).item())

            next_state, reward, terminated, truncated, info = env.step(action_discrete)
            episode_reward += reward
            episode_clicks += info['interaction']['is_click']

            try:
                agent.replay_buffer.push(state, action_cont.squeeze().cpu().numpy(), reward, next_state, terminated or truncated)
            except Exception:
                pass

            state = next_state
            global_step += 1

            # Optional update step (not implemented fully here)
            try:
                if hasattr(agent, 'update') and hasattr(agent, 'replay_buffer'):
                    if len(agent.replay_buffer) >= cfg.batch_size:
                        batch = agent.replay_buffer.sample(cfg.batch_size)
                        # agent.update(batch)  # Uncomment if you want in-loop updates
            except Exception:
                pass

            if terminated or truncated:
                break

        # Log scalars if online
        if wandb is not None and run is not None:
            wandb.log({'episode': episode, 'episode_reward': episode_reward, 'episode_clicks': episode_clicks, 'global_step': global_step}, step=episode)

        # Attention importance
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        try:
            importance = analyzer.analyze_component_importance(state_tensor)
            if wandb is not None and run is not None:
                wandb.log({f'attn_{k}': v for k, v in importance.items()}, step=episode)
        except Exception as e:
            print("Warning: failed to extract attention weights:", e)

        # Model predictions
        try:
            obs_batch = torch.randn(8, cfg.obs_dim).to(device)
            with torch.no_grad():
                actions, logp, mu, log_std = agent.policy(obs_batch)
                v = agent.value(obs_batch)
            if wandb is not None and run is not None:
                wandb.log({'pred_action_mean': mu.mean().item(), 'pred_action_std': log_std.exp().mean().item(), 'pred_value_mean': v.mean().item()}, step=episode)
        except Exception as e:
            print("Warning: failed to log model predictions:", e)

        # Checkpoint
        try:
            ckpt_dir = os.path.join("checkpoints", (run.name or run.id) if run is not None else "local")
            os.makedirs(ckpt_dir, exist_ok=True)
            if (episode + 1) % checkpoint_interval == 0 or (episode + 1) == n_episodes:
                ckpt_path = os.path.join(ckpt_dir, f"ckpt_ep{episode+1}.pt")
                torch.save({'episode': episode + 1, 'policy_state': agent.policy.state_dict(), 'value_state': agent.value.state_dict(), 'q1_state': agent.q1.state_dict(), 'q2_state': agent.q2.state_dict(), 'cfg': cfg}, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")
                try:
                    if wandb is not None and run is not None:
                        wandb.save(ckpt_path)
                except Exception:
                    pass
        except Exception as e:
            print("Warning: checkpoint save failed:", e)

        # Attention image
        try:
            if (episode + 1) % image_interval == 0 or (episode + 1) == n_episodes:
                img_path = os.path.join(analyzer.output_dir, f"attn_ep{episode+1}.png")
                analyzer.visualize_attention_patterns(state_tensor, save_path=str(img_path))
                try:
                    if wandb is not None and run is not None:
                        wandb.log({"attention_heatmap": wandb.Image(str(img_path))}, step=episode)
                except Exception:
                    pass
        except Exception as e:
            print("Warning: failed to log attention image:", e)

    if wandb is not None and run is not None:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train with optional W&B logging")
    parser.add_argument("--project", type=str, default="noetik-iql")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--online", action="store_true", help="Enable online W&B logging (requires wandb and login)")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--image-interval", type=int, default=10)

    args = parser.parse_args()
    main(project_name=args.project, run_name=args.name, n_episodes=args.episodes, online=args.online, checkpoint_interval=args.checkpoint_interval, image_interval=args.image_interval)
