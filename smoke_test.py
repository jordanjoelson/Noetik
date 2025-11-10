import os
import sys
# Ensure src is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import torch
from src.iql.config import IQLConfig
from src.iql.batch import Batch
from agent import IQLAgent


def main():
    device = "cpu"
    cfg = IQLConfig(obs_dim=128, act_dim=1, batch_size=8)
    agent = IQLAgent(cfg, device=device)

    B = 8
    obs = torch.randn(B, cfg.obs_dim)
    act = torch.randn(B, cfg.act_dim)
    rew = torch.randn(B)
    next_obs = torch.randn(B, cfg.obs_dim)
    done = torch.zeros(B)

    batch = Batch(obs=obs, act=act, rew=rew, next_obs=next_obs, done=done)

    # pick a parameter from policy's first attention block if available
    try:
        p = agent.policy.attention_blocks[0].q_proj.weight
        name = "policy.attention_blocks[0].q_proj.weight"
    except Exception:
        # fallback to policy.out_net first linear if attention absent
        p = next(agent.policy.parameters())
        name = "policy.first_param"

    before = p.data.clone()

    print("Running one update step...")
    logs = agent.update(batch)
    print("Logs:", logs)

    after = p.data
    change = (after - before).norm().item()
    print(f"Parameter '{name}' change norm after update: {change:.6f}")


if __name__ == '__main__':
    main()
