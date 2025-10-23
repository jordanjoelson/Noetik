from dataclasses import dataclass
import torch


@dataclass
class Batch:
    obs: torch.Tensor       # (B, obs_dim)
    act: torch.Tensor       # (B, act_dim)
    rew: torch.Tensor       # (B,)
    next_obs: torch.Tensor  # (B, obs_dim)
    done: torch.Tensor      # (B,) 
