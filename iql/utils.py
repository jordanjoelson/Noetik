import torch
import torch.nn as nn
from typing import Iterable
import pandas as pd
import numpy as np
from .batch import Batch


def fanin_init(m: nn.Module):
    """Initialize Linear layers uniformly in ±1/sqrt(fan_in).

    Why? If initial weights are too big/small, activations and gradients can explode
    or vanish. This simple scheme keeps things in a reasonable range at the start.
    """
    if isinstance(m, nn.Linear):
        bound = 1.0 / (m.weight.size(1) ** 0.5)
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.uniform_(m.bias, -bound, bound)


def build_mlp(in_dim: int, hidden: Iterable[int], out_dim: int, act=nn.ReLU) -> nn.Sequential:
    """Build a simple MLP: [Linear -> Activation] * N  then Linear -> out_dim.

    Args:
      in_dim: input feature size
      hidden: iterable of hidden layer sizes (e.g., (256, 256))
      out_dim: final output size
      act: activation class (default nn.ReLU). Try nn.SiLU for a smooth alternative.
    """
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), act()]
        prev = h
    layers += [nn.Linear(prev, out_dim)]
    net = nn.Sequential(*layers)
    net.apply(fanin_init)
    return net

def preprocess_kuairand(csv_path: str, obs_cols=None, act_cols=None, reward_col="long_view"):
    """
    Preprocess the KuaiRand 1k dataset for IQL training.

    Args:
        csv_path: path to kuairand CSV file
        obs_cols: list of columns to use as observations (state)
        act_cols: list of columns to use as actions
        reward_col: column to use as reward

    Returns:
        Batch object containing obs, act, rew, next_obs, done
    """
    import pandas as pd
    import torch
    from iql.batch import Batch

    df = pd.read_csv(csv_path, sep=",")

    if "is_rand" in df.columns:
        df = df[df["is_rand"] == 0].reset_index(drop=True)

    df = df.sort_values(["user_id", "time_ms"]).reset_index(drop=True)

    if obs_cols is None:
        obs_cols = ["is_click", "is_like", "is_follow", "is_comment", "is_forward", "play_time_ms"]
    if act_cols is None:
        act_cols = ["is_click", "is_like", "is_follow"]
    if reward_col not in df.columns:
        raise ValueError(f"Reward column {reward_col} not found in dataset")

    obs = torch.tensor(df[obs_cols].values, dtype=torch.float32)
    act = torch.tensor(df[act_cols].values, dtype=torch.float32)
    rew = torch.tensor(df[reward_col].values, dtype=torch.float32)

    next_obs = torch.tensor(df.groupby("user_id")[obs_cols].shift(-1).fillna(0).values, dtype=torch.float32)
    done = torch.tensor(df.groupby("user_id")[obs_cols[0]].shift(-1).isna().fillna(1).values, dtype=torch.float32)

    batch = Batch(obs=obs, act=act, rew=rew, next_obs=next_obs, done=done)

    return batch
