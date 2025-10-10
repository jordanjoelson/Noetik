import math
import torch
import torch.nn as nn
from .utils import build_mlp


class ValueNet(nn.Module):
    """V(s): predicts a scalar value for each state.

    Training signal comes from expectile regression to Q (see losses.py).
    """
    def __init__(self, obs_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Output shape: (B,) where B = batch size
        return self.net(obs).squeeze(-1)


class QNet(nn.Module):
    """Q(s,a): predicts a scalar value for each (state, action) pair.

    We concatenate obs and act along the feature dimension and feed to an MLP.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = build_mlp(obs_dim + act_dim, hidden, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x).squeeze(-1)  # (B,)


class GaussianTanhPolicy(nn.Module):
    """Diagonal Gaussian policy with Tanh squashing.

    • The backbone predicts [mu, log_std] for each action dimension.
    • We sample z ~ Normal(mu, std), then a = tanh(z) so actions are in [-1,1].
    • We also compute the correct log-prob of 'a' under the squashed Gaussian.

    Why Tanh? Many continuous-control tasks expect bounded actions. Tanh gives
    us a smooth way to stay inside bounds.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256),
                 log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.backbone = build_mlp(obs_dim, hidden, 2 * act_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.act_dim = act_dim

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        # 1) Get mu and log_std from the backbone
        mu_logstd = self.backbone(obs)
        mu, log_std = mu_logstd.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # 2) Sample pre-squash action z
        if deterministic:
            pre_tanh = mu
        else:
            noise = torch.randn_like(mu)
            pre_tanh = mu + std * noise

        # 3) Squash to [-1,1]
        action = torch.tanh(pre_tanh)

        # 4) Compute log-prob with Tanh correction.
        # Base Gaussian log-prob for z under N(mu, std):
        base_log_prob = -0.5 * (((pre_tanh - mu) / (std + 1e-8)) ** 2 + 2 * log_std + math.log(2 * math.pi))
        base_log_prob = base_log_prob.sum(dim=-1)  # sum over action dims
        # Change of variables: derivative of tanh is (1 - tanh(z)^2)
        log_det = torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        log_prob = base_log_prob - log_det

        return action, log_prob, mu, log_std