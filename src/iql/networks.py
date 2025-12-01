import math
import torch
import torch.nn as nn
from .utils import build_mlp
from .attention import AttentionBlock


class ValueNet(nn.Module):

    """V(s): predicts a scalar value for each state with attention mechanisms.

    The observation dimension can vary (64, 128, etc.). We compute feature
    splits from the provided obs_dim so the network will work regardless of
    exact state vector sizing.
    """
    def __init__(self, obs_dim: int, hidden=(256, 256), num_attention_blocks=2):
        super().__init__()
        
        # Derive feature splits from obs_dim to support different state sizes.
        # Default design intent: user ~25%, history ~50%, context ~25%.
        self.user_dim = obs_dim // 4
        self.history_dim = obs_dim // 2
        self.context_dim = obs_dim - (self.user_dim + self.history_dim)
        
        # Input projections (match derived dims)
        self.user_proj = nn.Linear(self.user_dim, hidden[0])
        self.history_proj = nn.Linear(self.history_dim, hidden[0])
        self.context_proj = nn.Linear(self.context_dim, hidden[0])
        
        # Attention blocks
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(
                input_dim=hidden[0],
                context_dim=hidden[0],
                head_dim=32,
                num_heads=8,
                dropout=0.1
            ) for _ in range(num_attention_blocks)
        ])
        
        # Final MLP layers
        self.out_net = build_mlp(hidden[0], hidden[1:], 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        
        # Split observation into components
        user = obs[:, :self.user_dim]
        history = obs[:, self.user_dim:self.user_dim+self.history_dim]
        context = obs[:, -self.context_dim:]
        
        # Project each component
        user = self.user_proj(user).unsqueeze(1)  # (B, 1, H)
        history = self.history_proj(history).unsqueeze(1)  # (B, 1, H) 
        context = self.context_proj(context).unsqueeze(1)  # (B, 1, H)
        
        # Combine sequences for attention
        x = torch.cat([user, history, context], dim=1)  # (B, 3, H)
        
        # Apply attention blocks
        for block in self.attention_blocks:
            x = block(x)
            
        # Average pooling across sequence dimension
        x = x.mean(dim=1)  # (B, H)
        
        # Final MLP layers
        return self.out_net(x).squeeze(-1)  # (B,)


class QNet(nn.Module):
    """Q(s,a): predicts a scalar value for each (state, action) pair with attention."""
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256), num_attention_blocks=2):
        super().__init__()
        
        # Derive feature splits from obs_dim (same policy as ValueNet)
        self.user_dim = obs_dim // 4
        self.history_dim = obs_dim // 2
        self.context_dim = obs_dim - (self.user_dim + self.history_dim)

        # Input projections
        self.user_proj = nn.Linear(self.user_dim, hidden[0])
        self.history_proj = nn.Linear(self.history_dim, hidden[0])
        self.context_proj = nn.Linear(self.context_dim, hidden[0])
        self.action_proj = nn.Linear(act_dim, hidden[0])
        
        # Attention blocks
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(
                input_dim=hidden[0],
                context_dim=hidden[0],
                head_dim=32,
                num_heads=8,
                dropout=0.1
            ) for _ in range(num_attention_blocks)
        ])
        
        # Final MLP layers
        self.out_net = build_mlp(hidden[0], hidden[1:], 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        
        # Split observation into components
        user = obs[:, :self.user_dim]
        history = obs[:, self.user_dim:self.user_dim+self.history_dim]
        context = obs[:, -self.context_dim:]
        
        # Project each component
        user = self.user_proj(user).unsqueeze(1)  # (B, 1, H)
        history = self.history_proj(history).unsqueeze(1)  # (B, 1, H)
        context = self.context_proj(context).unsqueeze(1)  # (B, 1, H)
        action = self.action_proj(act).unsqueeze(1)  # (B, 1, H)
        
        # Combine sequences for attention
        x = torch.cat([user, history, context, action], dim=1)  # (B, 4, H)
        
        # Apply attention blocks
        for block in self.attention_blocks:
            x = block(x)
            
        # Average pooling across sequence dimension
        x = x.mean(dim=1)  # (B, H)
        
        # Final MLP layers
        return self.out_net(x).squeeze(-1)  # (B,)


class GaussianTanhPolicy(nn.Module):
    """Diagonal Gaussian policy with Tanh squashing and attention mechanisms.

    • Uses attention to process state components
    • The backbone predicts [mu, log_std] for each action dimension
    • We sample z ~ Normal(mu, std), then a = tanh(z) so actions are in [-1,1]
    • We also compute the correct log-prob of 'a' under the squashed Gaussian
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256),
                 log_std_min=-5.0, log_std_max=2.0, num_attention_blocks=2):
        super().__init__()
        
        # Derive feature splits from obs_dim so policy matches value/q networks
        self.user_dim = obs_dim // 4
        self.history_dim = obs_dim // 2
        self.context_dim = obs_dim - (self.user_dim + self.history_dim)

        # Input projections
        self.user_proj = nn.Linear(self.user_dim, hidden[0])
        self.history_proj = nn.Linear(self.history_dim, hidden[0])
        self.context_proj = nn.Linear(self.context_dim, hidden[0])
        
        # Attention blocks
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(
                input_dim=hidden[0],
                context_dim=hidden[0],
                head_dim=32,
                num_heads=8,
                dropout=0.15   #slight increase in dropout for policy for added variability between attention heads
            ) for _ in range(num_attention_blocks)
        ])
        
        # Output head
        self.out_net = build_mlp(hidden[0], hidden[1:], 2 * act_dim)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.act_dim = act_dim

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        B = obs.shape[0]
        
        # Split observation into components
        user = obs[:, :self.user_dim]
        history = obs[:, self.user_dim:self.user_dim+self.history_dim]
        context = obs[:, -self.context_dim:]
        
        # Project each component
        user = self.user_proj(user).unsqueeze(1)  # (B, 1, H)
        history = self.history_proj(history).unsqueeze(1)  # (B, 1, H)
        context = self.context_proj(context).unsqueeze(1)  # (B, 1, H)
        
        # Combine sequences for attention
        x = torch.cat([user, history, context], dim=1)  # (B, 3, H)
        
        # Apply attention blocks
        for block in self.attention_blocks:
            x = block(x)
            
        # Average pooling across sequence dimension
        x = x.mean(dim=1)  # (B, H)
        
        # Get mu and log_std from output network
        mu_logstd = self.out_net(x)
        mu, log_std = mu_logstd.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # Sample pre-squash action z
        if deterministic:
            pre_tanh = mu
        else:
            noise = torch.randn_like(mu)
            pre_tanh = mu + std * noise

        # Squash to [-1,1]
        action = torch.tanh(pre_tanh)

        # Compute log-prob with Tanh correction
        # Base Gaussian log-prob for z under N(mu, std):
        base_log_prob = -0.5 * (((pre_tanh - mu) / (std + 1e-8)) ** 2 + 2 * log_std + math.log(2 * math.pi))
        base_log_prob = base_log_prob.sum(dim=-1)  # sum over action dims
        # Change of variables: derivative of tanh is (1 - tanh(z)^2)
        log_det = torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        log_prob = base_log_prob - log_det

        return action, log_prob, mu, log_std