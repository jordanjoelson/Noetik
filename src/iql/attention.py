import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with adaptive scaling and dynamic temperature."""
    def __init__(self, input_dim, head_dim, num_heads=8, dropout=0.17, temperature=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.input_dim = input_dim
        
        # Learnable scaling factors
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5))
        self.base_temperature = temperature
        self.dynamic_temp = nn.Parameter(torch.ones(num_heads))
        
        # Adaptive gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_heads),
            nn.Sigmoid()
        )
        
        # Linear projections for Q, K, V
        self.to_qkv = nn.Linear(input_dim, 3 * num_heads * head_dim, bias=False)
        self.to_out = nn.Linear(num_heads * head_dim, input_dim)
        
        # Mixture of experts-style weighting
        self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
        
        self.dropout = nn.Dropout(dropout)
        self.last_attn = None  # For storing attention weights
        
    def forward(self, x):
        B, N, C = x.shape  # Batch, Sequence length, Channels
        
        # Compute adaptive gating factors
        gates = self.gate(x.mean(dim=1))  # [B, num_heads]
        
        # Project to Q, K, V with gating
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N, self.num_heads, self.head_dim)
                     .transpose(1, 2), qkv)
        
        # Apply dynamic temperature per head
        temps = F.softplus(self.dynamic_temp).view(1, self.num_heads, 1, 1)
        
        # Compute attention scores with adaptive scaling
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        dots = dots / (temps * self.base_temperature)
        
        # Apply softmax with gating
        attn = F.softmax(dots, dim=-1)
        attn = attn * gates.view(B, self.num_heads, 1, 1)
        attn = self.dropout(attn)
        
        # Store attention weights for analysis
        self.last_attn = attn
        
        # Attend to values with head weighting
        out = torch.matmul(attn, v)  # [B, H, N, D]
        out = out.transpose(1, 2)  # [B, N, H, D]
        
        # Apply head weights for mixture-of-experts style combination
        head_weights = F.softmax(self.head_weights, dim=0)
        out = out * head_weights.view(1, 1, self.num_heads, 1)
        
        # Reshape and project to output
        out = out.reshape(B, N, self.num_heads * self.head_dim)
        out = self.to_out(out)
        
        return out

class SelfAttentionLayer(nn.Module):
    """Self-attention layer with residual connection and layer norm."""
    def __init__(self, input_dim, head_dim=32, num_heads=8, dropout=0.15):
        super().__init__()
        self.attention = MultiHeadAttention(input_dim, head_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Residual connection
        out = x + self.dropout(self.attention(self.norm(x)))
        return out

class CrossAttentionLayer(nn.Module):
    """Cross-attention layer between state and context."""
    def __init__(self, input_dim, context_dim, head_dim=32, num_heads=8, dropout=0.15):
        super().__init__()
        self.attention = MultiHeadAttention(input_dim, head_dim, num_heads, dropout)
        self.context_proj = nn.Linear(context_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, context):
        # Project context to input dimension
        context = self.context_proj(context)
        
        # Residual connection
        out = x + self.dropout(self.attention(torch.cat([self.norm(x), context], dim=1)))
        return out

class AttentionBlock(nn.Module):
    """
    Self- or cross-attention block with per-head adaptive scaling and temperature control.
    """

    def __init__(self, input_dim, context_dim=None, head_dim=32, num_heads=8, dropout=0.1, temp=1.0):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5  # default attention scale
        self.temperature = temp       # softmax temperature

        # Per-head learnable adaptive scale
        self.adaptive_scale = nn.Parameter(torch.ones(num_heads))

        self.q_proj = nn.Linear(input_dim, num_heads * head_dim)
        self.k_proj = nn.Linear(input_dim if context_dim is None else context_dim, num_heads * head_dim)
        self.v_proj = nn.Linear(input_dim if context_dim is None else context_dim, num_heads * head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, input_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

        # store last attention for analysis
        self.last_attn = None

    def forward(self, x, context=None):
        B, N, D = x.shape
        if context is None:
            context = x

        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Adaptive per-head scaling
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * (self.scale * self.adaptive_scale.view(1, -1, 1, 1))

        # Temperature control
        attn_scores = attn_scores / self.temperature

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Save for analysis
        self.last_attn = attn_weights.detach()

        # Weighted sum
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, -1)

        # Residual connection + LayerNorm
        x = self.norm(x + self.dropout(self.out_proj(attn_output)))
        return x
