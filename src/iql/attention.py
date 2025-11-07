import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    def __init__(self, input_dim, head_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.to_qkv = nn.Linear(input_dim, 3 * num_heads * head_dim, bias=False)
        self.to_out = nn.Linear(num_heads * head_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape  # Batch, Sequence length, Channels
        
        # Project to Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N, self.num_heads, self.head_dim)
                     .transpose(1, 2), qkv)
        
        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Attend to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        out = self.to_out(out)
        
        return out

class SelfAttentionLayer(nn.Module):
    """Self-attention layer with residual connection and layer norm."""
    def __init__(self, input_dim, head_dim=32, num_heads=8, dropout=0.1):
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
    def __init__(self, input_dim, context_dim, head_dim=32, num_heads=8, dropout=0.1):
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
    """Complete attention block with self-attention, cross-attention, and MLP."""
    def __init__(self, input_dim, context_dim=None, head_dim=32, num_heads=8, 
                 mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.self_attn = SelfAttentionLayer(input_dim, head_dim, num_heads, dropout)
        
        if context_dim is not None:
            self.cross_attn = CrossAttentionLayer(input_dim, context_dim, head_dim, 
                                                num_heads, dropout)
        else:
            self.cross_attn = None
            
        # MLP block
        mlp_hidden = int(input_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, input_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, context=None):
        x = self.self_attn(x)
        if context is not None and self.cross_attn is not None:
            x = self.cross_attn(x, context)
        x = x + self.mlp(x)
        return x