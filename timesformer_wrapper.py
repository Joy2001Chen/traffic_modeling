"""
Lightweight TimeSformer Wrapper for Traffic Sequence Modeling

This module implements a lightweight version of TimeSformer for temporal
modeling of traffic occupancy sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism for temporal modeling.
    
    Input: (batch_size, sequence_length, feature_dim)
    Output: (batch_size, sequence_length, feature_dim)
    """
    
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert feature_dim % num_heads == 0
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(feature_dim, 3 * feature_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(feature_dim, feature_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feature_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, feature_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TemporalBlock(nn.Module):
    """
    Temporal transformer block with self-attention and feed-forward network.
    
    Input: (batch_size, sequence_length, feature_dim)
    Output: (batch_size, sequence_length, feature_dim)
    """
    
    def __init__(
        self, 
        feature_dim: int, 
        num_heads: int = 8, 
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.attn = MultiHeadSelfAttention(feature_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(feature_dim)
        mlp_hidden_dim = int(feature_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, feature_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # Feed-forward with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class LightweightTimeSformer(nn.Module):
    """
    Lightweight TimeSformer for temporal modeling of traffic sequences.
    
    Input: (batch_size, sequence_length, feature_dim)
    Output: (batch_size, sequence_length, feature_dim) or (batch_size, feature_dim)
    """
    
    def __init__(
        self,
        feature_dim: int = 384,
        sequence_length: int = 8,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_cls_token: bool = True,
        output_mode: str = "sequence"  # "sequence" or "cls"
    ):
        """
        Initialize lightweight TimeSformer.
        
        Args:
            feature_dim: Input/output feature dimension
            sequence_length: Length of input sequence
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
            use_cls_token: Whether to use CLS token for global representation
            output_mode: "sequence" for full sequence output, "cls" for CLS token only
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.use_cls_token = use_cls_token
        self.output_mode = output_mode
        
        # CLS token (optional)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, sequence_length + 1, feature_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, sequence_length, feature_dim))
        
        # Transformer layers
        self.blocks = nn.ModuleList([
            TemporalBlock(feature_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(feature_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        if self.use_cls_token:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    torch.nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        torch.nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.LayerNorm):
                    torch.nn.init.constant_(module.bias, 0)
                    torch.nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TimeSformer.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, feature_dim)
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, feature_dim) or (batch_size, feature_dim)
        """
        batch_size, seq_len, feature_dim = x.shape
        
        # Add CLS token if enabled
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            seq_len = seq_len + 1
        
        # Add positional embedding
        x = x + self.pos_embed[:, :seq_len, :]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final layer norm
        x = self.norm(x)
        
        # Return output based on mode
        if self.output_mode == "cls" and self.use_cls_token:
            return x[:, 0, :]  # Return CLS token only
        elif self.output_mode == "sequence":
            if self.use_cls_token:
                return x[:, 1:, :]  # Return sequence without CLS token
            else:
                return x
        else:
            raise ValueError(f"Invalid output_mode: {self.output_mode}")
    
    def get_attention_weights(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Get attention weights from a specific layer.
        
        Args:
            x: Input tensor
            layer_idx: Layer index (-1 for last layer)
            
        Returns:
            Attention weights tensor
        """
        batch_size, seq_len, feature_dim = x.shape
        
        # Add CLS token if enabled
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            seq_len = seq_len + 1
        
        # Add positional embedding
        x = x + self.pos_embed[:, :seq_len, :]
        
        # Apply blocks up to the specified layer
        for i, block in enumerate(self.blocks):
            if i == layer_idx:
                # Get attention weights from this layer
                x_norm = block.norm1(x)
                qkv = block.attn.qkv(x_norm).reshape(batch_size, seq_len, 3, block.attn.num_heads, block.attn.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                attn = (q @ k.transpose(-2, -1)) * block.attn.scale
                attn = attn.softmax(dim=-1)
                return attn
            x = block(x)
        
        return None


class TemporalConv1D(nn.Module):
    """
    Alternative temporal modeling using 1D convolutions.
    
    Input: (batch_size, sequence_length, feature_dim)
    Output: (batch_size, sequence_length, feature_dim)
    """
    
    def __init__(
        self,
        feature_dim: int,
        kernel_sizes: list = [3, 5, 7],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feature_dim, feature_dim, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for k in kernel_sizes
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv1d(feature_dim * len(kernel_sizes), feature_dim, 1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, feature_dim)
        x = x.transpose(1, 2)  # (batch_size, feature_dim, seq_len)
        
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x))
        
        # Concatenate and fuse
        x = torch.cat(conv_outputs, dim=1)
        x = self.fusion(x)
        
        return x.transpose(1, 2)  # (batch_size, seq_len, feature_dim)


if __name__ == "__main__":
    # Test the TimeSformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test parameters
    batch_size = 2
    seq_len = 8
    feature_dim = 384
    
    # Create TimeSformer
    timesformer = LightweightTimeSformer(
        feature_dim=feature_dim,
        sequence_length=seq_len,
        num_layers=4,
        num_heads=8,
        output_mode="sequence"
    ).to(device)
    
    # Test input
    dummy_input = torch.randn(batch_size, seq_len, feature_dim).to(device)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    output = timesformer(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # Test CLS token mode
    timesformer_cls = LightweightTimeSformer(
        feature_dim=feature_dim,
        sequence_length=seq_len,
        num_layers=4,
        num_heads=8,
        output_mode="cls"
    ).to(device)
    
    output_cls = timesformer_cls(dummy_input)
    print(f"CLS output shape: {output_cls.shape}")
    
    # Test temporal conv
    temporal_conv = TemporalConv1D(feature_dim=feature_dim).to(device)
    conv_output = temporal_conv(dummy_input)
    print(f"Conv output shape: {conv_output.shape}")

