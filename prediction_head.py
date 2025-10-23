"""
Prediction Head Modules for Traffic Modeling

This module provides prediction heads for different tasks:
- Regression: Predict future occupancy values
- Classification: Classify traffic states (smooth/medium/congested)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RegressionHead(nn.Module):
    """
    Regression head for predicting future occupancy values.
    
    Input: (batch_size, feature_dim) or (batch_size, sequence_length, feature_dim)
    Output: (batch_size, num_predictions) - predicted occupancy values
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_predictions: int = 2,  # Predict 1-2 future frames
        hidden_dims: list = [256, 128],
        dropout: float = 0.1,
        input_mode: str = "cls"  # "cls" or "sequence"
    ):
        """
        Initialize regression head.
        
        Args:
            feature_dim: Input feature dimension
            num_predictions: Number of future predictions
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            input_mode: Input mode ("cls" for CLS token, "sequence" for full sequence)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_predictions = num_predictions
        self.input_mode = input_mode
        
        # Build MLP layers
        layers = []
        prev_dim = feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_predictions))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through regression head.
        
        Args:
            x: Input tensor of shape (batch_size, feature_dim) or (batch_size, seq_len, feature_dim)
            
        Returns:
            Predicted occupancy values of shape (batch_size, num_predictions)
        """
        if self.input_mode == "sequence" and len(x.shape) == 3:
            # Use global average pooling for sequence input
            x = x.mean(dim=1)  # (batch_size, feature_dim)
        
        return self.mlp(x)


class ClassificationHead(nn.Module):
    """
    Classification head for traffic state classification.
    
    Input: (batch_size, feature_dim) or (batch_size, sequence_length, feature_dim)
    Output: (batch_size, num_classes) - class probabilities
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 3,  # smooth, medium, congested
        hidden_dims: list = [256, 128],
        dropout: float = 0.1,
        input_mode: str = "cls"  # "cls" or "sequence"
    ):
        """
        Initialize classification head.
        
        Args:
            feature_dim: Input feature dimension
            num_classes: Number of traffic state classes
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            input_mode: Input mode ("cls" for CLS token, "sequence" for full sequence)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.input_mode = input_mode
        
        # Build MLP layers
        layers = []
        prev_dim = feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.
        
        Args:
            x: Input tensor of shape (batch_size, feature_dim) or (batch_size, seq_len, feature_dim)
            
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        if self.input_mode == "sequence" and len(x.shape) == 3:
            # Use global average pooling for sequence input
            x = x.mean(dim=1)  # (batch_size, feature_dim)
        
        return self.mlp(x)


class Conv1DPredictionHead(nn.Module):
    """
    Conv1D-based prediction head for temporal modeling.
    
    Input: (batch_size, sequence_length, feature_dim)
    Output: (batch_size, num_predictions) or (batch_size, num_classes)
    """
    
    def __init__(
        self,
        feature_dim: int,
        sequence_length: int,
        task_type: str = "regression",  # "regression" or "classification"
        num_outputs: int = 2,  # num_predictions for regression, num_classes for classification
        conv_channels: list = [256, 128],
        kernel_sizes: list = [3, 3],
        dropout: float = 0.1
    ):
        """
        Initialize Conv1D prediction head.
        
        Args:
            feature_dim: Input feature dimension
            sequence_length: Input sequence length
            task_type: Task type ("regression" or "classification")
            num_outputs: Number of outputs
            conv_channels: Convolutional layer channels
            kernel_sizes: Kernel sizes for conv layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.task_type = task_type
        self.num_outputs = num_outputs
        
        # Build Conv1D layers
        conv_layers = []
        prev_channels = feature_dim
        
        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            conv_layers.extend([
                nn.Conv1d(prev_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Global pooling and output layer
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(prev_channels, num_outputs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Conv1D head.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, feature_dim)
            
        Returns:
            Output tensor of shape (batch_size, num_outputs)
        """
        # Transpose for Conv1D: (batch_size, feature_dim, sequence_length)
        x = x.transpose(1, 2)
        
        # Apply Conv1D layers
        x = self.conv_layers(x)
        
        # Global pooling
        x = self.global_pool(x)  # (batch_size, channels, 1)
        x = x.squeeze(-1)  # (batch_size, channels)
        
        # Output layer
        output = self.output_layer(x)
        
        return output


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head for both regression and classification.
    
    Input: (batch_size, feature_dim) or (batch_size, sequence_length, feature_dim)
    Output: Tuple of (regression_output, classification_output)
    """
    
    def __init__(
        self,
        feature_dim: int,
        regression_outputs: int = 2,
        num_classes: int = 3,
        shared_hidden_dims: list = [256],
        task_hidden_dims: list = [128],
        dropout: float = 0.1,
        input_mode: str = "cls"
    ):
        """
        Initialize multi-task head.
        
        Args:
            feature_dim: Input feature dimension
            regression_outputs: Number of regression outputs
            num_classes: Number of classification classes
            shared_hidden_dims: Shared hidden layer dimensions
            task_hidden_dims: Task-specific hidden layer dimensions
            dropout: Dropout rate
            input_mode: Input mode ("cls" or "sequence")
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.input_mode = input_mode
        
        # Shared layers
        shared_layers = []
        prev_dim = feature_dim
        
        for hidden_dim in shared_hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # Task-specific heads
        self.regression_head = nn.Sequential(
            nn.Linear(prev_dim, task_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(task_hidden_dims[0], regression_outputs)
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(prev_dim, task_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(task_hidden_dims[0], num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-task head.
        
        Args:
            x: Input tensor of shape (batch_size, feature_dim) or (batch_size, seq_len, feature_dim)
            
        Returns:
            Tuple of (regression_output, classification_output)
        """
        if self.input_mode == "sequence" and len(x.shape) == 3:
            # Use global average pooling for sequence input
            x = x.mean(dim=1)  # (batch_size, feature_dim)
        
        # Shared representation
        shared_features = self.shared_layers(x)
        
        # Task-specific outputs
        regression_output = self.regression_head(shared_features)
        classification_output = self.classification_head(shared_features)
        
        return regression_output, classification_output


class AttentionPoolingHead(nn.Module):
    """
    Attention-based pooling head for sequence inputs.
    
    Input: (batch_size, sequence_length, feature_dim)
    Output: (batch_size, num_outputs)
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_outputs: int,
        attention_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize attention pooling head.
        
        Args:
            feature_dim: Input feature dimension
            num_outputs: Number of outputs
            attention_dim: Attention dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_outputs = num_outputs
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_outputs)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention pooling head.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, feature_dim)
            
        Returns:
            Output tensor of shape (batch_size, num_outputs)
        """
        # Compute attention weights
        attention_weights = self.attention(x)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended_features = torch.sum(x * attention_weights, dim=1)  # (batch_size, feature_dim)
        
        # Output layer
        output = self.output_layer(attended_features)
        
        return output


if __name__ == "__main__":
    # Test prediction heads
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    batch_size = 4
    feature_dim = 384
    seq_len = 8
    
    # Test regression head
    print("Testing Regression Head:")
    reg_head = RegressionHead(feature_dim=feature_dim, num_predictions=2).to(device)
    
    # Test with CLS token input
    cls_input = torch.randn(batch_size, feature_dim).to(device)
    reg_output = reg_head(cls_input)
    print(f"CLS input shape: {cls_input.shape}")
    print(f"Regression output shape: {reg_output.shape}")
    
    # Test with sequence input
    seq_input = torch.randn(batch_size, seq_len, feature_dim).to(device)
    reg_head_seq = RegressionHead(feature_dim=feature_dim, num_predictions=2, input_mode="sequence").to(device)
    reg_output_seq = reg_head_seq(seq_input)
    print(f"Sequence input shape: {seq_input.shape}")
    print(f"Regression output shape: {reg_output_seq.shape}")
    
    # Test classification head
    print("\nTesting Classification Head:")
    cls_head = ClassificationHead(feature_dim=feature_dim, num_classes=3).to(device)
    cls_output = cls_head(cls_input)
    print(f"Classification output shape: {cls_output.shape}")
    
    # Test Conv1D head
    print("\nTesting Conv1D Head:")
    conv_head = Conv1DPredictionHead(
        feature_dim=feature_dim,
        sequence_length=seq_len,
        task_type="regression",
        num_outputs=2
    ).to(device)
    conv_output = conv_head(seq_input)
    print(f"Conv1D output shape: {conv_output.shape}")
    
    # Test multi-task head
    print("\nTesting Multi-Task Head:")
    multi_head = MultiTaskHead(feature_dim=feature_dim).to(device)
    reg_out, cls_out = multi_head(cls_input)
    print(f"Multi-task regression output shape: {reg_out.shape}")
    print(f"Multi-task classification output shape: {cls_out.shape}")
    
    # Test attention pooling head
    print("\nTesting Attention Pooling Head:")
    att_head = AttentionPoolingHead(feature_dim=feature_dim, num_outputs=2).to(device)
    att_output = att_head(seq_input)
    print(f"Attention pooling output shape: {att_output.shape}")
