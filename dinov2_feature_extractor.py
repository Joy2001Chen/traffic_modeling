"""
DINOv2 Feature Extractor for Traffic Occupancy Heatmaps

This module provides a wrapper around facebook/dinov2-small for extracting
visual features from occupancy heatmap images.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from typing import Optional, Tuple


class DINOv2FeatureExtractor(nn.Module):
    """
    DINOv2-based feature extractor for occupancy heatmaps.
    
    Input: (batch_size, channels, height, width) - typically (B, 3, 224, 224)
    Output: (batch_size, feature_dim) - typically (B, 384) for dinov2-small
    """
    
    def __init__(
        self, 
        model_name: str = "facebook/dinov2-small",
        feature_dim: int = 384,
        freeze_backbone: bool = False,
        device: Optional[str] = None
    ):
        """
        Initialize DINOv2 feature extractor.
        
        Args:
            model_name: HuggingFace model identifier
            feature_dim: Output feature dimension (384 for dinov2-small)
            freeze_backbone: Whether to freeze the backbone weights
            device: Device to load model on
        """
        super().__init__()
        
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load DINOv2 model and processor
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Move to device
        self.model = self.model.to(self.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # Extract features using DINOv2
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.model(x)
            # Use CLS token representation
            features = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            
        return features
    
    def extract_features_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features for a batch of images.
        
        Args:
            images: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        return self.forward(images)
    
    def get_feature_dim(self) -> int:
        """Get the output feature dimension."""
        return self.feature_dim
    
    def save_pretrained(self, save_path: str):
        """Save the model to a local path."""
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
    
    def load_pretrained(self, load_path: str):
        """Load the model from a local path."""
        self.model = AutoModel.from_pretrained(load_path)
        self.processor = AutoImageProcessor.from_pretrained(load_path)
        self.model = self.model.to(self.device)


class DINOv2SequenceProcessor(nn.Module):
    """
    Process sequences of images using DINOv2 feature extractor.
    
    Input: (batch_size, sequence_length, channels, height, width)
    Output: (batch_size, sequence_length, feature_dim)
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        feature_dim: int = 384,
        freeze_backbone: bool = False,
        device: Optional[str] = None
    ):
        super().__init__()
        
        self.feature_extractor = DINOv2FeatureExtractor(
            model_name=model_name,
            feature_dim=feature_dim,
            freeze_backbone=freeze_backbone,
            device=device
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process sequence of images.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
            
        Returns:
            Feature tensor of shape (batch_size, sequence_length, feature_dim)
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Reshape to process all images at once
        x_reshaped = x.view(batch_size * seq_len, channels, height, width)
        
        # Extract features
        features = self.feature_extractor(x_reshaped)
        
        # Reshape back to sequence format
        features = features.view(batch_size, seq_len, -1)
        
        return features


if __name__ == "__main__":
    # Test the feature extractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create feature extractor
    extractor = DINOv2FeatureExtractor(device=device)
    
    # Test with dummy data
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    
    print(f"Input shape: {dummy_images.shape}")
    
    # Extract features
    features = extractor(dummy_images)
    print(f"Output shape: {features.shape}")
    print(f"Feature dimension: {extractor.get_feature_dim()}")
    
    # Test sequence processor
    seq_processor = DINOv2SequenceProcessor(device=device)
    seq_length = 8
    dummy_sequence = torch.randn(batch_size, seq_length, 3, 224, 224).to(device)
    
    print(f"\nSequence input shape: {dummy_sequence.shape}")
    seq_features = seq_processor(dummy_sequence)
    print(f"Sequence output shape: {seq_features.shape}")

