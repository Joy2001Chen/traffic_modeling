"""
Training Script for Traffic Occupancy Modeling

This script provides a complete training pipeline for traffic occupancy
prediction using DINOv2 + TimeSformer architecture.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
import json
import numpy as np
from datetime import datetime
import logging
from typing import Tuple

# Import our custom modules
from dinov2_feature_extractor import DINOv2SequenceProcessor
from timesformer_wrapper import LightweightTimeSformer
from prediction_head import RegressionHead, ClassificationHead, Conv1DPredictionHead
from traffic_dataset import TrafficDataModule


class TrafficModel(nn.Module):
    """
    Complete traffic modeling pipeline combining DINOv2, TimeSformer, and prediction head.
    
    Architecture:
    1. DINOv2 feature extractor for visual features
    2. TimeSformer for temporal modeling
    3. Prediction head for final output
    """
    
    def __init__(
        self,
        feature_dim: int = 384,
        sequence_length: int = 8,
        task_type: str = "regression",
        num_outputs: int = 2,
        timesformer_layers: int = 4,
        timesformer_heads: int = 8,
        freeze_backbone: bool = False,
        head_type: str = "mlp"  # "mlp", "conv1d", "attention"
    ):
        """
        Initialize traffic model.
        
        Args:
            feature_dim: Feature dimension from DINOv2
            sequence_length: Length of input sequence
            task_type: Task type ("regression" or "classification" or "anomaly")
            num_outputs: Number of outputs (predictions or classes)
            timesformer_layers: Number of TimeSformer layers
            timesformer_heads: Number of attention heads
            freeze_backbone: Whether to freeze DINOv2 backbone
            head_type: Type of prediction head
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.task_type = task_type
        self.num_outputs = num_outputs
        
        # DINOv2 feature extractor
        self.feature_extractor = DINOv2SequenceProcessor(
            feature_dim=feature_dim,
            freeze_backbone=freeze_backbone
        )
        
        # TimeSformer for temporal modeling
        self.timesformer = LightweightTimeSformer(
            feature_dim=feature_dim,
            sequence_length=sequence_length,
            num_layers=timesformer_layers,
            num_heads=timesformer_heads,
            output_mode="sequence"
        )
        
        # Prediction head
        if head_type == "mlp":
            if task_type == "regression":
                self.prediction_head = RegressionHead(
                    feature_dim=feature_dim,
                    num_predictions=num_outputs,
                    input_mode="sequence"
                )
            else:  # classification
                self.prediction_head = ClassificationHead(
                    feature_dim=feature_dim,
                    num_classes=num_outputs,
                    input_mode="sequence"
                )
        elif head_type == "conv1d":
            self.prediction_head = Conv1DPredictionHead(
                feature_dim=feature_dim,
                sequence_length=sequence_length,
                task_type=task_type,
                num_outputs=num_outputs
            )
        elif head_type == "attention":
            from prediction_head import AttentionPoolingHead
            self.prediction_head = AttentionPoolingHead(
                feature_dim=feature_dim,
                num_outputs=num_outputs
            )
        else:
            raise ValueError(f"Unsupported head type: {head_type}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_outputs)
        """
        # Extract visual features using DINOv2
        visual_features = self.feature_extractor(x)  # (batch_size, seq_len, feature_dim)
        
        # Apply temporal modeling using TimeSformer
        temporal_features = self.timesformer(visual_features)  # (batch_size, seq_len, feature_dim)
        
        # Generate final predictions
        predictions = self.prediction_head(temporal_features)  # (batch_size, num_outputs)
        
        return predictions


class Trainer:
    """
    Trainer class for traffic modeling.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        task_type: str = "regression",
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        save_dir: str = "./checkpoints"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            task_type: Task type ("regression" or "classification")
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.task_type = task_type
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min' if task_type == "regression" else 'max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Setup loss function
        if task_type == "regression":
            self.criterion = nn.MSELoss()
            self.metric_name = "MSE"
        else:  # classification or anomaly
            self.criterion = nn.CrossEntropyLoss()
            self.metric_name = "Accuracy"
        
        # Training history
        self.train_history = {
            'loss': [],
            'metric': []
        }
        self.val_history = {
            'loss': [],
            'metric': []
        }
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.save_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, average_metric)
        """
        self.model.train()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (sequences, targets) in enumerate(pbar):
            sequences = sequences.to(self.device).float()
            targets = targets.to(self.device).float()
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(sequences)
            
            # Compute loss
            if self.task_type == "regression" and targets.ndim == 1:
                targets = targets.unsqueeze(1)
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Compute metric
            if self.task_type == "regression":
                metric = torch.mean((predictions - targets) ** 2).item()
            else:  # classification or anomaly
                metric = (predictions.argmax(dim=1) == targets).float().mean().item()
            
            total_loss += loss.item()
            total_metric += metric
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Metric': f'{metric:.4f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_metric = total_metric / num_batches
        
        return avg_loss, avg_metric
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average_loss, average_metric)
        """
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for sequences, targets in tqdm(self.val_loader, desc="Validation"):
                sequences = sequences.to(self.device).float()
                targets = targets.to(self.device).float()
                
                # Forward pass
                predictions = self.model(sequences)
                
                # Compute loss
                if self.task_type == "regression" and targets.ndim == 1:
                    targets = targets.unsqueeze(1)
                loss = self.criterion(predictions, targets)
                
                # Compute metric
                if self.task_type == "regression":
                    metric = torch.mean((predictions - targets) ** 2).item()
                else:  # classification or anomaly
                    metric = (predictions.argmax(dim=1) == targets).float().mean().item()
                
                total_loss += loss.item()
                total_metric += metric
        
        avg_loss = total_loss / num_batches
        avg_metric = total_metric / num_batches
        
        return avg_loss, avg_metric
    
    def test(self) -> Tuple[float, float]:
        """
        Test the model.
        
        Returns:
            Tuple of (average_loss, average_metric)
        """
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = len(self.test_loader)
        
        with torch.no_grad():
            for sequences, targets in tqdm(self.test_loader, desc="Testing"):
                sequences = sequences.to(self.device).float()
                targets = targets.to(self.device).float()
                
                # Forward pass
                predictions = self.model(sequences)
                
                # Compute loss
                if self.task_type == "regression" and targets.ndim == 1:
                    targets = targets.unsqueeze(1)
                loss = self.criterion(predictions, targets)
                
                # Compute metric
                if self.task_type == "regression":
                    metric = torch.mean((predictions - targets) ** 2).item()
                else:  # classification or anomaly
                    metric = (predictions.argmax(dim=1) == targets).float().mean().item()
                
                total_loss += loss.item()
                total_metric += metric
        
        avg_loss = total_loss / num_batches
        avg_metric = total_metric / num_batches
        
        return avg_loss, avg_metric
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        
        return checkpoint['epoch']
    
    def train(self, num_epochs: int, save_freq: int = 10):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            save_freq: Frequency to save checkpoints
        """
        best_metric = float('inf') if self.task_type == "regression" else 0.0
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Task type: {self.task_type}")
        self.logger.info(f"Device: {self.device}")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_metric = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metric = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss if self.task_type == "regression" else -val_metric)
            
            # Update history
            self.train_history['loss'].append(train_loss)
            self.train_history['metric'].append(train_metric)
            self.val_history['loss'].append(val_loss)
            self.val_history['metric'].append(val_metric)
            
            # Log results
            self.logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train {self.metric_name}: {train_metric:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val {self.metric_name}: {val_metric:.4f}"
            )
            
            # Save checkpoint
            is_best = False
            if self.task_type == "regression":
                if val_loss < best_metric:
                    best_metric = val_loss
                    is_best = True
            else:  # classification
                if val_metric > best_metric:
                    best_metric = val_metric
                    is_best = True
            
            if epoch % save_freq == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        # Final test
        test_loss, test_metric = self.test()
        self.logger.info(f"Final Test - Loss: {test_loss:.4f}, {self.metric_name}: {test_metric:.4f}")
        
        # Save final model
        self.save_checkpoint(num_epochs, is_best=False)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Traffic Occupancy Model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--sequence_length", type=int, default=8, help="Sequence length")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--task_type", type=str, default="regression", choices=["regression", "classification", "anomaly"], help="Task type")
    parser.add_argument("--num_outputs", type=int, default=2, help="Number of outputs")
    parser.add_argument("--data_format", type=str, default="simulated", choices=["simulated", "pems"], help="Data format")
    
    # Model arguments
    parser.add_argument("--feature_dim", type=int, default=384, help="Feature dimension")
    parser.add_argument("--timesformer_layers", type=int, default=4, help="TimeSformer layers")
    parser.add_argument("--timesformer_heads", type=int, default=8, help="TimeSformer heads")
    parser.add_argument("--head_type", type=str, default="mlp", choices=["mlp", "conv1d", "attention"], help="Head type")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze DINOv2 backbone")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Save directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    # System arguments
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create data module
    data_module = TrafficDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
        task_type=args.task_type,
        num_workers=args.num_workers,
        data_format=args.data_format
    )
    
    # Setup data loaders
    train_loader, val_loader, test_loader = data_module.get_data_loaders()
    
    # Create model
    model = TrafficModel(
        feature_dim=args.feature_dim,
        sequence_length=args.sequence_length,
        task_type=args.task_type,
        num_outputs=args.num_outputs,
        timesformer_layers=args.timesformer_layers,
        timesformer_heads=args.timesformer_heads,
        freeze_backbone=args.freeze_backbone,
        head_type=args.head_type
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        task_type=args.task_type,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume) + 1
    
    # Train model
    trainer.train(num_epochs=args.num_epochs)
    
    print("Training completed!")


if __name__ == "__main__":
    main()

