"""
Traffic Dataset for Occupancy Heatmap Sequences

This module provides dataset classes for loading and preprocessing
traffic occupancy heatmap sequences from PeMS or simulated data.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import glob
from typing import List, Tuple, Optional, Dict, Any
import random


class TrafficHeatmapDataset(Dataset):
    """
    Dataset for traffic occupancy heatmap sequences.
    
    Supports both PeMS data format and simulated data.
    Each sample consists of T consecutive frames of occupancy heatmaps.
    """
    
    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 8,
        image_size: int = 224,
        task_type: str = "regression",  # "regression" or "classification"
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
        data_format: str = "pems",  # "pems" or "simulated"
        split: str = "train",  # "train", "val", "test"
        overlap_ratio: float = 0.5  # Overlap between consecutive sequences
    ):
        """
        Initialize traffic heatmap dataset.
        
        Args:
            data_dir: Directory containing the data
            sequence_length: Number of frames in each sequence (T)
            image_size: Size to resize images to
            task_type: Type of task ("regression" or "classification")
            transform: Image transformations
            target_transform: Target transformations
            data_format: Data format ("pems" or "simulated")
            split: Data split ("train", "val", "test")
            overlap_ratio: Overlap ratio between consecutive sequences
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.task_type = task_type
        self.data_format = data_format
        self.split = split
        self.overlap_ratio = overlap_ratio
        
        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
            
        self.target_transform = target_transform
        
        # Load data paths and labels
        self.data_paths, self.labels = self._load_data()
        
        print(f"Loaded {len(self.data_paths)} sequences for {split} split")
        
    def _get_default_transforms(self) -> transforms.Compose:
        """Get default image transformations."""
        if self.split == "train":
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _load_data(self) -> Tuple[List[str], List[Any]]:
        """Load data paths and labels."""
        if self.data_format == "simulated":
            return self._load_simulated_data()
        elif self.data_format == "pems":
            return self._load_pems_data()
        else:
            raise ValueError(f"Unsupported data format: {self.data_format}")
    
    def _load_simulated_data(self) -> Tuple[List[str], List[Any]]:
        """Load simulated data paths and labels."""
        data_paths = []
        labels = []
        
        # Create simulated data if it doesn't exist
        simulated_dir = os.path.join(self.data_dir, "simulated", self.split)
        os.makedirs(simulated_dir, exist_ok=True)
        
        # Generate simulated sequences
        num_sequences = 100 if self.split == "train" else 20
        for seq_idx in range(num_sequences):
            seq_dir = os.path.join(simulated_dir, f"sequence_{seq_idx:04d}")
            os.makedirs(seq_dir, exist_ok=True)
            
            # Generate sequence of heatmap images
            seq_paths = []
            for frame_idx in range(self.sequence_length + 2):  # Extra frames for targets
                # Create simulated heatmap
                heatmap = self._generate_simulated_heatmap(frame_idx, seq_idx)
                
                # Save as PNG
                img_path = os.path.join(seq_dir, f"frame_{frame_idx:03d}.png")
                Image.fromarray(heatmap).save(img_path)
                seq_paths.append(img_path)
            
            # Store sequence paths (excluding target frames)
            data_paths.append(seq_paths[:self.sequence_length])
            
            # Generate labels based on task type
            if self.task_type == "regression":
                # Predict future occupancy (average of last 2 frames)
                target_frames = seq_paths[self.sequence_length:self.sequence_length+2]
                label = self._compute_occupancy_label(target_frames)
            elif self.task_type == "classification":
                # Classify current traffic state
                current_frames = seq_paths[self.sequence_length-2:self.sequence_length]
                label = self._compute_traffic_class(current_frames)
            elif self.task_type == "anomaly":
                # Detect temporal anomalies in the current window
                current_frames = seq_paths[:self.sequence_length]
                label = self._compute_anomaly_label(current_frames)
            else:
                raise ValueError(f"Unsupported task_type: {self.task_type}")
            
            labels.append(label)
        
        return data_paths, labels
    
    def _load_pems_data(self) -> Tuple[List[str], List[Any]]:
        """Load PeMS data paths and labels."""
        data_paths = []
        labels = []
        
        # Look for PeMS data structure
        pems_dir = os.path.join(self.data_dir, "pems", self.split)
        
        if not os.path.exists(pems_dir):
            print(f"PeMS data directory not found: {pems_dir}")
            print("Attempting to download PeMS data...")
            
            # Try to download PeMS data
            if self._download_pems_data():
                print("PeMS data downloaded successfully!")
            else:
                print("Failed to download PeMS data. Falling back to simulated data...")
                return self._load_simulated_data()
        
        # Find all sequence directories
        seq_dirs = sorted(glob.glob(os.path.join(pems_dir, "sequence_*")))
        
        for seq_dir in seq_dirs:
            # Get all frame files in sequence
            frame_files = sorted(glob.glob(os.path.join(seq_dir, "*.png")))
            
            if len(frame_files) < self.sequence_length + 2:
                continue
            
            # Create overlapping sequences
            step_size = max(1, int(self.sequence_length * (1 - self.overlap_ratio)))
            
            for start_idx in range(0, len(frame_files) - self.sequence_length - 1, step_size):
                seq_frames = frame_files[start_idx:start_idx + self.sequence_length]
                data_paths.append(seq_frames)
                
                # Generate labels
                if self.task_type == "regression":
                    target_frames = frame_files[start_idx + self.sequence_length:start_idx + self.sequence_length + 2]
                    label = self._compute_occupancy_label(target_frames)
                elif self.task_type == "classification":
                    current_frames = frame_files[start_idx + self.sequence_length - 2:start_idx + self.sequence_length]
                    label = self._compute_traffic_class(current_frames)
                elif self.task_type == "anomaly":
                    current_frames = frame_files[start_idx:start_idx + self.sequence_length]
                    label = self._compute_anomaly_label(current_frames)
                else:
                    raise ValueError(f"Unsupported task_type: {self.task_type}")
                
                labels.append(label)
        
        return data_paths, labels
    
    def _download_pems_data(self) -> bool:
        """
        Download PeMS data using the PeMS downloader.
        
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Import here to avoid circular imports
            from pems_downloader import PeMSDownloader
            
            # Check for credentials in environment variables or config file
            username = os.getenv('PEMS_USERNAME')
            password = os.getenv('PEMS_PASSWORD')
            
            if not username or not password:
                print("PeMS credentials not found. Please set PEMS_USERNAME and PEMS_PASSWORD environment variables.")
                print("Alternatively, you can manually download PeMS data and place it in the correct directory structure.")
                return False
            
            # Create downloader
            downloader = PeMSDownloader(
                username=username,
                password=password,
                data_dir=self.data_dir,
                headless=True
            )
            
            try:
                # Download data for a week (adjust as needed)
                from datetime import datetime, timedelta
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)
                
                traffic_data = downloader.download_traffic_data(
                    vds_ids=["400001", "400002", "400003", "400004", "400005"],
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    data_type="occupancy"
                )
                
                # Convert to heatmaps
                downloader.convert_to_heatmaps(
                    traffic_data=traffic_data,
                    output_dir=os.path.join(self.data_dir, "pems"),
                    image_size=self.image_size,
                    sequence_length=self.sequence_length
                )
                
                return True
                
            finally:
                downloader.close()
                
        except ImportError:
            print("PeMS downloader not available. Please install required dependencies:")
            print("pip install selenium pandas matplotlib webdriver-manager")
            return False
        except Exception as e:
            print(f"Failed to download PeMS data: {e}")
            return False
    
    def _generate_simulated_heatmap(self, frame_idx: int, seq_idx: int) -> np.ndarray:
        """Generate a simulated occupancy heatmap."""
        # Create base heatmap
        heatmap = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Add some traffic patterns
        center_x, center_y = 112, 112
        
        # Simulate traffic flow
        for i in range(10):
            angle = (frame_idx * 0.1 + i * 0.3) % (2 * np.pi)
            radius = 30 + i * 5
            
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            
            if 0 <= x < 224 and 0 <= y < 224:
                # Add occupancy (red channel for traffic)
                intensity = int(255 * (0.3 + 0.7 * np.random.random()))
                heatmap[y-2:y+3, x-2:x+3, 0] = intensity
        
        # Add some noise
        noise = np.random.randint(0, 50, heatmap.shape, dtype=np.uint8)
        heatmap = np.clip(heatmap.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return heatmap
    
    def _compute_occupancy_label(self, target_frames: List[str]) -> float:
        """Compute occupancy label from target frames."""
        total_occupancy = 0.0
        
        for frame_path in target_frames:
            if os.path.exists(frame_path):
                img = Image.open(frame_path).convert('RGB')
                img_array = np.array(img)
                # Compute occupancy as average red channel intensity
                occupancy = np.mean(img_array[:, :, 0]) / 255.0
                total_occupancy += occupancy
        
        return total_occupancy / len(target_frames)
    
    def _compute_traffic_class(self, current_frames: List[str]) -> int:
        """Compute traffic class from current frames."""
        total_occupancy = 0.0
        
        for frame_path in current_frames:
            if os.path.exists(frame_path):
                img = Image.open(frame_path).convert('RGB')
                img_array = np.array(img)
                occupancy = np.mean(img_array[:, :, 0]) / 255.0
                total_occupancy += occupancy
        
        avg_occupancy = total_occupancy / len(current_frames)
        
        # Classify: 0=smooth, 1=medium, 2=congested
        if avg_occupancy < 0.3:
            return 0
        elif avg_occupancy < 0.7:
            return 1
        else:
            return 2

    def _compute_anomaly_label(self, frames: List[str], rel_threshold: float = 0.6, z_threshold: float = 3.0) -> int:
        """Compute anomaly label (0/1) from a sequence of frames.

        Heuristics:
        - Compute per-frame occupancy (mean of red channel / 255).
        - If any relative jump between consecutive frames exceeds rel_threshold, flag anomaly.
        - Additionally, z-score against sequence mean/std for robustness.
        """
        occupancies: List[float] = []
        for frame_path in frames:
            if os.path.exists(frame_path):
                img = Image.open(frame_path).convert('RGB')
                img_array = np.array(img)
                occ = float(np.mean(img_array[:, :, 0]) / 255.0)
                occupancies.append(occ)
        if len(occupancies) < 3:
            return 0
        # Relative jump detection
        for i in range(1, len(occupancies)):
            prev = max(occupancies[i-1], 1e-6)
            rel = abs(occupancies[i] - occupancies[i-1]) / prev
            if rel > rel_threshold:
                return 1
        # Z-score detection
        mean = float(np.mean(occupancies))
        std = float(np.std(occupancies) + 1e-6)
        for v in occupancies:
            z = abs(v - mean) / std
            if z > z_threshold:
                return 1
        return 0
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        """Get a sample from the dataset."""
        frame_paths = self.data_paths[idx]
        label = self.labels[idx]
        
        # Load and transform images
        images = []
        for frame_path in frame_paths:
            try:
                img = Image.open(frame_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {frame_path}: {e}")
                # Create dummy image if loading fails
                dummy_img = torch.zeros(3, self.image_size, self.image_size)
                images.append(dummy_img)
        
        # Stack images into sequence tensor
        sequence = torch.stack(images, dim=0)  # (T, C, H, W)
        
        # Apply target transform if specified
        if self.target_transform:
            label = self.target_transform(label)
        
        # Ensure label dtype matches task
        if self.task_type == "regression":
            if isinstance(label, (int, float)):
                label = torch.tensor(label, dtype=torch.float32)
            elif isinstance(label, torch.Tensor):
                label = label.float()
        else:  # classification or anomaly -> integer class id
            if isinstance(label, (int, np.integer)):
                label = torch.tensor(int(label), dtype=torch.long)
            elif isinstance(label, torch.Tensor):
                label = label.long()
            else:
                label = torch.tensor(int(label), dtype=torch.long)
        
        return sequence, label


class TrafficDataModule:
    """
    Data module for managing traffic datasets and data loaders.
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 8,
        sequence_length: int = 8,
        image_size: int = 224,
        task_type: str = "regression",
        num_workers: int = 4,
        data_format: str = "simulated",
        overlap_ratio: float = 0.5
    ):
        """
        Initialize traffic data module.
        
        Args:
            data_dir: Directory containing the data
            batch_size: Batch size for data loaders
            sequence_length: Number of frames in each sequence
            image_size: Size to resize images to
            task_type: Type of task ("regression" or "classification")
            num_workers: Number of worker processes for data loading
            data_format: Data format ("pems" or "simulated")
            overlap_ratio: Overlap ratio between consecutive sequences
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.task_type = task_type
        self.num_workers = num_workers
        self.data_format = data_format
        self.overlap_ratio = overlap_ratio
        
        # Create datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Create data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def setup(self):
        """Setup datasets and data loaders."""
        # Create datasets
        self.train_dataset = TrafficHeatmapDataset(
            data_dir=self.data_dir,
            sequence_length=self.sequence_length,
            image_size=self.image_size,
            task_type=self.task_type,
            data_format=self.data_format,
            split="train",
            overlap_ratio=self.overlap_ratio
        )
        
        self.val_dataset = TrafficHeatmapDataset(
            data_dir=self.data_dir,
            sequence_length=self.sequence_length,
            image_size=self.image_size,
            task_type=self.task_type,
            data_format=self.data_format,
            split="val",
            overlap_ratio=self.overlap_ratio
        )
        
        self.test_dataset = TrafficHeatmapDataset(
            data_dir=self.data_dir,
            sequence_length=self.sequence_length,
            image_size=self.image_size,
            task_type=self.task_type,
            data_format=self.data_format,
            split="test",
            overlap_ratio=self.overlap_ratio
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, validation, and test data loaders."""
        if self.train_loader is None:
            self.setup()
        
        return self.train_loader, self.val_loader, self.test_loader


if __name__ == "__main__":
    # Test the dataset
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create data module
    data_module = TrafficDataModule(
        data_dir=data_dir,
        batch_size=4,
        sequence_length=8,
        task_type="regression",
        data_format="simulated"
    )
    
    # Setup and get data loaders
    train_loader, val_loader, test_loader = data_module.get_data_loaders()
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    for batch_idx, (sequences, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Sequence shape: {sequences.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels}")
        
        if batch_idx >= 2:  # Test only first few batches
            break
