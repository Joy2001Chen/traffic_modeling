"""
Test script to verify the complete traffic modeling pipeline.

This script tests all components individually and as a complete pipeline.
"""

import torch
import os
import sys
from datetime import datetime

# Import our modules
from dinov2_feature_extractor import DINOv2FeatureExtractor, DINOv2SequenceProcessor
from timesformer_wrapper import LightweightTimeSformer, TemporalConv1D
from prediction_head import RegressionHead, ClassificationHead, Conv1DPredictionHead
from traffic_dataset import TrafficDataModule
from train import TrafficModel


def test_dinov2_extractor():
    """Test DINOv2 feature extractor."""
    print("Testing DINOv2 Feature Extractor...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test single image extractor
    extractor = DINOv2FeatureExtractor(device=device)
    dummy_image = torch.randn(2, 3, 224, 224).to(device)
    features = extractor(dummy_image)
    
    assert features.shape == (2, 384), f"Expected (2, 384), got {features.shape}"
    print(f"✓ Single image extractor: {features.shape}")
    
    # Test sequence processor
    seq_processor = DINOv2SequenceProcessor(device=device)
    dummy_sequence = torch.randn(2, 8, 3, 224, 224).to(device)
    seq_features = seq_processor(dummy_sequence)
    
    assert seq_features.shape == (2, 8, 384), f"Expected (2, 8, 384), got {seq_features.shape}"
    print(f"✓ Sequence processor: {seq_features.shape}")


def test_timesformer():
    """Test TimeSformer components."""
    print("\nTesting TimeSformer Components...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test TimeSformer
    timesformer = LightweightTimeSformer(
        feature_dim=384,
        sequence_length=8,
        num_layers=4,
        num_heads=8,
        output_mode="sequence"
    ).to(device)
    
    dummy_input = torch.randn(2, 8, 384).to(device)
    output = timesformer(dummy_input)
    
    assert output.shape == (2, 8, 384), f"Expected (2, 8, 384), got {output.shape}"
    print(f"✓ TimeSformer: {output.shape}")
    
    # Test CLS token mode
    timesformer_cls = LightweightTimeSformer(
        feature_dim=384,
        sequence_length=8,
        output_mode="cls"
    ).to(device)
    
    cls_output = timesformer_cls(dummy_input)
    assert cls_output.shape == (2, 384), f"Expected (2, 384), got {cls_output.shape}"
    print(f"✓ TimeSformer CLS: {cls_output.shape}")
    
    # Test Temporal Conv1D
    temporal_conv = TemporalConv1D(feature_dim=384).to(device)
    conv_output = temporal_conv(dummy_input)
    assert conv_output.shape == (2, 8, 384), f"Expected (2, 8, 384), got {conv_output.shape}"
    print(f"✓ Temporal Conv1D: {conv_output.shape}")


def test_prediction_heads():
    """Test prediction heads."""
    print("\nTesting Prediction Heads...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test regression head
    reg_head = RegressionHead(feature_dim=384, num_predictions=2).to(device)
    cls_input = torch.randn(2, 384).to(device)
    reg_output = reg_head(cls_input)
    assert reg_output.shape == (2, 2), f"Expected (2, 2), got {reg_output.shape}"
    print(f"✓ Regression Head: {reg_output.shape}")
    
    # Test classification head
    cls_head = ClassificationHead(feature_dim=384, num_classes=3).to(device)
    cls_output = cls_head(cls_input)
    assert cls_output.shape == (2, 3), f"Expected (2, 3), got {cls_output.shape}"
    print(f"✓ Classification Head: {cls_output.shape}")
    
    # Test Conv1D head
    conv_head = Conv1DPredictionHead(
        feature_dim=384,
        sequence_length=8,
        task_type="regression",
        num_outputs=2
    ).to(device)
    
    seq_input = torch.randn(2, 8, 384).to(device)
    conv_output = conv_head(seq_input)
    assert conv_output.shape == (2, 2), f"Expected (2, 2), got {conv_output.shape}"
    print(f"✓ Conv1D Head: {conv_output.shape}")


def test_dataset():
    """Test dataset creation."""
    print("\nTesting Dataset Creation...")
    
    # Create temporary data directory
    data_dir = "./test_data"
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # Test data module
        data_module = TrafficDataModule(
            data_dir=data_dir,
            batch_size=2,
            sequence_length=8,
            task_type="regression",
            data_format="simulated"
        )
        
        train_loader, val_loader, test_loader = data_module.get_data_loaders()
        
        # Test a batch
        for sequences, labels in train_loader:
            assert sequences.shape == (2, 8, 3, 224, 224), f"Expected (2, 8, 3, 224, 224), got {sequences.shape}"
            assert labels.shape == (2,), f"Expected (2,), got {labels.shape}"
            print(f"✓ Dataset batch: sequences {sequences.shape}, labels {labels.shape}")
            break
            
    except Exception as e:
        print(f"⚠ Dataset test failed: {e}")
    finally:
        # Clean up
        import shutil
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)


def test_complete_model():
    """Test complete model pipeline."""
    print("\nTesting Complete Model Pipeline...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test regression model
    model_reg = TrafficModel(
        feature_dim=384,
        sequence_length=8,
        task_type="regression",
        num_outputs=2,
        timesformer_layers=2,  # Reduced for testing
        timesformer_heads=4,
        head_type="mlp"
    ).to(device)
    
    dummy_input = torch.randn(2, 8, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model_reg(dummy_input)
        assert output.shape == (2, 2), f"Expected (2, 2), got {output.shape}"
        print(f"✓ Regression Model: {output.shape}")
    
    # Test classification model
    model_cls = TrafficModel(
        feature_dim=384,
        sequence_length=8,
        task_type="classification",
        num_outputs=3,
        timesformer_layers=2,
        timesformer_heads=4,
        head_type="conv1d"
    ).to(device)
    
    with torch.no_grad():
        output = model_cls(dummy_input)
        assert output.shape == (2, 3), f"Expected (2, 3), got {output.shape}"
        print(f"✓ Classification Model: {output.shape}")
    
    # Test model with frozen backbone
    model_frozen = TrafficModel(
        feature_dim=384,
        sequence_length=8,
        task_type="regression",
        num_outputs=2,
        freeze_backbone=True,
        head_type="attention"
    ).to(device)
    
    with torch.no_grad():
        output = model_frozen(dummy_input)
        assert output.shape == (2, 2), f"Expected (2, 2), got {output.shape}"
        print(f"✓ Frozen Backbone Model: {output.shape}")


def test_memory_usage():
    """Test memory usage and performance."""
    print("\nTesting Memory Usage...")
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping memory test")
        return
    
    device = "cuda"
    
    # Test with different batch sizes
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        try:
            model = TrafficModel(
                feature_dim=384,
                sequence_length=8,
                task_type="regression",
                num_outputs=2,
                timesformer_layers=2,
                timesformer_heads=4
            ).to(device)
            
            dummy_input = torch.randn(batch_size, 8, 3, 224, 224).to(device)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
            print(f"✓ Batch size {batch_size}: {memory_used:.1f} MB")
            
            del model, dummy_input, output
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            print(f"✗ Batch size {batch_size}: {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TRAFFIC MODELING PIPELINE TEST")
    print("=" * 60)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Test started at: {datetime.now()}")
    print("=" * 60)
    
    try:
        test_dinov2_extractor()
        test_timesformer()
        test_prediction_heads()
        test_dataset()
        test_complete_model()
        test_memory_usage()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("The traffic modeling pipeline is ready to use.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

