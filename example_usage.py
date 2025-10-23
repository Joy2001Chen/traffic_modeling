"""
Example Usage of Traffic Modeling Pipeline

This script demonstrates how to use the traffic modeling components
for both training and inference.
"""

import torch
import os
from dinov2_feature_extractor import DINOv2SequenceProcessor
from timesformer_wrapper import LightweightTimeSformer
from prediction_head import RegressionHead, ClassificationHead
from traffic_dataset import TrafficDataModule
from train import TrafficModel, Trainer


def example_1_basic_usage():
    """Example 1: Basic component usage."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Component Usage")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create individual components
    feature_extractor = DINOv2SequenceProcessor(device=device)
    timesformer = LightweightTimeSformer(
        feature_dim=384,
        sequence_length=8,
        num_layers=4,
        num_heads=8,
        output_mode="sequence"
    ).to(device)
    
    regression_head = RegressionHead(
        feature_dim=384,
        num_predictions=2,
        input_mode="sequence"
    ).to(device)
    
    # Create dummy input: (batch_size=2, seq_len=8, channels=3, height=224, width=224)
    dummy_sequences = torch.randn(2, 8, 3, 224, 224).to(device)
    print(f"Input shape: {dummy_sequences.shape}")
    
    # Process through pipeline
    with torch.no_grad():
        # Extract visual features
        visual_features = feature_extractor(dummy_sequences)
        print(f"Visual features shape: {visual_features.shape}")
        
        # Apply temporal modeling
        temporal_features = timesformer(visual_features)
        print(f"Temporal features shape: {temporal_features.shape}")
        
        # Generate predictions
        predictions = regression_head(temporal_features)
        print(f"Predictions shape: {predictions.shape}")
        print(f"Sample predictions: {predictions[0].cpu().numpy()}")
    
    print("✓ Basic usage example completed!")


def example_2_complete_model():
    """Example 2: Using the complete model."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Complete Model Usage")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create complete model
    model = TrafficModel(
        feature_dim=384,
        sequence_length=8,
        task_type="regression",
        num_outputs=2,
        timesformer_layers=4,
        timesformer_heads=8,
        head_type="mlp"
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with dummy data
    dummy_input = torch.randn(4, 8, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Model output shape: {output.shape}")
        print(f"Sample outputs: {output[0].cpu().numpy()}")
    
    # Test classification model
    cls_model = TrafficModel(
        feature_dim=384,
        sequence_length=8,
        task_type="classification",
        num_outputs=3,
        head_type="conv1d"
    ).to(device)
    
    with torch.no_grad():
        cls_output = cls_model(dummy_input)
        print(f"Classification output shape: {cls_output.shape}")
        print(f"Sample class probabilities: {torch.softmax(cls_output[0], dim=0).cpu().numpy()}")
    
    print("✓ Complete model example completed!")


def example_3_dataset_usage():
    """Example 3: Dataset and data loading."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Dataset Usage")
    print("=" * 60)
    
    # Create temporary data directory
    data_dir = "./example_data"
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # Create data module
        data_module = TrafficDataModule(
            data_dir=data_dir,
            batch_size=4,
            sequence_length=8,
            task_type="regression",
            data_format="simulated"
        )
        
        # Get data loaders
        train_loader, val_loader, test_loader = data_module.get_data_loaders()
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Examine a batch
        for sequences, labels in train_loader:
            print(f"Batch sequences shape: {sequences.shape}")
            print(f"Batch labels shape: {labels.shape}")
            print(f"Sample labels: {labels.numpy()}")
            break
        
        print("✓ Dataset usage example completed!")
        
    except Exception as e:
        print(f"⚠ Dataset example failed: {e}")
    finally:
        # Clean up
        import shutil
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)


def example_4_training_setup():
    """Example 4: Training setup (without actual training)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Training Setup")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create temporary data directory
    data_dir = "./example_data"
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # Create data module
        data_module = TrafficDataModule(
            data_dir=data_dir,
            batch_size=2,
            sequence_length=8,
            task_type="regression",
            data_format="simulated"
        )
        
        train_loader, val_loader, test_loader = data_module.get_data_loaders()
        
        # Create model
        model = TrafficModel(
            feature_dim=384,
            sequence_length=8,
            task_type="regression",
            num_outputs=2,
            timesformer_layers=2,  # Reduced for example
            timesformer_heads=4,
            head_type="mlp"
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            task_type="regression",
            device=device,
            learning_rate=1e-4,
            save_dir="./example_checkpoints"
        )
        
        print("✓ Training setup completed!")
        print("To start training, call: trainer.train(num_epochs=10)")
        
        # Test one training step
        print("\nTesting one training step...")
        model.train()
        
        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # Forward pass
            trainer.optimizer.zero_grad()
            predictions = model(sequences)
            loss = trainer.criterion(predictions, targets)
            
            print(f"Training step - Loss: {loss.item():.4f}")
            print(f"Predictions: {predictions[0].cpu().detach().numpy()}")
            print(f"Targets: {targets[0].cpu().numpy()}")
            break
        
        print("✓ Training step test completed!")
        
    except Exception as e:
        print(f"⚠ Training setup failed: {e}")
    finally:
        # Clean up
        import shutil
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        if os.path.exists("./example_checkpoints"):
            shutil.rmtree("./example_checkpoints")


def example_5_model_comparison():
    """Example 5: Compare different model configurations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Model Configuration Comparison")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_input = torch.randn(2, 8, 3, 224, 224).to(device)
    
    # Different model configurations
    configs = [
        {
            "name": "Small Model",
            "timesformer_layers": 2,
            "timesformer_heads": 4,
            "head_type": "mlp"
        },
        {
            "name": "Medium Model", 
            "timesformer_layers": 4,
            "timesformer_heads": 8,
            "head_type": "conv1d"
        },
        {
            "name": "Large Model",
            "timesformer_layers": 6,
            "timesformer_heads": 12,
            "head_type": "attention"
        }
    ]
    
    for config in configs:
        print(f"\n{config['name']}:")
        
        model = TrafficModel(
            feature_dim=384,
            sequence_length=8,
            task_type="regression",
            num_outputs=2,
            timesformer_layers=config["timesformer_layers"],
            timesformer_heads=config["timesformer_heads"],
            head_type=config["head_type"]
        ).to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count:,}")
        
        with torch.no_grad():
            output = model(dummy_input)
            print(f"  Output shape: {output.shape}")
            print(f"  Sample output: {output[0].cpu().numpy()}")
    
    print("✓ Model comparison completed!")


def main():
    """Run all examples."""
    print("TRAFFIC MODELING PIPELINE - USAGE EXAMPLES")
    print("=" * 60)
    
    try:
        example_1_basic_usage()
        example_2_complete_model()
        example_3_dataset_usage()
        example_4_training_setup()
        example_5_model_comparison()
        
        print("\n" + "=" * 60)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("The traffic modeling pipeline is ready for use.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ EXAMPLE FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
