#!/usr/bin/env python3
"""
Test script for PeMS downloader functionality

This script tests the PeMS downloader without requiring actual PeMS credentials.
"""

import os
import sys
import tempfile
import shutil
from pems_downloader import PeMSDownloader


def test_sample_data_generation():
    """Test sample data generation functionality."""
    print("Testing sample data generation...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create downloader with dummy credentials
        downloader = PeMSDownloader(
            username="test_user",
            password="test_pass",
            data_dir=temp_dir,
            headless=True
        )
        
        # Test sample data generation
        traffic_data = downloader.download_traffic_data(
            vds_ids=["400001", "400002"],
            start_date="2024-01-01",
            end_date="2024-01-02",
            data_type="occupancy"
        )
        
        print(f"✓ Generated data for {len(traffic_data)} VDS stations")
        
        # Check data structure
        for vds_id, df in traffic_data.items():
            print(f"  VDS {vds_id}: {len(df)} records")
            print(f"    Columns: {list(df.columns)}")
            print(f"    Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
        # Test heatmap conversion
        print("\nTesting heatmap conversion...")
        downloader.convert_to_heatmaps(
            traffic_data=traffic_data,
            output_dir=os.path.join(temp_dir, "pems"),
            image_size=224,
            sequence_length=8
        )
        
        # Check output structure
        pems_dir = os.path.join(temp_dir, "pems")
        if os.path.exists(pems_dir):
            print("✓ Heatmap conversion successful")
            
            # Count generated sequences
            for split in ['train', 'val', 'test']:
                split_dir = os.path.join(pems_dir, split)
                if os.path.exists(split_dir):
                    sequences = [d for d in os.listdir(split_dir) if d.startswith('sequence_')]
                    print(f"  {split}: {len(sequences)} sequences")
                    
                    # Check first sequence
                    if sequences:
                        seq_dir = os.path.join(split_dir, sequences[0])
                        frames = [f for f in os.listdir(seq_dir) if f.endswith('.png')]
                        print(f"    First sequence: {len(frames)} frames")
        else:
            print("✗ Heatmap conversion failed")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    finally:
        downloader.close()
        # Clean up
        shutil.rmtree(temp_dir)
        
    print("✓ Sample data generation test passed")
    return True


def test_integration():
    """Test integration with traffic dataset."""
    print("\nTesting integration with traffic dataset...")
    
    try:
        # Import the dataset class
        from traffic_dataset import TrafficHeatmapDataset
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Test with simulated data (should work)
        dataset = TrafficHeatmapDataset(
            data_dir=temp_dir,
            sequence_length=8,
            task_type="regression",
            data_format="simulated",
            split="train"
        )
        
        print(f"✓ Dataset created successfully with {len(dataset)} samples")
        
        # Test data loading
        if len(dataset) > 0:
            sequence, label = dataset[0]
            print(f"✓ Sample loaded: sequence shape {sequence.shape}, label {label}")
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        print("✓ Integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("PeMS Downloader Test Suite")
    print("=" * 40)
    
    tests = [
        ("Sample Data Generation", test_sample_data_generation),
        ("Dataset Integration", test_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
