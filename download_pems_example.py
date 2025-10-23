#!/usr/bin/env python3
"""
Example script for downloading PeMS data

This script demonstrates how to use the PeMS downloader to automatically
download and process traffic data.
"""

import os
import sys
from pems_downloader import PeMSDownloader


def main():
    """Example usage of PeMS downloader."""
    
    # Check if credentials are provided
    username = os.getenv('PEMS_USERNAME')
    password = os.getenv('PEMS_PASSWORD')
    
    if not username or not password:
        print("Please set your PeMS credentials:")
        print("export PEMS_USERNAME='your_username'")
        print("export PEMS_PASSWORD='your_password'")
        print("\nOr run the script with credentials:")
        print("PEMS_USERNAME='your_username' PEMS_PASSWORD='your_password' python download_pems_example.py")
        return
    
    print("Starting PeMS data download...")
    print(f"Username: {username}")
    print("Password: [HIDDEN]")
    
    # Create downloader
    downloader = PeMSDownloader(
        username=username,
        password=password,
        data_dir="./data",
        headless=True,  # Set to False to see browser
        download_delay=2.0
    )
    
    try:
        # Download traffic data
        print("\nDownloading traffic data...")
        traffic_data = downloader.download_traffic_data(
            vds_ids=["400001", "400002", "400003"],
            start_date="2024-01-01",
            end_date="2024-01-07",
            data_type="occupancy"
        )
        
        print(f"Downloaded data for {len(traffic_data)} VDS stations")
        
        # Convert to heatmap sequences
        print("\nConverting to heatmap sequences...")
        downloader.convert_to_heatmaps(
            traffic_data=traffic_data,
            output_dir="./data/pems",
            image_size=224,
            sequence_length=8
        )
        
        print("\nPeMS data download completed successfully!")
        print("You can now run training with:")
        print("python train.py --data_format pems")
        
    except Exception as e:
        print(f"Error during download: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your PeMS credentials")
        print("2. Ensure you have internet connection")
        print("3. Install required dependencies: pip install selenium pandas matplotlib webdriver-manager")
        print("4. Try running with headless=False to see browser behavior")
        
    finally:
        downloader.close()


if __name__ == "__main__":
    main()
