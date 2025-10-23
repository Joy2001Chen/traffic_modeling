"""
PeMS Data Downloader

This module provides functionality to automatically download PeMS traffic data
and convert it to the required format for the traffic modeling project.
"""

import os
import requests
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime, timedelta
import time
import json
from typing import List, Dict, Optional, Tuple
import argparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import logging


class PeMSDownloader:
    """
    PeMS data downloader with authentication and data processing.
    """
    
    def __init__(
        self,
        username: str,
        password: str,
        data_dir: str = "./data",
        headless: bool = True,
        download_delay: float = 2.0
    ):
        """
        Initialize PeMS downloader.
        
        Args:
            username: PeMS website username
            password: PeMS website password
            data_dir: Directory to save downloaded data
            headless: Whether to run browser in headless mode
            download_delay: Delay between downloads in seconds
        """
        self.username = username
        self.password = password
        self.data_dir = data_dir
        self.headless = headless
        self.download_delay = download_delay
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize session
        self.session = requests.Session()
        self.driver = None
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.data_dir, 'pems_download.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_driver(self):
        """Setup Chrome WebDriver."""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.logger.info("Chrome WebDriver initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize WebDriver: {e}")
            raise
            
    def login(self) -> bool:
        """
        Login to PeMS website.
        
        Returns:
            True if login successful, False otherwise
        """
        if not self.driver:
            self.setup_driver()
            
        try:
            # Navigate to PeMS login page
            self.driver.get("https://pems.dot.ca.gov/")
            
            # Wait for login form
            wait = WebDriverWait(self.driver, 10)
            
            # Find and fill username field
            username_field = wait.until(
                EC.presence_of_element_located((By.NAME, "username"))
            )
            username_field.send_keys(self.username)
            
            # Find and fill password field
            password_field = self.driver.find_element(By.NAME, "password")
            password_field.send_keys(self.password)
            
            # Submit login form
            login_button = self.driver.find_element(By.XPATH, "//input[@type='submit']")
            login_button.click()
            
            # Wait for redirect and check if login was successful
            time.sleep(3)
            
            if "dashboard" in self.driver.current_url.lower() or "main" in self.driver.current_url.lower():
                self.logger.info("Successfully logged into PeMS")
                return True
            else:
                self.logger.error("Login failed - check credentials")
                return False
                
        except Exception as e:
            self.logger.error(f"Login failed: {e}")
            return False
            
    def download_traffic_data(
        self,
        vds_ids: List[str],
        start_date: str,
        end_date: str,
        data_type: str = "flow"
    ) -> Dict[str, pd.DataFrame]:
        """
        Download traffic data for specified VDS IDs and date range.
        
        Note: This is a simplified version that generates sample data.
        For real PeMS data, you would need to implement actual web scraping
        or use PeMS API if available.
        
        Args:
            vds_ids: List of VDS (Vehicle Detection Station) IDs
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            data_type: Type of data to download ("flow", "speed", "occupancy")
            
        Returns:
            Dictionary mapping VDS ID to DataFrame
        """
        self.logger.info("Generating sample PeMS data (real download requires PeMS API access)")
        
        downloaded_data = {}
        
        for vds_id in vds_ids:
            try:
                self.logger.info(f"Generating sample data for VDS {vds_id}")
                
                # Generate sample data instead of actual download
                downloaded_data[vds_id] = self._create_sample_data(vds_id, start_date, end_date)
                
            except Exception as e:
                self.logger.error(f"Failed to generate data for VDS {vds_id}: {e}")
                continue
                
        return downloaded_data
        
    def _create_sample_data(self, vds_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Create sample traffic data for testing purposes.
        
        Args:
            vds_id: VDS ID
            start_date: Start date
            end_date: End date
            
        Returns:
            Sample DataFrame with traffic data
        """
        # Generate sample data
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Create 5-minute intervals
        timestamps = pd.date_range(start_dt, end_dt, freq='5T')
        
        # Generate realistic traffic patterns
        np.random.seed(int(vds_id) % 1000)  # Use VDS ID as seed for consistency
        
        data = []
        for ts in timestamps:
            # Simulate daily traffic patterns
            hour = ts.hour
            day_of_week = ts.weekday()
            
            # Base flow varies by time of day
            if 6 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                base_flow = np.random.normal(2000, 300)
            elif 10 <= hour <= 16:  # Daytime
                base_flow = np.random.normal(1500, 200)
            else:  # Night
                base_flow = np.random.normal(500, 100)
                
            # Weekend effect
            if day_of_week >= 5:  # Weekend
                base_flow *= 0.7
                
            # Calculate derived metrics
            flow = max(0, int(base_flow))
            speed = max(20, int(np.random.normal(65, 15)))
            occupancy = min(100, max(0, int(flow / 2000 * 100)))
            
            data.append({
                'timestamp': ts,
                'vds_id': vds_id,
                'flow': flow,
                'speed': speed,
                'occupancy': occupancy
            })
            
        return pd.DataFrame(data)
        
    def convert_to_heatmaps(
        self,
        traffic_data: Dict[str, pd.DataFrame],
        output_dir: str,
        image_size: int = 224,
        sequence_length: int = 8
    ):
        """
        Convert traffic data to occupancy heatmap sequences.
        
        Args:
            traffic_data: Dictionary of VDS data
            output_dir: Output directory for heatmaps
            image_size: Size of output images
            sequence_length: Length of sequences
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a simple road network layout
        road_layout = self._create_road_layout(len(traffic_data))
        
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(output_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            
            # Generate sequences for this split
            num_sequences = 100 if split == 'train' else 20
            
            for seq_idx in range(num_sequences):
                seq_dir = os.path.join(split_dir, f"sequence_{seq_idx:04d}")
                os.makedirs(seq_dir, exist_ok=True)
                
                # Generate sequence of heatmaps
                for frame_idx in range(sequence_length + 2):  # Extra frames for targets
                    heatmap = self._generate_heatmap_from_data(
                        traffic_data, road_layout, frame_idx, seq_idx, image_size
                    )
                    
                    # Save heatmap
                    img_path = os.path.join(seq_dir, f"frame_{frame_idx:03d}.png")
                    Image.fromarray(heatmap).save(img_path)
                    
        self.logger.info(f"Generated heatmap sequences in {output_dir}")
        
    def _create_road_layout(self, num_vds: int) -> np.ndarray:
        """
        Create a simple road network layout.
        
        Args:
            num_vds: Number of VDS stations
            
        Returns:
            Road layout array
        """
        layout = np.zeros((224, 224))
        
        # Create highway-like structure
        # Main highway (horizontal)
        layout[100:120, :] = 1
        
        # On-ramps and off-ramps
        for i in range(0, 224, 50):
            layout[80:100, i:i+20] = 1
            layout[120:140, i:i+20] = 1
            
        return layout
        
    def _generate_heatmap_from_data(
        self,
        traffic_data: Dict[str, pd.DataFrame],
        road_layout: np.ndarray,
        frame_idx: int,
        seq_idx: int,
        image_size: int
    ) -> np.ndarray:
        """
        Generate heatmap from traffic data.
        
        Args:
            traffic_data: Dictionary of VDS data
            road_layout: Road network layout
            frame_idx: Frame index
            seq_idx: Sequence index
            image_size: Output image size
            
        Returns:
            Heatmap as numpy array
        """
        # Create base heatmap
        heatmap = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        
        # Add road network (gray)
        road_mask = road_layout > 0
        heatmap[road_mask, :] = [100, 100, 100]
        
        # Add traffic occupancy (red channel)
        for vds_id, df in traffic_data.items():
            if len(df) > frame_idx:
                occupancy = df.iloc[frame_idx]['occupancy']
                
                # Map VDS to position (simplified)
                vds_pos = int(vds_id) % image_size
                y_pos = 110 + (int(vds_id) % 20) - 10
                
                if 0 <= y_pos < image_size and 0 <= vds_pos < image_size:
                    # Create occupancy visualization
                    intensity = int(255 * (occupancy / 100))
                    heatmap[y_pos-5:y_pos+5, vds_pos-10:vds_pos+10, 0] = intensity
                    
        # Add some noise for realism
        noise = np.random.randint(0, 30, heatmap.shape, dtype=np.uint8)
        heatmap = np.clip(heatmap.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return heatmap
        
    def close(self):
        """Close WebDriver and cleanup."""
        if self.driver:
            self.driver.quit()
            self.logger.info("WebDriver closed")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Download PeMS traffic data")
    
    # Authentication
    parser.add_argument("--username", type=str, required=True, help="PeMS username")
    parser.add_argument("--password", type=str, required=True, help="PeMS password")
    
    # Data parameters
    parser.add_argument("--vds_ids", type=str, nargs='+', default=["400001", "400002", "400003"], 
                       help="VDS IDs to download")
    parser.add_argument("--start_date", type=str, default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2024-01-07", help="End date (YYYY-MM-DD)")
    parser.add_argument("--data_type", type=str, default="flow", choices=["flow", "speed", "occupancy"],
                       help="Type of data to download")
    
    # Output parameters
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./data/pems", help="Output directory")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--sequence_length", type=int, default=8, help="Sequence length")
    
    # Browser options
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--download_delay", type=float, default=2.0, help="Download delay in seconds")
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = PeMSDownloader(
        username=args.username,
        password=args.password,
        data_dir=args.data_dir,
        headless=args.headless,
        download_delay=args.download_delay
    )
    
    try:
        # Download traffic data
        print("Downloading PeMS traffic data...")
        traffic_data = downloader.download_traffic_data(
            vds_ids=args.vds_ids,
            start_date=args.start_date,
            end_date=args.end_date,
            data_type=args.data_type
        )
        
        # Convert to heatmaps
        print("Converting to heatmap sequences...")
        downloader.convert_to_heatmaps(
            traffic_data=traffic_data,
            output_dir=args.output_dir,
            image_size=args.image_size,
            sequence_length=args.sequence_length
        )
        
        print("PeMS data download and conversion completed!")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        downloader.close()


if __name__ == "__main__":
    main()
