"""
Script to download Vietnam Housing Dataset from Kaggle

Usage:
1. Install kaggle API: pip install kaggle
2. Set up Kaggle API credentials (see: https://www.kaggle.com/docs/api)
3. Run: python download_dataset.py
"""

import os
import sys
from pathlib import Path

def download_kaggle_dataset():
    """Download dataset from Kaggle using API."""
    try:
        import kaggle
    except ImportError:
        print("Error: kaggle package not found.")
        print("Please install it using: pip install kaggle")
        sys.exit(1)
    
    # Set dataset information
    dataset_name = "ladcva/vietnam-housing-dataset-hanoi"
    
    # Create data directory if not exists
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    try:
        print(f"Downloading dataset: {dataset_name}")
        print(f"Destination: {data_dir}")
        
        # Download dataset
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(data_dir),
            unzip=True
        )
        
        print("✓ Dataset downloaded successfully!")
        print(f"Files in {data_dir}:")
        for file in data_dir.iterdir():
            if file.is_file():
                print(f"  - {file.name} ({file.stat().st_size / 1024 / 1024:.2f} MB)")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have Kaggle API credentials set up")
        print("2. Go to https://www.kaggle.com/account")
        print("3. Click 'Create New API Token'")
        print("4. Place kaggle.json in:")
        print("   - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json")
        print("   - Linux/Mac: ~/.kaggle/kaggle.json")
        sys.exit(1)

def download_manual():
    """Print manual download instructions."""
    print("\n" + "="*70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print("\nIf automatic download fails, please download manually:")
    print("\n1. Go to: https://www.kaggle.com/datasets/")
    print("   Search for: 'Vietnam Housing Dataset'")
    print("\n2. Click 'Download' button")
    print("\n3. Extract the downloaded zip file")
    print("\n4. Copy 'vietnam_housing_dataset.csv' to:")
    print(f"   {Path(__file__).parent / 'data' / 'vietnam_housing_dataset.csv'}")
    print("\n" + "="*70)

if __name__ == "__main__":
    print("Vietnam Housing Dataset Downloader")
    print("-" * 40)
    
    choice = input("\nDownload method:\n1. Automatic (Kaggle API)\n2. Manual instructions\nChoose (1/2): ")
    
    if choice == "1":
        download_kaggle_dataset()
    else:
        download_manual()
