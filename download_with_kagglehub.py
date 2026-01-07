import kagglehub
import shutil
import os
from pathlib import Path

# Download latest version
print("Downloading dataset from Kaggle using kagglehub...")
path = kagglehub.dataset_download("ladcva/vietnam-housing-dataset-hanoi")

print(f"Path to dataset files: {path}")

# Copy the dataset to the data folder
data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)

# Find the CSV file in the downloaded path
downloaded_files = list(Path(path).glob("*.csv"))
if downloaded_files:
    source_file = downloaded_files[0]
    dest_file = data_dir / "vietnam_housing_dataset.csv"
    
    print(f"Copying {source_file.name} to {dest_file}...")
    shutil.copy2(source_file, dest_file)
    
    file_size = dest_file.stat().st_size / (1024 * 1024)  # Size in MB
    print(f"✓ Dataset copied successfully!")
    print(f"✓ File: {dest_file}")
    print(f"✓ Size: {file_size:.2f} MB")
else:
    print("❌ No CSV file found in downloaded dataset")
