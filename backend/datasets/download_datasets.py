#!/usr/bin/env python3
"""
Dataset Download and Setup Utility
Downloads and prepares all required medical datasets for the federated learning system.
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path

# Dataset configurations
DATASETS = {
    'PTB-XL': {
        'url': 'https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip',
        'filename': 'ptb-xl.zip',
        'extract_to': 'ptb-xl',
        'description': 'ECG dataset with 21,837 clinical 12-lead ECG records',
        'size': '~850 MB'
    },
    'UCI-Parkinsons': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data',
        'filename': 'parkinsons.data',
        'extract_to': 'uci-parkinsons',
        'description': 'Voice measurements from 31 people (23 with Parkinson\'s)',
        'size': '~40 KB'
    },
    'WESAD': {
        'url': 'manual',
        'filename': 'WESAD.zip',
        'extract_to': 'wesad',
        'description': 'Wearable Stress and Affect Detection dataset',
        'size': '~2 GB',
        'instructions': '''
        WESAD requires manual download:
        1. Visit: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
        2. Fill out the request form
        3. Download WESAD.zip
        4. Place it in: datasets/downloads/WESAD.zip
        '''
    },
    'OhioT1DM': {
        'url': 'manual',
        'filename': 'OhioT1DM.zip',
        'extract_to': 'ohio-t1dm',
        'description': 'Blood glucose levels from Type 1 Diabetes patients',
        'size': '~50 MB',
        'instructions': '''
        OhioT1DM requires manual download:
        1. Visit: http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html
        2. Request access via the form
        3. Download the dataset
        4. Place it in: datasets/downloads/OhioT1DM.zip
        '''
    }
}

# Note: HandPD and PC-GITA are not publicly available without institutional access
# We'll use synthetic data generation for handwriting in this implementation

class DatasetDownloader:
    def __init__(self, base_dir='datasets'):
        self.base_dir = Path(base_dir)
        self.downloads_dir = self.base_dir / 'downloads'
        self.raw_dir = self.base_dir / 'raw'
        
        # Create directories
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url, filename):
        """Download a file with progress indicator"""
        filepath = self.downloads_dir / filename
        
        if filepath.exists():
            print(f"✓ {filename} already downloaded")
            return filepath
        
        print(f"Downloading {filename}...")
        
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r  Progress: {percent}%")
            sys.stdout.flush()
        
        try:
            urllib.request.urlretrieve(url, filepath, progress_hook)
            print(f"\n✓ Downloaded {filename}")
            return filepath
        except Exception as e:
            print(f"\n✗ Failed to download {filename}: {e}")
            return None
    
    def extract_archive(self, filepath, extract_to):
        """Extract zip or tar archive"""
        extract_path = self.raw_dir / extract_to
        
        if extract_path.exists():
            print(f"✓ {extract_to} already extracted")
            return extract_path
        
        print(f"Extracting {filepath.name}...")
        
        try:
            if filepath.suffix == '.zip':
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
            elif filepath.suffix in ['.tar', '.gz', '.tgz']:
                with tarfile.open(filepath, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_path)
            else:
                # For single files, just copy
                extract_path.mkdir(parents=True, exist_ok=True)
                shutil.copy(filepath, extract_path / filepath.name)
            
            print(f"✓ Extracted to {extract_to}")
            return extract_path
        except Exception as e:
            print(f"✗ Failed to extract {filepath.name}: {e}")
            return None
    
    def setup_dataset(self, dataset_name):
        """Download and setup a specific dataset"""
        if dataset_name not in DATASETS:
            print(f"✗ Unknown dataset: {dataset_name}")
            return False
        
        config = DATASETS[dataset_name]
        print(f"\n{'='*60}")
        print(f"Setting up: {dataset_name}")
        print(f"Description: {config['description']}")
        print(f"Size: {config['size']}")
        print(f"{'='*60}")
        
        # Check if manual download required
        if config['url'] == 'manual':
            manual_path = self.downloads_dir / config['filename']
            if not manual_path.exists():
                print(f"\n⚠ Manual download required:")
                print(config['instructions'])
                return False
            else:
                print(f"✓ Found manually downloaded file: {config['filename']}")
                filepath = manual_path
        else:
            # Auto download
            filepath = self.download_file(config['url'], config['filename'])
            if not filepath:
                return False
        
        # Extract
        extract_path = self.extract_archive(filepath, config['extract_to'])
        return extract_path is not None
    
    def setup_all(self):
        """Setup all datasets"""
        print("\n" + "="*60)
        print("FEDERATED ADAPTIVE LEARNING SYSTEM - Dataset Setup")
        print("="*60)
        
        results = {}
        for dataset_name in DATASETS.keys():
            results[dataset_name] = self.setup_dataset(dataset_name)
        
        # Summary
        print("\n" + "="*60)
        print("SETUP SUMMARY")
        print("="*60)
        
        for dataset_name, success in results.items():
            status = "✓ Ready" if success else "✗ Needs manual setup"
            print(f"{dataset_name:20s}: {status}")
        
        # Create synthetic handwriting data note
        handwriting_dir = self.raw_dir / 'handwriting-synthetic'
        handwriting_dir.mkdir(exist_ok=True)
        
        readme = handwriting_dir / 'README.txt'
        readme.write_text('''
Synthetic Handwriting Data

HandPD and PC-GITA datasets require institutional access.
For this implementation, we generate synthetic handwriting data
that simulates tremor patterns and motor control variations.

The synthetic data generator creates:
- Spiral drawing tasks
- Handwriting stroke sequences
- Tremor frequency analysis
- Motor control metrics

This allows the system to demonstrate multi-modal fusion
without requiring access to restricted datasets.
        ''')
        
        print(f"\nhandwriting-synthetic: ✓ Synthetic data generator ready")
        
        print("\n" + "="*60)
        print("Next steps:")
        print("1. Complete any manual downloads listed above")
        print("2. Run: python backend/datasets/verify_datasets.py")
        print("3. Start the backend: cd backend && uvicorn app.main:app")
        print("="*60 + "\n")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and setup medical datasets')
    parser.add_argument('--dataset', type=str, help='Specific dataset to setup (default: all)')
    parser.add_argument('--base-dir', type=str, default='datasets', help='Base directory for datasets')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.base_dir)
    
    if args.dataset:
        downloader.setup_dataset(args.dataset)
    else:
        downloader.setup_all()

if __name__ == '__main__':
    main()
