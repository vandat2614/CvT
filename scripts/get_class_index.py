import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data import create_data_loaders

def parse_args():
    parser = argparse.ArgumentParser(description='Create class index mapping')
    parser.add_argument('--data-dir',
                       required=True,
                       help='path to dataset directory')
    parser.add_argument('--output-dir',
                       default='output',
                       help='directory to save class mapping')
    return parser.parse_args()

def create_class_mapping(data_dir, output_dir):
    """Create and save class index mapping"""
    # Get class names from dataloader
    train_loader, _, _ = create_data_loaders(
        data_dir=data_dir,
        img_size=(224, 224),
        batch_size=1,
        num_workers=0
    )
    
    if train_loader is None:
        raise ValueError("No training data found!")
    
    # Create mappings
    class_to_idx = train_loader.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save mappings
    mappings = {
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'num_classes': len(class_to_idx)
    }
    
    output_path = Path(output_dir) / 'class_mapping.json'
    with open(output_path, 'w') as f:
        json.dump(mappings, f, indent=4)
    
    print(f"Class mapping saved to {output_path}")
    print(f"Found {len(class_to_idx)} classes")

def main():
    args = parse_args()
    create_class_mapping(args.data_dir, args.output_dir)

if __name__ == '__main__':
    main()