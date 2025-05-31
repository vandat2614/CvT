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
    parser = argparse.ArgumentParser(description='Save class names from dataset')
    parser.add_argument('--data-dir',
                       required=True,
                       help='path to dataset directory')
    parser.add_argument('--output-dir',
                       default='app/configs',
                       help='directory to save classes.json')
    return parser.parse_args()

def save_class_names(data_dir, output_dir):
    """Extract and save class names from dataset"""
    # Create data loader with minimal batch size just to get classes
    train_loader, _, _ = create_data_loaders(
        data_dir=data_dir,
        img_size=(224, 224),
        batch_size=1,
        num_workers=0
    )
    
    if train_loader is None:
        raise ValueError("No training data found!")
        
    # Get class names
    class_names = train_loader.dataset.classes
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to JSON
    output_path = Path(output_dir) / 'classes.json'
    with open(output_path, 'w') as f:
        json.dump({
            'class_names': class_names,
            'num_classes': len(class_names)
        }, f, indent=4)
    
    print(f"Class names saved to {output_path}")

def main():
    args = parse_args()
    save_class_names(args.data_dir, args.output_dir)

if __name__ == '__main__':
    main()