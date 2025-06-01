import sys
from pathlib import Path
import torch
import yaml
import argparse
from collections import defaultdict
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils import create_model

def count_parameters(model):
    """Count number of trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_number(num):
    """Format large numbers with commas and M/B suffixes"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)

def get_file_size(filepath):
    """Get file size in MB"""
    size_bytes = os.path.getsize(filepath)
    return size_bytes / (1024 * 1024)

def parse_args():
    parser = argparse.ArgumentParser(description='Count model parameters')
    parser.add_argument('--config', 
                       required=True,
                       help='path to app config file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load app config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Collect all model information
    model_info = defaultdict(dict)
    errors = []
    
    demo_dir = project_root / 'demo'
    
    # Process each model
    for model_name, model_cfg in config['MODELS'].items():
        try:
            # Get model parameters
            config_path = project_root / 'configs' / Path(model_cfg['config_path']).name
            with open(config_path) as f:
                model_config = yaml.safe_load(f)
            
            model = create_model(model_config, num_classes=185)
            num_params = count_parameters(model)
            
            # Get actual weight file size
            weight_path = demo_dir / model_cfg['weights_path']
            if weight_path.exists():
                file_size = get_file_size(weight_path)
            else:
                file_size = None
            
            model_info[model_name] = {
                'parameters': num_params,
                'weight_size': file_size
            }
            
        except Exception as e:
            errors.append(f"{model_name}: {str(e)}")
    
    # Print table
    print("\nModel Information:")
    print("-" * 75)
    print(f"{'Model':<15} {'Parameters':<15} {'Weight File Size':<20} {'Status':<15}")
    print("-" * 75)
    
    for model_name in config['MODELS'].keys():
        if model_name in model_info:
            info = model_info[model_name]
            params = format_number(info['parameters'])
            
            if info['weight_size'] is not None:
                size = f"{info['weight_size']:.2f}MB"
                status = "✓"
            else:
                size = "Not found"
                status = "✗"
                
            print(f"{model_name:<15} {params:<15} {size:<20} {status:<15}")
    
    print("-" * 75)
    
    # Print errors at the end
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"Error: {error}")

if __name__ == '__main__':
    main()