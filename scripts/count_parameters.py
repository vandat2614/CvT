import sys
from pathlib import Path
import torch
import yaml
import argparse

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
    
    print("\nModel Parameter Counts:")
    print("-" * 50)
    print(f"{'Model':<15} {'Parameters':<15} {'Size':<10}")
    print("-" * 50)
    
    # Process each model
    for model_name, model_cfg in config['MODELS'].items():
        try:
            # Fix path resolution for model config
            config_path = project_root / 'configs' / Path(model_cfg['config_path']).name
            with open(config_path) as f:
                model_config = yaml.safe_load(f)
            
            # Create model
            model = create_model(model_config, num_classes=185)
            
            # Count parameters
            num_params = count_parameters(model)
            
            # Calculate approximate model size
            model_size = num_params * 4 / (1024 * 1024)  # Size in MB
            
            print(f"{model_name:<15} {format_number(num_params):<15} {model_size:.1f}MB")
            
        except Exception as e:
            print(f"{model_name:<15} Error: {str(e)}")
    
    print("-" * 50)
if __name__ == '__main__':
    main()