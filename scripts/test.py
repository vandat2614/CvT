import sys
import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data import create_data_loaders
from src.utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Test Vision Models')
    parser.add_argument('--config', 
                       required=True,
                       help='path to config file')
    parser.add_argument('--data-dir',
                       required=True,
                       help='path to dataset directory')
    parser.add_argument('--weights',
                       required=True,
                       help='path to model weights (can be checkpoint or state_dict)')
    parser.add_argument('--output-dir',
                       default='results',
                       help='path to save results')
    parser.add_argument('--device',
                       default='cuda',
                       help='device to use (cuda or cpu)')

    return parser.parse_args()

def evaluate_model(model, test_loader, device, logger):
    model.eval()
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(pred.cpu().numpy())

    # Convert to numpy arrays
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    class_names = test_loader.dataset.classes

    # Calculate and log classification report
    report = classification_report(
        all_targets,
        all_predictions,
        target_names=class_names,
        digits=4
    )
    logger.info("\nClassification Report:")
    logger.info(f"\n{report}")

    # Calculate and log per-class accuracy
    logger.info("\nPer-class Accuracy:")
    for i, class_name in enumerate(class_names):
        mask = all_targets == i
        class_correct = (all_predictions[mask] == i).sum()
        class_total = mask.sum()
        class_acc = (class_correct / class_total) * 100
        logger.info(f"{class_name}: {class_acc:.2f}% ({class_correct}/{class_total})")

def main():
    args = parse_args()
    
    # Create results directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(os.path.join(args.output_dir, 'test.log'))
    logger.info('Starting evaluation...')
    
    # Setup device
    device = torch.device(args.device)
    logger.info(f'Using device: {device}')
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data loader
    _, _, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        img_size=tuple(config['TRAIN']['IMAGE_SIZE']),
        batch_size=config['TRAIN']['BATCH_SIZE'],
        num_workers=config['WORKERS']
    )
    
    if test_loader is None:
        logger.error('No test data found!')
        return
    
    num_classes = len(test_loader.dataset.classes)
    logger.info(f'Number of classes: {num_classes}')
    
    # Create model based on config
    model = create_model(config, num_classes)
    model = model.to(device)
    
    # Load weights
    logger.info(f'Loading weights from {args.weights}')
    weights = torch.load(args.weights, map_location=device)
    if isinstance(weights, dict) and 'model' in weights:
        # Loading from checkpoint
        model.load_state_dict(weights['model'])
    else:
        # Loading just the model state dict
        model.load_state_dict(weights)
    
    # Evaluate model
    evaluate_model(model, test_loader, device, logger)
    logger.info('Evaluation complete.')

if __name__ == '__main__':
    main()