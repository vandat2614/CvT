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

from src.models import ConvolutionalVisionTransformer
from src.data import create_data_loaders
from src.utils import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description='Test CvT model')
    parser.add_argument('--config', 
                       default='configs/cvt-13-224x224.yaml',
                       help='path to config file')
    parser.add_argument('--data-dir',
                       required=True,
                       help='path to dataset directory')
    parser.add_argument('--checkpoint',
                       default='',
                       help='path to model checkpoint')
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
    
    # Setup logging directly to test.log
    logger = setup_logging(args.output_dir, filename='test.log')
    logger.info('Starting evaluation...')
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
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
    
    # Create model
    model = ConvolutionalVisionTransformer(
        in_chans=3,
        num_classes=num_classes,
        init=config['MODEL']['SPEC']['INIT'],
        spec=config['MODEL']['SPEC']
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f'Loading checkpoint from {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
    else:
        logger.info('Using randomly initialized model')
    
    model = model.to(device)
    
    # Evaluate model
    evaluate_model(model, test_loader, device, logger)
    logger.info('Evaluation complete.')

if __name__ == '__main__':
    main()