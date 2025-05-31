import sys
import os
import argparse
import yaml
import torch
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import classification_report

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data import create_data_loaders
from src.utils import create_model, setup_console_logger



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
    parser.add_argument('--verbose',
                       type=lambda x: x.lower() == 'true',
                       default=True,
                       help='Print classification report and per-class accuracy (default: True)')
    return parser.parse_args()

def evaluate_model(model, test_loader, device, output_dir, logger, verbose=True):
    model.eval()
    all_targets = []
    all_predictions = []
    
    # Run inference silently if not verbose
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(pred.cpu().numpy())
            
            if verbose and (batch_idx + 1) % 10 == 0:
                logger.info(f'Processed {batch_idx + 1}/{len(test_loader)} batches')

    # Convert to numpy arrays
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    class_names = test_loader.dataset.classes

    # Generate report
    report = classification_report(
        all_targets,
        all_predictions,
        target_names=class_names,
        digits=4
    )
    
    print("?????????????????")

    # Save report to file
    report_path = Path(output_dir) / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)

    # Calculate per-class stats
    class_stats = {}
    for i, class_name in enumerate(class_names):
        mask = all_targets == i
        class_correct = int((all_predictions[mask] == i).sum())
        class_total = int(mask.sum())
        class_acc = float((class_correct / class_total) * 100)
        
        class_stats[class_name] = {
            'accuracy': round(class_acc, 2),
            'correct_samples': class_correct,
            'total_samples': class_total
        }
    
    # Save to JSON
    stats_path = Path(output_dir) / 'class_accuracy.json'
    with open(stats_path, 'w') as f:
        json.dump(class_stats, f, indent=4)

    # Print all information only if verbose
    if verbose:
        logger.info('\nClassification Report:')
        logger.info('\n' + report)
        logger.info(f'Classification report saved to {report_path}')
        
        logger.info('\nPer-class Statistics:')
        for class_name, stats in class_stats.items():
            logger.info(f'{class_name}: {stats["accuracy"]:.2f}% ({stats["correct_samples"]}/{stats["total_samples"]})')
        logger.info(f'\nPer-class statistics saved to {stats_path}')

def main():
    args = parse_args()
    
    # Create results directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup console logger
    logger = setup_console_logger()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Only log if verbose
    if args.verbose:
        logger.info('Starting evaluation...')
        logger.info(f'Using device: {device}')
    
    # Create data loader
    _, _, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        img_size=tuple(config['TEST']['IMAGE_SIZE']),
        batch_size=config['TEST']['BATCH_SIZE'],
        num_workers=config['WORKERS']
    )
    
    if test_loader is None:
        if args.verbose:
            logger.error('No test data found!')
        return
    
    if args.verbose:
        num_classes = len(test_loader.dataset.classes)
        logger.info(f'Number of classes: {num_classes}')
        logger.info(f'Test batch size: {config["TEST"]["BATCH_SIZE"]}')
        logger.info(f'Total test samples: {len(test_loader.dataset)}')
    
    # Create and load model
    num_classes = len(test_loader.dataset.classes)
    model = create_model(config, num_classes)
    model = model.to(device)
    
    if args.verbose:
        logger.info(f'Loading weights from {args.weights}')
    weights = torch.load(args.weights, map_location=device)
    if isinstance(weights, dict) and 'model' in weights:
        model.load_state_dict(weights['model'])
    else:
        model.load_state_dict(weights)
    
    # Evaluate model
    evaluate_model(model, test_loader, device, args.output_dir, logger, args.verbose)
    
    if args.verbose:
        logger.info('\nEvaluation complete.')