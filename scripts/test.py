import sys
import os
import datetime
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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

def evaluate_model(model, test_loader, device, logger, save_dir):
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

    # Calculate and save confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

    # Create results DataFrame
    results_df = pd.DataFrame({
        'True_Label': [class_names[i] for i in all_targets],
        'Predicted_Label': [class_names[i] for i in all_predictions]
    })
    results_df.to_csv(os.path.join(save_dir, 'test_results.csv'), index=False)

    # Log accuracy
    accuracy = (all_targets == all_predictions).mean() * 100
    logger.info(f'Test Accuracy: {accuracy:.2f}%')

def main():
    args = parse_args()
    
    # Create results directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(args.output_dir, f'evaluation_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(os.path.join(results_dir, 'test.log'))
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
    logger.info('Starting evaluation...')
    evaluate_model(model, test_loader, device, logger, results_dir)
    logger.info(f'Evaluation complete. Results saved to {results_dir}')

if __name__ == '__main__':
    main()