import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import yaml
import torch
import torch.nn as nn

from src.models import ConvolutionalVisionTransformer, ViT
from src.data import create_data_loaders
from src.utils import setup_logging, EarlyStopping


def parse_args():
    parser = argparse.ArgumentParser(description='Train CvT model')
    parser.add_argument('--config', 
                       default='configs/cvt-13-224x224.yaml',
                       help='path to config file')
    parser.add_argument('--data-dir',
                       required=True,
                       help='path to dataset directory')
    parser.add_argument('--checkpoint-dir',
                       default='checkpoints',
                       help='path to save model checkpoints')
    parser.add_argument('--log-dir',
                       default='logs',
                       help='path to save training logs')
    parser.add_argument('--device', 
                       default='cuda',
                       help='device to use (cuda or cpu)')
    parser.add_argument('--resume',
                       default='',
                       help='path to resume from checkpoint')
    parser.add_argument('--early-stopping', 
                       type=lambda x: x.lower() == 'true',
                       default=True,
                       help='Enable early stopping (default: True)')
    parser.add_argument('--patience',
                       type=int,
                       default=7,
                       help='patience epochs for early stopping')
    parser.add_argument('--min-delta',
                       type=float,
                       default=1e-3,
                       help='minimum change in val accuracy to be considered as improvement')
    return parser.parse_args()

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, logger):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        # Update scheduler
        scheduler.step() 
        
        # Calculate accuracy
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        total_loss += loss.item()
        current_lr = optimizer.param_groups[0]['lr']

            
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    logger.info(f'Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, LR: {current_lr:.6f}')
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device, logger):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    val_loss /= len(val_loader)
    accuracy = 100. * correct / len(val_loader.dataset)
    
    logger.info(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return val_loss, accuracy

def main():
    args = parse_args()
    
    # Setup directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info('Starting training...')
    logger.info(f'Config file: {args.config}')
    logger.info(f'Data directory: {args.data_dir}')

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load config
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.error(f"Error loading config file: {e}")
        return

    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(
        data_dir=args.data_dir,
        img_size=tuple(config['TRAIN']['IMAGE_SIZE']),
        batch_size=config['TRAIN']['BATCH_SIZE'],
        num_workers=config['WORKERS']
    )
    
    num_classes = len(train_loader.dataset.classes)
    logger.info(f'Number of classes: {num_classes}')

    # Create model based on config name
    if config['MODEL']['NAME'] == 'cls_cvt':
        model = ConvolutionalVisionTransformer(
            in_chans=3,
            num_classes=num_classes,
            init=config['MODEL']['SPEC']['INIT'],
            spec=config['MODEL']['SPEC']
        )
    elif config['MODEL']['NAME'] == 'cls_vit':
        model = ViT(
            image_size=config['TRAIN']['IMAGE_SIZE'][0],
            patch_size=config['MODEL']['SPEC']['PATCH_SIZE'],
            num_classes=num_classes,
            dim=config['MODEL']['SPEC']['EMBED_DIM'],
            depth=config['MODEL']['SPEC']['DEPTH'],
            heads=config['MODEL']['SPEC']['NUM_HEADS'],
            dim_head=config['MODEL']['SPEC']['DIM_HEAD'],
            mlp_dim=int(config['MODEL']['SPEC']['EMBED_DIM'] * config['MODEL']['SPEC']['MLP_RATIO']),
            pool=config['MODEL']['SPEC']['POOL'],
            dropout=config['MODEL']['SPEC']['DROP_RATE'],
            emb_dropout=config['MODEL']['SPEC']['ATTN_DROP_RATE'],
            init=config['MODEL']['SPEC']['INIT']
        )
    else:
        raise ValueError(f"Unknown model type: {config['MODEL']['NAME']}")

    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['TRAIN']['LR'],
        weight_decay=config['TRAIN']['WD']
    )

    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['TRAIN']['LR'],
        epochs=config['TRAIN']['END_EPOCH'],
        steps_per_epoch=len(train_loader),
        pct_start=config['TRAIN']['WARMUP_EPOCHS'] / config['TRAIN']['END_EPOCH'],
        div_factor=25,
        final_div_factor=1e4
    )

    # Resume from checkpoint if specified
    start_epoch = config['TRAIN']['BEGIN_EPOCH']
    best_acc = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f'Loading checkpoint from {args.resume}')
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            logger.info(f'Loaded checkpoint from epoch {start_epoch}')
            logger.info(f'Previous best accuracy: {best_acc:.2f}%')

    # Initialize early stopping
    if args.early_stopping:
        early_stopping = EarlyStopping(
            patience=args.patience,
            min_delta=args.min_delta,
            mode='max'
        )

    # Training loop
    num_epochs = config['TRAIN']['END_EPOCH']
    for epoch in range(start_epoch, num_epochs):
        logger.info(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, 
            optimizer, scheduler, device, epoch+1, logger
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, 
            device, logger
        )
        
        # Save checkpoint if validation accuracy improves
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc,
                'config': config
            }
            save_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, save_path)
            logger.info(f'Saved new best model with accuracy: {best_acc:.2f}% to {save_path}')
        
        if args.early_stopping and early_stopping(val_acc):
            logger.info(f'Early stopping triggered after {epoch + 1} epochs')
            break

        # Save last checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_acc': best_acc,
            'config': config
        }
        save_path = os.path.join(args.checkpoint_dir, 'last_model.pth')
        torch.save(checkpoint, save_path)

if __name__ == '__main__':
    main()