import torch
import torch.nn as nn

def create_optimizer(config, model_parameters):
    """Create optimizer based on config"""

    if config['TRAIN']['OPTIMIZER'] == 'SGD':
        return torch.optim.SGD(
            model_parameters,
            lr=config['TRAIN']['LR'],
            momentum=config['TRAIN']['MOMENTUM'],
            weight_decay=config['TRAIN']['WD']
        )
    
    elif config['TRAIN']['OPTIMIZER'] == 'AdamW':
        return torch.optim.AdamW(
            model_parameters,
            lr=config['TRAIN']['LR'],
            weight_decay=config['TRAIN']['WD']
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['TRAIN']['OPTIMIZER']}")

def create_scheduler(config, optimizer, train_loader=None):
    """Create learning rate scheduler based on config"""
    scheduler_config = config['TRAIN']['SCHEDULER']
    
    if scheduler_config['NAME'] == 'OneCycleLR':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['TRAIN']['LR'],
            epochs=config['TRAIN']['END_EPOCH'],
            steps_per_epoch=len(train_loader),
            pct_start=scheduler_config['WARMUP_EPOCHS'] / config['TRAIN']['END_EPOCH'],
            div_factor=scheduler_config['DIV_FACTOR'],
            final_div_factor=scheduler_config['FINAL_DIV_FACTOR']
        )
    elif scheduler_config['NAME'] == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config['STEP_SIZE'],
            gamma=scheduler_config['GAMMA']
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_config['NAME']}")