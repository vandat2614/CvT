import torch.nn as nn
from torchvision.models import resnet18
from src.models import ConvolutionalVisionTransformer, ViT

def init_resnet_weights(model, init_type='kaiming'):
    """Initialize ResNet weights if not using pretrained model
    
    Args:
        model (nn.Module): ResNet model
        init_type (str): Initialization method
    """
    if init_type == 'kaiming':
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def create_model(config, num_classes):
    """Factory method to create models based on config"""
    if config['MODEL']['NAME'] == 'cls_cvt':
        return ConvolutionalVisionTransformer(
            in_chans=3,
            num_classes=num_classes,
            init=config['MODEL']['SPEC']['INIT'],
            spec=config['MODEL']['SPEC']
        )
    elif config['MODEL']['NAME'] == 'cls_vit':
        return ViT(
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
    elif config['MODEL']['NAME'] == 'cls_resnet18':
        model = resnet18(pretrained=config['MODEL']['SPEC']['PRETRAINED'])
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Add dropout if specified
        if config['MODEL']['SPEC']['DROP_RATE'] > 0:
            model.fc = nn.Sequential(
                nn.Dropout(config['MODEL']['SPEC']['DROP_RATE']),
                model.fc
            )
            
        # Initialize weights if not pretrained
        if not config['MODEL']['SPEC']['PRETRAINED'] and config['MODEL']['SPEC']['INIT'] == 'kaiming':
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            
        return model
    else:
        raise ValueError(f"Unknown model type: {config['MODEL']['NAME']}")