
from __future__ import annotations
import torch.nn as nn
import torchvision

def get_model(backbone: str = 'resnet18', pretrained: bool = True, out_dim: int = 10) -> nn.Module:
    if backbone == 'resnet18':
        model = torchvision.models.resnet18(weights=(
            torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        ))
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, out_dim)
        return model
    elif backbone == 'resnet34':
        model = torchvision.models.resnet34(weights=(
            torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        ))
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, out_dim)
        return model
    else:
        raise ValueError(f'Unsupported backbone: {backbone}')
