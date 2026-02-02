
from __future__ import annotations
from torchvision import transforms

def build_transforms(image_size: int):
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomRotation(degrees=7, fill=(0,0,0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25]),
    ])
    return train_tfms, val_tfms
