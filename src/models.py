import torch
import torch.nn as nn
from torchvision import models
from config import Config


def get_model(model_name=Config.MODEL_NAME,
              num_classes=Config.NUM_CLASSES,
              pretrained=Config.PRETRAINED,
              freeze_backbone=Config.FREEZE_BACKBONE):
    if model_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif model_name == 'resnet34':
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
    elif model_name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def print_model_summary(model):
    print("=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(model)
    print("=" * 60)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60)


if __name__ == '__main__':
    model = get_model()
    print_model_summary(model)
