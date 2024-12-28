import torch.nn as nn
from torchvision import models

def build_vgg16(pretrained=True, freeze=True, num_classes=10):
    """
    Builds a VGG16 model, replaces final layers with custom classification for CIFAR-10.
    Optionally freeze convolutional layers.
    """
    model = models.vgg16(weights='IMAGENET1K_V1' if pretrained else None)

    # Freeze all layers if freeze=True
    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier fully
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, num_classes)
    )
    return model

def build_mobilenet_v2(pretrained=True, freeze=True, num_classes=10):
    """
    Builds a MobileNetV2, modifies classification layer, optional freezing.
    """
    model = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)

    # Freeze if needed
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier
    model.classifier = nn.Sequential(
        nn.Linear(1280, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model

def build_mobilenet_v3_large(pretrained=True, freeze=True, num_classes=10):
    """
    Builds a MobileNetV3 Large, modifies classification layer, optional freezing.
    """
    model = models.mobilenet_v3_large(weights='IMAGENET1K_V1' if pretrained else None)

    # Freeze
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # Only replacing the last sublayer in model.classifier
    # model.classifier[3] is typically the last linear
    model.classifier[3] = nn.Sequential(
        nn.Linear(1280, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    return model
