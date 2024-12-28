import torch
import torchvision
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

def get_cifar10_loaders(data_dir='./data', batch_size=128, train_ratio=0.8, num_workers=2):
    """
    Loads CIFAR-10, splits it into train/validation, 
    and provides DataLoaders for train, validation, and test sets.
    """
    # For pretrained ImageNet models, we need to resize and normalize accordingly
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_ops = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # Download CIFAR-10 if not present
    train_full = torchvision.datasets.CIFAR10(root=data_dir, train=True, 
                                              download=True, transform=transform_ops)
    test_data = torchvision.datasets.CIFAR10(root=data_dir, train=False, 
                                             download=True, transform=transform_ops)

    # Create train/validation split
    dataset_size = len(train_full)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    train_set, val_set = random_split(train_full, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
