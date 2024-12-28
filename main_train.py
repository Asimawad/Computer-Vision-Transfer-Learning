import torch
import torch.nn as nn
import torch.optim as optim

from data import get_cifar10_loaders
from models import (
    build_vgg16,
    build_mobilenet_v2,
    build_mobilenet_v3_large
)
from train_utils import train_one_epoch, validate, test_model
from visualization import plot_metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Get data loaders
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        data_dir='./data', batch_size=128, train_ratio=0.8, num_workers=2)

    # 2. Build models
    #    You can easily add or remove models here
    model_dict = {
        'vgg16': build_vgg16(pretrained=True, freeze=True, num_classes=10),
        'mobilenet_v2': build_mobilenet_v2(pretrained=True, freeze=True, num_classes=10),
        'mobilenet_v3': build_mobilenet_v3_large(pretrained=True, freeze=True, num_classes=10)
    }

    for model_name, model in model_dict.items():
        print(f"\n===== Training {model_name} =====")
        model = model.to(device)

        # 3. Set up optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        num_epochs = 10  # Example

        # 4. Training loop
        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            # Log
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            print(f"Epoch [{epoch+1}/{num_epochs}]"
                  f" - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
                  f" - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = f'checkpoints/{model_name}_best.pth'
                torch.save(model.state_dict(), save_path)
                print(f"==> Saved best {model_name} model (Val Acc: {best_val_acc:.2f}%)")

        # Plot the training history
        plot_metrics(history, title=model_name)

        # Evaluate on test set
        model.eval()  # ensure model is in eval mode
        test_acc = test_model(model, test_loader, device)
        print(f"{model_name} Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
