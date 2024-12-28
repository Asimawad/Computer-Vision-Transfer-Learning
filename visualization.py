import matplotlib.pyplot as plt

def plot_metrics(history, title='Training History'):
    """
    Expects `history` to be a dictionary with keys like:
      - 'train_loss', 'val_loss', 'train_acc', 'val_acc'
      each a list of values per epoch.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12,5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()
