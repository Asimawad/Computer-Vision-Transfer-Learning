import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load the pretrained MobileNetV2 model
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mobilenet.to(device)

# Preprocessing transformations
trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

def apply_mask(img, width, height, mask_size):
    """Apply a mask to the image at the specified location."""
    img_tensor = trans(img).unsqueeze(0).to(device)
    image_copy = img_tensor.clone()
    image_copy[:, :, width:width + mask_size, height:height + mask_size] = 0
    return image_copy

def generate_images(img, mask_size, stride):
    """Generate masked images by sliding a mask across the input image."""
    masked_images = []
    for height in range(0, 224, stride):
        for width in range(0, 224, stride):
            masked_image = apply_mask(img, width, height, mask_size)
            masked_images.append(masked_image)
    return masked_images

def get_predictions(masked_images):
    """Get the top predictions for a batch of masked images."""
    predictions = []
    for img_tensor in masked_images:
        img = norm(img_tensor)
        prediction = mobilenet(img)
        probabilities = F.softmax(prediction[0], dim=0)
        predictions.append(probabilities)
    return predictions

def get_saliency_map(masked_images):
    """Generate a saliency map based on the model's output probabilities."""
    saliency_map = []
    for img_tensor in masked_images:
        img = norm(img_tensor)
        prediction = mobilenet(img)
        probabilities = F.softmax(prediction[0], dim=0)
        saliency_map.append(probabilities.max(0)[0].item())
    return saliency_map

def visualize_saliency_map(saliency_map, stride, mask_size):
    """Visualize the saliency map as a heatmap."""
    saliency = np.array(saliency_map).reshape((224 // stride, 224 // stride))
    plt.imshow(saliency, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Saliency Map (Mask Size: {mask_size}, Stride: {stride})")
    plt.show()

def main():
    # Load and display the input image
    img_path = 'path_to_your_image.jpg'  # Replace with your image path
    img = Image.open(img_path)
    plt.imshow(img)
    plt.title("Input Image")
    plt.show()

    # Masking parameters
    mask_size = 30
    stride = 20

    # Generate masked images
    masked_images = generate_images(img, mask_size=mask_size, stride=stride)

    # Create saliency map
    saliency_map = get_saliency_map(masked_images)

    # Visualize the saliency map
    visualize_saliency_map(saliency_map, stride=stride, mask_size=mask_size)

if __name__ == "__main__":
    main()