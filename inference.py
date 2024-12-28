import argparse
import torch
import os
from models import (
    build_vgg16,
    build_mobilenet_v2,
    build_mobilenet_v3_large
)
from train_utils import test_model
from data import get_cifar10_loaders
from visualization import plot_metrics
from PIL import Image
from torchvision import transforms

def preprocess_image(img_path):
    # Example of a direct transform, or import from inference_utils
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_ops = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    img = Image.open(img_path).convert('RGB')
    return transform_ops(img).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(description='Inference script for CIFAR-10 Transfer Learning')
    parser.add_argument('--model_name', default='mobilenet_v3', type=str,
                        help='Model name: vgg16, mobilenet_v2, mobilenet_v3')
    parser.add_argument('--weights', default='checkpoints/mobilenet_v3_best.pth', type=str,
                        help='Path to saved model weights')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image')
    args = parser.parse_args()

    # Build model
    if args.model_name == 'vgg16':
        model = build_vgg16(pretrained=False, freeze=False)  # We won't freeze since we are only loading
    elif args.model_name == 'mobilenet_v2':
        model = build_mobilenet_v2(pretrained=False, freeze=False)
    elif args.model_name == 'mobilenet_v3':
        model = build_mobilenet_v3_large(pretrained=False, freeze=False)
    else:
        raise ValueError(f"Unknown model: {args.model_name}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"Weights file not found: {args.weights}")

    # Load weights
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # Preprocess the image
    img_tensor = preprocess_image(args.image_path).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)
        print(f"Predicted class index: {predicted.item()}")

if __name__ == "__main__":
    main()
