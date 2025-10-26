# inference.py

import torch
import torchvision
import torchvision.transforms as transforms
import argparse

# You must import the model definition to recreate the model structure
from model import ResNet18 

def run_inference():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='CIFAR-10 Inference Script')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to the trained model checkpoint (.pth file)')
    parser.add_argument('--data_dir', default='./data', type=str, help='directory for CIFAR-10 data')
    args = parser.parse_args()

    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Model ---
    # 1. Instantiate the same model architecture used for training
    model = ResNet18(num_classes=10)
    
    # 2. Load the saved weights (state_dict) into the model structure
    #    map_location ensures it works even if you trained on GPU and infer on CPU
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    # 3. Move the model to the appropriate device (GPU or CPU)
    model.to(device)
    
    # 4. CRITICAL: Set the model to evaluation mode
    #    This disables layers like Dropout and sets BatchNorm layers to use running stats
    model.eval()
    print("Model loaded successfully from checkpoint.")

    # --- Prepare Data ---
    # For inference, we only need basic normalization, not augmentation
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

    test_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    # CIFAR-10 class names for friendly output
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # --- Run Inference on a Batch ---
    print("\nRunning inference on a sample batch from the test set...")
    
    # Get a single batch of test images and labels
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # Disable gradient calculation for efficiency
    with torch.no_grad():
        outputs = model(images)
        # Get predictions by finding the index of the max logit (score)
        _, predicted_indices = torch.max(outputs, 1)

    # --- Display Results ---
    print("\n--- Inference Results (Prediction vs. Ground Truth) ---")
    for i in range(images.size(0)):
        predicted_class = classes[predicted_indices[i]]
        actual_class = classes[labels[i]]
        print(f"Image #{i+1}: Model Predicted: {predicted_class:<10} | Actual: {actual_class}")

if __name__ == '__main__':
    run_inference()