# inference.py

import torch
import argparse
import os
import numpy as np
from PIL import Image

# Import your custom modules
from src.model import UNet
from src.utils import visualize_segmentation

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Run inference on a UNet model.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model checkpoint (.pth.tar file)')
    parser.add_argument('--image-dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--mask-dir', type=str, required=True, help='Directory containing corresponding ground truth masks')
    parser.add_argument('--output-dir', type=str, default='./results', help='Directory to save visualization results')
    parser.add_argument('--num-classes', type=int, default=4, help='Number of segmentation classes')
    args = parser.parse_args()

    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Load Model ---
    model = UNet(n_channels=1, n_classes=args.num_classes).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Model loaded successfully.")

    # --- Run Inference on Test Images ---
    test_image_filenames = sorted([f for f in os.listdir(args.image_dir) if f.startswith('case_')])
    
    # Let's visualize the first 5 images
    for image_filename in test_image_filenames[:5]:
        image_path = os.path.join(args.image_dir, image_filename)
        
        # --- THIS IS THE CORRECTED LOGIC ---
        mask_filename = image_filename.replace("case_", "seg_")
        mask_path = os.path.join(args.mask_dir, mask_filename)
        
        # Load image and mask
        image = Image.open(image_path).convert("L")
        true_mask = Image.open(mask_path)
        image_np = np.array(image)
        true_mask_np = np.array(true_mask)

        # Preprocess image for model
        input_tensor = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Perform inference
        with torch.no_grad():
            pred_logits = model(input_tensor)
            pred_softmax = torch.softmax(pred_logits, dim=1)
            pred_mask = torch.argmax(pred_softmax, dim=1).squeeze(0).cpu().numpy()
            
        # Save visualization
        output_filename = image_filename.replace("case_", "segmentation_").replace(".nii", "")
        output_path = os.path.join(args.output_dir, output_filename)
        visualize_segmentation(image_np, true_mask_np, pred_mask, output_path)

if __name__ == "__main__":
    main()