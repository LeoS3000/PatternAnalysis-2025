# predict.py

import torch
import argparse
import os
import numpy as np
import nibabel as nib
import torchio as tio
from monai.inferers import sliding_window_inference


from src.models import UNet3D 
from src.utils import save_point_cloud_visualization

# Define the target size used during training for consistency
TARGET_SIZE = (128, 128, 128)

def predict(args):
    """
    Runs inference on a single 3D NIfTI image and saves the segmentation output.
    """
    # setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # load Model
    model = UNet3D(n_channels=1, n_classes=args.num_classes).to(device)
    
    print(f"=> Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval() # Set model to evaluation mode

    # load and preprocess
    print(f"Loading and preprocessing input image: {args.input_path}")
    
    # define the same basic transforms used for validation
    transforms = tio.Compose([
        tio.CropOrPad(TARGET_SIZE, padding_mode=0),
        # You may want to add tio.RescaleIntensity() or other normalization here
        # if your model expects it and it's not part of the network itself.
    ])

    # load the image using TorchIO, which also stores the affine matrix for saving
    subject = tio.Subject(
        mri=tio.ScalarImage(args.input_path)
    )
    subject = transforms(subject)
    input_tensor = subject.mri.data.float().unsqueeze(0).to(device, non_blocking=True)
    
    # run Inference
    print("Running sliding-window inference...")
    with torch.no_grad():
        # Use mixed precision for performance, matching the validation function
        with torch.cuda.amp.autocast(dtype=torch.float16):
            # Using sliding_window_inference is crucial for handling large images
            # that don't fit into memory at once. [cite: 126]
            prediction = sliding_window_inference(
                inputs=input_tensor,
                roi_size=TARGET_SIZE,
                sw_batch_size=1,
                predictor=model,
                overlap=0.25 # Overlap from training script
            )

    # --- Post-process and Save Output ---
    # Convert model output (logits) to a segmentation mask
    predicted_mask = torch.argmax(prediction.squeeze(0), dim=0).cpu().numpy().astype(np.uint8)
    
    print(f"Prediction generated with shape: {predicted_mask.shape}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Save as NIfTI file
    # We use the affine matrix from the original loaded image to preserve its orientation
    nifti_img = nib.Nifti1Image(predicted_mask, affine=subject.mri.affine)
    nifti_output_path = args.output_path + ".nii.gz"
    nib.save(nifti_img, nifti_output_path)
    print(f"Segmentation mask saved to: {nifti_output_path}")

    # 2. Save as 3D Point Cloud visualization
    ply_output_path = args.output_path + ".ply"
    save_point_cloud_visualization(predicted_mask, output_path=ply_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run 3D Segmentation Inference')
    parser.add_argument('--input_path', type=str, required=True, 
                        help='Path to the input NIfTI image file.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Base path for the output files (e.g., "results/patient_01_seg"). Suffixes (.nii.gz, .ply) will be added.')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/best_model.pth.tar',
                        help='Path to the saved model checkpoint.')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='Number of classes for the segmentation model.')
    
    args = parser.parse_args()
    predict(args)