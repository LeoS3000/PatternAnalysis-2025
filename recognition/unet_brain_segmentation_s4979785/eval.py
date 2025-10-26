import torch
from torch.utils.data import DataLoader
import argparse
import os
import json
from tqdm import tqdm
import torchio as tio
import numpy as np
import open3d as o3d


from src.dataset import ProstateNiftiDataset
from src.models import UNet3D


def dice_score(pred, target, smooth=1.):
    """
    Calculates the Dice Similarity Coefficient for each class.
    """
    pred = torch.softmax(pred, dim=1)
    # Convert prediction to a hard, one-hot encoded mask
    pred_mask = torch.argmax(pred, dim=1)
    pred_one_hot = torch.nn.functional.one_hot(pred_mask, num_classes=target.shape[1]).permute(0, 4, 1, 2, 3).float()

    dice_per_class = []
    # Iterate over each class (channel)
    for i in range(target.shape[1]):
        pred_c = pred_one_hot[:, i, ...].contiguous().view(-1)
        target_c = target[:, i, ...].contiguous().view(-1)

        intersection = (pred_c * target_c).sum()
        score = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
        dice_per_class.append(score.item())

    return dice_per_class

def save_point_cloud_visualization(pred_mask, output_path="results/segmentation.ply"):
    """
    Saves a 3D segmentation mask as a colored point cloud.
    """
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define a color map for the classes (RGB, values 0-1)
    color_map = {
        1: [0, 1, 0],   # Prostate
        2: [1, 1, 0],   # Bladder
        3: [1, 0, 0],   # Rectum
        4: [0, 0, 1],   # Bone
        5: [1, 0, 1],   # Body Outline
    }

    all_points = []
    all_colors = []

    for class_id, color in color_map.items():
        points = np.argwhere(pred_mask == class_id)
        if points.size > 0:
            colors = np.tile(color, (points.shape[0], 1))
            all_points.append(points)
            all_colors.append(colors)

    if not all_points:
        print(f"No foreground objects found in {os.path.basename(output_path)}.")
        return

    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Point cloud visualization saved to {output_path}")

#  Main Evaluation Function 
def main(args):
    #  Setup 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    #  Load Test Set Filenames 
    with open(args.test_files_json, 'r') as f:
        test_files = json.load(f)
    print(f"Found {len(test_files)} files in the test set.")

    #  Data Loading 
    img_dir = os.path.join(args.data_dir, "semantic_MRs_anon")
    mask_dir = os.path.join(args.data_dir, "semantic_labels_anon")

    # Define preprocessing pipeline (resizing, no augmentation)
    test_transforms = tio.Compose([
        tio.CropOrPad((128, 128, 128), padding_mode=0),
    ])

    test_dataset = ProstateNiftiDataset(
        image_dir=img_dir,
        mask_dir=mask_dir,
        filenames=test_files,
        num_classes=args.num_classes,
        transforms=test_transforms
    )
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2, pin_memory=True)

    #  Load Model 
    model = UNet3D(n_channels=1, n_classes=args.num_classes).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print("Model loaded successfully from checkpoint.")

    #  Run Evaluation 
    model.eval()
    all_dsc_scores = []
    loop = tqdm(test_loader, leave=True, desc="Evaluating Test Set")

    with torch.no_grad():
        for i, (x, y) in enumerate(loop):
            x = x.to(device)
            y = y.to(device)

            # Sliding window inference can be more robust for 3D
            # but direct inference is fine if images are resized.
            preds_logits = model(x)
            scores = dice_score(preds_logits, y)
            all_dsc_scores.append(scores)

            # Visualize the first few samples
            if i < args.num_visualizations:
                pred_mask = torch.argmax(torch.softmax(preds_logits, dim=1), dim=1).squeeze(0).cpu().numpy()
                output_filename = f"point_cloud_{test_files[i].replace('.nii.gz', '.ply')}"
                output_path = os.path.join(args.output_dir, output_filename)
                save_point_cloud_visualization(pred_mask, output_path)

    #  Report Final Scores 
    avg_dsc_per_class = torch.tensor(all_dsc_scores).mean(axis=0).tolist()
    avg_foreground_dsc = sum(avg_dsc_per_class[1:]) / (args.num_classes - 1)

    print("\n Final Test Set Results ")
    print(f"Dice Score (per class): {[round(s, 4) for s in avg_dsc_per_class]}")
    print(f"Average Foreground Dice Score: {avg_foreground_dsc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the 3D U-Net model on a test set.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the best_model.pth.tar file')
    parser.add_argument('--data-dir', type=str, required=True, help='Base directory for the HipMRI data')
    parser.add_argument('--test-files-json', type=str, required=True, help='Path to the JSON file containing the list of test filenames')
    parser.add_argument('--output-dir', type=str, default='./results', help='Directory to save visualization results')
    parser.add_argument('--num-classes', type=int, default=6, help='Number of segmentation classes')
    parser.add_argument('--num-visualizations', type=int, default=3, help='Number of point cloud visualizations to save')
    args = parser.parse_args()
    main(args)