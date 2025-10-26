# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm


# Import your custom modules from the 'src' directory
from sklearn.model_selection import train_test_split
from src.dataset import ProstateNiftiDataset
from src.model import UNet3D
from src.utils import dice_loss, dice_score, save_checkpoint

import torchio as tio

TARGET_SIZE = (128, 128, 128)
# Define the 3D augmentation pipeline
transforms = tio.Compose([
    tio.CropOrPad(TARGET_SIZE, padding_mode=0),
    tio.RandomFlip(axes=('LR',)),          # Randomly flip left-right with a 50% chance
    tio.RandomAffine(
        scales=(0.9, 1.2),                 # Randomly scale the image
        degrees=15,                        # Randomly rotate by up to 15 degrees
        isotropic=True,
    ),
    tio.RandomNoise(std=0.01),             # Add a bit of random noise
    tio.RandomBlur(std=(0, 1)),            # Apply a random blur
])

val_transforms = tio.Compose([
    tio.CropOrPad(TARGET_SIZE, padding_mode=0), # <-- ADD THIS LINE
])

def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    """Runs one full epoch of training."""
    loop = tqdm(loader, leave=True)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass with Automatic Mixed Precision
        with torch.amp.autocast(device_type='cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())

def validate_model(loader, model, device, num_classes):
    """Validates the model and returns DSC scores."""
    model.eval()
    all_dsc_scores = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            scores = dice_score(preds, y)
            all_dsc_scores.append(scores)
    
    # Calculate average DSC for each class across all batches
    avg_dsc_per_class = torch.tensor(all_dsc_scores).mean(axis=0).tolist()
    
    model.train() # Set model back to training mode
    return avg_dsc_per_class

def main(args):
    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Data Loading ---
    base_img_dir = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
    base_mask_dir = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"

    # Create a validation split
    all_filenames = sorted([f for f in os.listdir(base_img_dir) if f.endswith(('.nii', '.nii.gz'))])
    train_files, val_files = train_test_split(
        all_filenames,
        test_size=0.2,      # 20% of the data will be for validation
        random_state=42     # The "seed" for the random shuffle
    )

    NUM_CLASSES = 6

    train_dataset = ProstateNiftiDataset(
        image_dir=base_img_dir,
        mask_dir=base_mask_dir,
        filenames=train_files,  # <-- Pass the list of training files
        num_classes=NUM_CLASSES,
        transforms=transforms
    )

    val_dataset = ProstateNiftiDataset(
        image_dir=base_img_dir,
        mask_dir=base_mask_dir,
        filenames=val_files,    # <-- Pass the list of validation files
        num_classes=NUM_CLASSES,
        transforms=val_transforms
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    # Model, Loss, Optimizer 
    model = UNet3D(n_channels=1, n_classes=args.num_classes).to(device)
    bce_loss = nn.CrossEntropyLoss()
    
    def combined_loss(pred, target):
        return bce_loss(pred, target) + dice_loss(pred, target)


    # Optimizer and LR Scheduler
    initial_lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 0.985 ** epoch
    )
    scaler = torch.cuda.amp.GradScaler()

    # Training Loop
    best_dsc = 0.0
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        train_one_epoch(train_loader, model, optimizer, combined_loss, scaler, device)
        
        # Step the learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current learning rate: {current_lr:.6f}")
        
        # Validation
        dsc_scores = validate_model(val_loader, model, device, args.num_classes)
        avg_dsc = sum(dsc_scores[1:]) / (args.num_classes - 1) # Avg DSC of foreground classes
        
        print(f"Validation DSC (per class): {[round(s, 4) for s in dsc_scores]}")
        print(f"Average Foreground DSC: {avg_dsc:.4f}")

        # Save model if its the best one so far
        if avg_dsc > best_dsc:
            best_dsc = avg_dsc
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, directory=args.checkpoint_dir, filename="best_model.pth.tar")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train UNet for Brain Segmentation')
    # ... all your parser.add_argument lines ...
    # Example arguments (add your actual arguments as needed):
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    # parser.add_argument('--lr', type=float, default=5e-4)  # Not needed, now hardcoded
    args = parser.parse_args()
    main(args)