# src/utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

def dice_loss(pred, target, smooth=1.):
    """
    Calculates the Dice Loss for 2D or 3D segmentation.
    Args:
        pred (torch.Tensor): The model's raw output logits (B, C, ...).
        target (torch.Tensor): The one-hot encoded ground truth mask (B, C, ...).
        smooth (float): A smoothing factor to avoid division by zero.
    Returns:
        torch.Tensor: The calculated Dice loss.
    """
    pred = F.softmax(pred, dim=1)
    # Flatten all dimensions except batch and channel
    pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
    target = target.contiguous().view(target.shape[0], target.shape[1], -1)
    
    intersection = (pred * target).sum(dim=2)
    dice_coeff = (2. * intersection + smooth) / (pred.sum(dim=2) + target.sum(dim=2) + smooth)
    dice_coeff = dice_coeff.mean()  # mean over batch and classes

    return 1 - dice_coeff

def dice_score(pred, target, smooth=1.):
    """
    Calculates the Dice Similarity Coefficient for each class for 2D or 3D segmentation.
    Args:
        pred (torch.Tensor): The model's raw output logits (B, C, ...).
        target (torch.Tensor): The one-hot encoded ground truth mask (B, C, ...).
        smooth (float): A smoothing factor.
    Returns:
        list: A list of DSC scores for each class.
    """
    pred = F.softmax(pred, dim=1)
    # Convert prediction to a hard, one-hot encoded mask
    pred_mask = torch.argmax(pred, dim=1)
    pred_one_hot = F.one_hot(pred_mask, num_classes=target.shape[1]).permute(0, 4, 1, 2, 3) if pred_mask.ndim == 4 else F.one_hot(pred_mask, num_classes=target.shape[1]).permute(0, 3, 1, 2)
    pred_one_hot = pred_one_hot.float()

    dice_per_class = []
    num_classes = target.shape[1]
    # Flatten all dimensions except batch and channel
    for i in range(num_classes):
        pred_c = pred_one_hot[:, i].contiguous().view(pred_one_hot.shape[0], -1)
        target_c = target[:, i].contiguous().view(target.shape[0], -1)
        intersection = (pred_c * target_c).sum(dim=1)
        score = (2. * intersection + smooth) / (pred_c.sum(dim=1) + target_c.sum(dim=1) + smooth)
        dice_per_class.append(score.mean().item())  # mean over batch

    return dice_per_class

def save_checkpoint(state, directory="checkpoints", filename="my_checkpoint.pth.tar"):
    """Saves model and optimizer state."""
    print("=> Saving checkpoint")
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    torch.save(state, filepath)

def load_checkpoint(checkpoint_path, model, optimizer):
    """Loads model and optimizer state from a checkpoint file."""
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def visualize_segmentation(image, true_mask, pred_mask, output_path="results/segmentation.png"):
    """
    Saves a side-by-side visualization of the image, ground truth, and prediction.
    """
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    vis_scaling_factor = 60
    
    true_mask_display = true_mask.astype(np.float32)
    pred_mask_display = pred_mask.astype(np.float32) * vis_scaling_factor
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original MR Image')
    axes[0].axis('off')
    
    # Use the new display-ready variables
    vmax_val = true_mask_display.max()
    axes[1].imshow(true_mask_display, cmap='jet', vmin=0, vmax=vmax_val)
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask_display, cmap='jet', vmin=0, vmax=vmax_val)
    axes[2].set_title('Predicted Segmentation')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Visualization saved to {output_path}")