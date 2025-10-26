# src/dataset.py

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import numpy as np
import nibabel as nib
import torchio as tio

class ProstateNiftiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, filenames, num_classes, transforms=None):
        """
        Args:
            image_dir (str): Base directory with all the 3D Nifti input images.
            mask_dir (str): Base directory with all the 3D Nifti segmentation masks.
            filenames (list): A list of filenames to be included in this dataset instance.
            num_classes (int): Number of classes for one-hot encoding.
            transforms (callable, optional): Optional 3D transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.images = filenames
        self.transforms = transforms
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.nii', '.nii.gz'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_filename = self.images[index]
        image_path = os.path.join(self.image_dir, image_filename)
        mask_filename = image_filename.replace("_LFOV.nii.gz", "_SEMANTIC.nii.gz")
        mask_path = os.path.join(self.mask_dir, mask_filename)
        # --- THIS SECTION IS MODIFIED FOR TORCHIO ---

        # 1. Create a TorchIO Subject
        #    We add the affine matrix to ensure transforms are applied correctly in physical space
        subject = tio.Subject(
            mri=tio.ScalarImage(image_path),
            mask=tio.LabelMap(mask_path),
        )

        # 2. Apply transformations if they exist
        if self.transforms:
            subject = self.transforms(subject)

        # 3. Extract the transformed tensors
        #    TorchIO returns tensors in [C, D, H, W] format already
        image_tensor = subject.mri.data.float()
        mask_tensor = subject.mask.data.squeeze(0).long() # Squeeze channel dim and ensure it's Long

        # --- END OF MODIFIED SECTION ---

        # One-Hot Encode the Mask (this now happens *after* augmentation)
        mask_one_hot = F.one_hot(mask_tensor, num_classes=self.num_classes)
        mask_one_hot = mask_one_hot.permute(3, 0, 1, 2).float()

        return image_tensor, mask_one_hot