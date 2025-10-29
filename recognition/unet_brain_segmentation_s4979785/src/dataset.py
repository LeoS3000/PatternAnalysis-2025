# src/dataset.py

from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import torchio as tio

class ProstateNiftiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, filenames, num_classes, transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.images = filenames  # Use the passed list of filenames
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_filename = self.images[index]
        image_path = os.path.join(self.image_dir, image_filename)
        mask_filename = image_filename.replace("_LFOV.nii.gz", "_SEMANTIC_LFOV.nii.gz")
        mask_path = os.path.join(self.mask_dir, mask_filename)

        # add the affine matrix to ensure transforms are applied correctly in physical space
        subject = tio.Subject(
            mri=tio.ScalarImage(image_path),
            mask=tio.LabelMap(mask_path),
        )

        # apply transformations if they exist
        if self.transforms:
            subject = self.transforms(subject)

        # extract the transformed tensors
        image_tensor = subject.mri.data.float()
        mask_tensor = subject.mask.data.squeeze(0).long() # Squeeze channel dim and ensure it's Long

        # o-h encode the mask
        mask_one_hot = F.one_hot(mask_tensor, num_classes=self.num_classes)
        mask_one_hot = mask_one_hot.permute(3, 0, 1, 2).float()

        return image_tensor, mask_one_hot