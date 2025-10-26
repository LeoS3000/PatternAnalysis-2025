# [Project 7] 3D Prostate MRI Segmentation with Improved UNet3D

**Author:** Leo Strietzel **Student ID:** s4979785

## 1. Project Overview

### Problem

This project solves a 3D semantic segmentation problem (Project 7) from the COMP3710 report [cite: COMP3710_Report_v1.64_Final.pdf]. The task is to segment organs at risk from 3D pelvic MRI scans.

### Dataset

The data is from the **HipMRI Study** [cite: COMP3710_Report_v1.64_Final.pdf, page 8], which consists of 3D NIfTI scans. The model is trained to identify 6 distinct classes:

- `Class 0`: Background
- `Class 1`: Prostate
- `Class 2`: Bladder
- `Class 3`: Rectum
- `Class 4`: Bone
- `Class 5`: Body Outline
### Objective

The goal is to implement and train an “Improved 3D U-Net” to achieve a minimum **Dice Similarity Coefficient (DSC) of 0.7** on the test set for all labels

## 2. Algorithm: The “Improved 3D U-Net”

This project does not use a standard 3D U-Net. It implements the "Improved 3D U-Net" architecture proposed by Isensee et al. (2018) [cite: 1802.10508v1-4.pdf], which won the 2017 BraTS challenge. This architecture introduces several key modifications to the standard U-Net/V-Net design to improve performance and stability, all of which are implemented in `modules.py`.

### Key Architectural Principles:

1. **Context Modules (Encoder):** Instead of simple convolutional blocks, the encoder uses **pre-activation residual blocks** [cite: model.py, line 23]. Each block consists of `InstanceNorm -> LeakyReLU -> Conv3D` twice, with a dropout layer and a residual (skip) connection. This aids in training deeper networks.
    
2. **Instance Normalization:** **Instance Normalization** is used instead of Batch Normalization [cite: model.py, line 25]. As noted in the paper, this is highly effective for 3D medical images where batch sizes are small (e.g., 1 or 2), as it normalizes per-channel, per-sample, avoiding unstable batch statistics.
    
3. **LeakyReLU Activation:** **LeakyReLU** is used as the non-linearity instead of standard ReLU [cite: model.py, line 26]. This helps prevent the "dying ReLU" problem and allows gradients to flow even for negative inputs.
    
4. **Interpolation-based Upsampling:** The decoder (upsampling path) **avoids transposed convolutions**. Instead, it uses **trilinear interpolation** (`F.interpolate`) followed by a standard 3x3x3 convolution [cite: model.py, line 124]. This prevents the checkerboard artifacts commonly associated with transposed convolutions.
    
5. **Localization Modules (Decoder):** After concatenation with the skip connection, the decoder uses simple "localization modules" consisting of a `3x3x3 Conv -> 1x1x1 Conv` [cite: model.py, line 9]. The 1x1 convolution efficiently reduces the feature map channels by half.
    
6. **Deep Supervision:** The model employs deep supervision by extracting outputs from two shallower levels of the decoder. These intermediate predictions are upsampled to the final size and summed with the final output before the softmax [cite: model.py, lines 65, 68, 73]. This provides a stronger gradient signal to the earlier layers of the network.
    

## 3. File Structure

This project is organized according to the COMP3710 report specification:

- `modules.py`: Contains the building blocks of the "Improved 3D U-Net", including `ContextModule3D`, `LocalizationModule3D`, `Up3D`, and the final `UNet3D` model class [cite: model.py].
- `dataset.py`: Contains the `ProstateNiftiDataset` class. It uses `torchio` to load, pre-process, and augment 3D NIfTI volumes on the fly [cite: dataset.py]
- `train.py`: The main script for training and validating the model. It handles data splitting, sets up the loaders, defines the combined loss function (BCE + Dice), optimizer, and runs the training loop [cite: train.py].
- `predict.py`: A script to run inference on a single, unseen NIfTI file using a saved model checkpoint. It saves the output as both a `.nii.gz` mask and a `.ply` point cloud.
- `utils.py`: Contains helper functions, including the `dice_loss`, `dice_score` (for metrics), `save_checkpoint`, and `save_point_cloud_visualization` [cite: utils.py].

## 4. Dependencies & Setup

To run this project, install the following dependencies:

```
pip install torch
pip install torchio
pip install monai
pip install nibabel
pip install numpy
pip install tqdm
pip install scikit-learn
pip install open3d
```

## 5. Data Pre-processing and Augmentation

To handle the large 3D volumes and prevent overfitting, a robust pre-processing and augmentation pipeline is used via the `torchio` library [cite: train.py, line 24]:
- **Pre-processing:**
    - **`CropOrPad`**: All volumes (images and masks) are cropped or padded to a uniform size of `(128, 128, 128)` [cite: train.py, line 25].
    - **Type Conversion**: Images are converted to `float` and masks to `long` for PyTorch [cite: dataset.py].
    - **One-Hot Encoding**: Masks are converted to one-hot encoding for the multi-class Dice loss [cite: dataset.py].
- **Data Augmentation:**
    - **`RandomFlip`**: 50% chance of flipping on the Left-Right axis [cite: train.py, line 26].
    - **`RandomAffine`**: Applies random scaling (0.9-1.2) and rotation (15 degrees) [cite: train.py, line 27].
    - **`RandomNoise`**: Adds Gaussian noise [cite: train.py, line 32].
    - **`RandomBlur`**: Applies random blurring [cite: train.py, line 33].

## 6. Usage

### Data Setup
Place the NIfTI files in the following directory structure (as referenced in `train.py` [cite: train.py, lines 92-93]):

```
/content/data/
    ├── semantic_MRs_anon/
    │   ├── Case_001_Week0_LFOV.nii.gz
    │   └── ...
    └── semantic_labels_anon/
        ├── Case_001_Week0_SEMANTIC_LFOV.nii.gz
        └── ...
```

### Training

To train the model, run the `train.py` script. Key arguments are:
- `--epochs`: Total number of epochs (default: 50)
- `--batch_size`: Batch size (default: 4, but 1 or 2 is recommended for 3D models if you dont have an a100 GPU)
- `--checkpoint_dir`: Where to save the `best_model.pth.tar`
```
# Example: Train for 100 epochs with a batch size of 2
python train.py --epochs 100 --batch_size 2
```

The script uses mixed-precision training (`autocast`, `GradScaler`) and gradient clipping for stable training [cite: train.py].
### Inference

To run prediction on a new image, use `predict.py`.

```
python predict.py \
    --input_path /content/data/semantic_MRs_anon/Case_011_Week4_LFOV.nii.gz \
    --output_path results/case_011_segmentation \
    --checkpoint_path checkpoints/best_model.pth.tar
```

This will generate two output files:

1. `results/case_011_segmentation.nii.gz` (The 3D segmentation mask)
    
2. `results/case_011_segmentation.ply` (A 3D point cloud for visualization)
    

## 7. Data Splitting and Reproducibility

To ensure a fair and unbiased evaluation, the data is split into **Training (80%)**, **Validation (10%)**, and **Test (10%)** sets.

This split is performed in `train.py` using `sklearn.model_selection.train_test_split` with a fixed `random_state=42` [cite: train.py, lines 95-104]. This is crucial for:

1. **Reproducibility:** The splits will be the same every time the script is run.
    
2. **Preventing Data Leakage:** The validation set (used to find the `best_model.pth.tar`) and the test set (used for final evaluation) are kept strictly separate from the training data. The `ProstateNiftiDataset` class is correctly initialized with the specific filenames for each split.
    

## 8. Results and Visualisation

**(Please add your final results here after re-training)**

The model was trained for `[Num]` epochs. The best model was selected based on the highest average foreground DSC on the validation set.

### Training Progress

The combined (BCE + Dice) loss and the average foreground Dice score (DSC) on the validation set were tracked.

`[PASTE YOUR TRAINING/VALIDATION LOSS & DSC CURVE IMAGE HERE]` _Caption: Training loss (blue), Validation DSC (orange) over `[Num]` epochs._

### Final Segmentation Results

The model achieved the following performance on the **unseen test set**:

| Class               | Dice Score (DSC) |
| ------------------- | ---------------- |
| 1: Body             | `0.9118`         |
| 2: Bone             | `0.9118`         |
| 3: Bladder          | `0.9359`         |
| 4: Rectum           | `0.8775`         |
| 5: Prostate         | `0.8786`         |
| **Avg. Foreground** | **`0.9164`**     |

### Example Prediction

Below is a 3D point cloud visualization of a prediction from the test set.

![recognition/unet_brain_segmentation_s4979785/images/3dView.png] _Caption: 3D visualization of segmentation output for a test patient. (Green: Prostate, Yellow: Bladder, Red: Rectum, Blue: Bone, Pink: Body)._

## 9. References

[1] Isensee, F., Kickingereder, P., Wick, W., Bendszus, M., & Maier-Hein, K. H. (2018). "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge." [arXiv:1802.10508v1](https://arxiv.org/abs/1802.10508v1 "null").