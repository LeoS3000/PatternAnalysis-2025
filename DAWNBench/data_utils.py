# data_utils.py

import torch
import torchvision
import torchvision.transforms as transforms

class Cutout:
    """A Cutout augmentation that randomly masks a square region of an image."""
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones((h, w), dtype=torch.float32)
        y = torch.randint(h, (1,)).item()
        x = torch.randint(w, (1,)).item()

        y1 = torch.clamp(torch.tensor(y - self.length // 2), 0, h).int()
        y2 = torch.clamp(torch.tensor(y + self.length // 2), 0, h).int()
        x1 = torch.clamp(torch.tensor(x - self.length // 2), 0, w).int()
        x2 = torch.clamp(torch.tensor(x + self.length // 2), 0, w).int()

        mask[y1:y2, x1:x2] = 0.
        img = img * mask.unsqueeze(0)
        return img

def get_cifar10_loaders(train_batch_size, num_workers=4, data_dir='./data'):
    """Returns train and test DataLoaders for CIFAR-10."""
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
        Cutout(length=16)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    # Use a larger batch size for evaluation as it doesn't require gradients
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=train_batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader