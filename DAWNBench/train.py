# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse

# Import our custom modules
from model import ResNet18
from data_utils import get_cifar10_loaders

def evaluate(model, loader, device):
    """Evaluates the model's accuracy on the given data loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training with SGD')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--warmup_epochs', default=5, type=int, help='number of epochs for linear warmup')
    parser.add_argument('--batch_size', default=512, type=int, help='mini-batch size')
    parser.add_argument('--base_lr', default=0.1, type=float, help='base learning rate for batch size 256')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('--num_workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--save_path', default='resnet18_cifar10.pth', type=str, help='path to save the final model')
    args = parser.parse_args()

    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data ---
    train_loader, test_loader = get_cifar10_loaders(args.batch_size, args.num_workers)

    # --- Model ---
    model = ResNet18().to(device)

    # --- Loss, Optimizer, Scheduler & AMP ---
    criterion = nn.CrossEntropyLoss()
    
    # Linearly scale learning rate with batch size
    learning_rate = args.base_lr * (args.batch_size / 256)
    print(f"Using scaled learning rate: {learning_rate:.4f}")

    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # Scheduler with linear warm-up and cosine annealing
    warmup_iters = args.warmup_epochs * len(train_loader)
    total_iters = (args.epochs - args.warmup_epochs) * len(train_loader)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_iters
    )
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_iters, eta_min=1e-5
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_iters]
    )

    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    # --- Training Loop ---
    start_time = time.time()
    print("Starting training...")

    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        val_accuracy = evaluate(model, test_loader, device)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}, Val Acc: {val_accuracy:.2f}%, LR: {current_lr:.5f}")

    end_time = time.time()
    print(f"\nTotal Training Time: {(end_time - start_time)/60:.2f} minutes")

    final_accuracy = evaluate(model, test_loader, device)
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")

    print(f"\nSaving model to {args.save_path}...")
    torch.save(model.state_dict(), args.save_path)
    print("Model saved successfully!")

if __name__ == '__main__':
    main()