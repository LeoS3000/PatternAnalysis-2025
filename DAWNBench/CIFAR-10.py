import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# --- 1. Setup and Data Preparation ---

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a Cutout augmentation class
class Cutout:
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones((h, w), dtype=torch.float32)
        y = torch.randint(h, (1,)).item()
        x = torch.randint(w, (1,)).item()

        y1 = torch.clamp(torch.tensor(y - self.length // 2), 0, h)
        y2 = torch.clamp(torch.tensor(y + self.length // 2), 0, h)
        x1 = torch.clamp(torch.tensor(x - self.length // 2), 0, w)
        x2 = torch.clamp(torch.tensor(x + self.length // 2), 0, w)

        mask[y1:y2, x1:x2] = 0.
        img = img * mask.unsqueeze(0)
        return img

# Define data transformations
# These are standard stats for CIFAR10
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
    Cutout(length=16) # Powerful augmentation
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

# Hyperparameters
BATCH_SIZE = 1024 # Use a large batch size for speed
NUM_WORKERS = 8 # Adjust based on your system's capabilities

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# --- 2. The Model Architecture (ResNet9) ---

def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

model = ResNet9().to(device)

# --- 3. Training and Evaluation ---

# Hyperparameters for training
EPOCHS = 30
MAX_LR = 0.01
WEIGHT_DECAY = 1e-4

# Loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, MAX_LR, epochs=EPOCHS, steps_per_epoch=len(train_loader))

# Automatic Mixed Precision (AMP) for speed
scaler = torch.amp.GradScaler('cuda')

def evaluate(model, loader):
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

# Training loop
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with AMP
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        # Backward pass with AMP
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update learning rate
        scheduler.step()

    # Evaluate at the end of the epoch
    val_accuracy = evaluate(model, test_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Validation Accuracy: {val_accuracy:.2f}%")

end_time = time.time()
print(f"Total Training Time: {(end_time - start_time)/60:.2f} minutes")

# Final check on test set
final_accuracy = evaluate(model, test_loader)
print(f"\nFinal Test Accuracy: {final_accuracy:.2f}%")