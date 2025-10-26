# train.py
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import argparse
from tqdm import tqdm
import numpy as np

from model import VAE, vae_loss_function
from dataset import BrainScanDataset

def main(args):
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(os.path.join(args.output_dir, "reconstructions"), exist_ok=True)

    # --- 2. Data Loading ---
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(), # Normalizes to [0, 1]
    ])
    
    dataset = BrainScanDataset(root_dir=args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # --- 3. Model, Optimizer, Loss ---
    model = VAE(img_channels=1, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # --- 4. Training Loop ---
    print("Starting training...")
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in progress_bar:
            images = batch.to(device)
            
            optimizer.zero_grad()
            recon_images, mu, logvar = model(images)
            loss = vae_loss_function(recon_images, images, mu, logvar)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item() / len(images))
            
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{args.num_epochs} -- Average Loss: {avg_loss:.4f}")

        # Save some reconstruction samples
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                # Get a fixed batch of images for consistent visualization
                fixed_images = next(iter(dataloader)).to(device)
                recon_fixed, _, _ = model(fixed_images)
                
                comparison = torch.cat([fixed_images, recon_fixed])
                save_image(comparison.cpu(), 
                           os.path.join(args.output_dir, f"reconstructions/recon_epoch_{epoch+1}.png"), 
                           nrow=args.batch_size)

    # --- 5. Save Model and Latent Vectors ---
    print("Training finished. Saving model and latent vectors...")
    torch.save(model.state_dict(), os.path.join(args.output_dir, "vae_model.pth"))

    # Generate and save latent vectors for visualization
    model.eval()
    all_mu = []
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            mu, _ = model.encode(images)
            all_mu.append(mu.cpu().numpy())
    
    latent_vectors = np.concatenate(all_mu, axis=0)
    np.save(os.path.join(args.output_dir, "latent_vectors.npy"), latent_vectors)
    print("Latent vectors saved. You can now run visualize.py")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a VAE on brain scans")
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the directory of brain scan PNGs')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory to save results')
    parser.add_argument('--img-size', type=int, default=128, help='Size to resize images to (e.g., 128 for 128x128)')
    parser.add_argument('--latent-dim', type=int, default=32, help='Dimension of the latent space')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate for the Adam optimizer')
    
    args = parser.parse_args()
    main(args)