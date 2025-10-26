# visualize.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
import argparse
import os

from model import VAE
from torchvision.utils import make_grid

def plot_umap(latent_vectors, save_path):
    """Generates and saves a UMAP plot of the latent vectors."""
    print("Running UMAP on latent vectors...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
    embedding = reducer.fit_transform(latent_vectors)
    
    plt.figure(figsize=(12, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.7)
    plt.title('UMAP Projection of the Latent Space')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"UMAP plot saved to {save_path}")

def plot_manifold(model, device, save_path, grid_size=20, img_size=128):
    """Generates a 2D manifold of generated images."""
    print("Generating 2D manifold...")
    model.eval()
    
    # Create a grid of points in the latent space
    # We sample from a normal distribution to match the learned latent space
    z_samples = torch.randn(grid_size * grid_size, model.fc_mu.out_features).to(device)
    
    with torch.no_grad():
        generated_images = model.decode(z_samples).cpu()
    
    # Create a grid of images
    grid = make_grid(generated_images, nrow=grid_size, padding=2, normalize=True)
    
    plt.figure(figsize=(15, 15))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title('Generated Manifold from Latent Space')
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()
    print(f"Manifold image saved to {save_path}")

def main(args):
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. Load Latent Vectors ---
    latent_path = os.path.join(args.output_dir, "latent_vectors.npy")
    if not os.path.exists(latent_path):
        print(f"Error: Latent vectors not found at {latent_path}. Run train.py first.")
        return
    latent_vectors = np.load(latent_path)

    # --- 3. Generate UMAP Plot ---
    plot_umap(latent_vectors, os.path.join(args.output_dir, "umap_manifold.png"))
    
    # --- 4. Load Model and Generate Manifold ---
    model_path = os.path.join(args.output_dir, "vae_model.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run train.py first.")
        return
        
    model = VAE(img_channels=1, latent_dim=args.latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    plot_manifold(model, device, os.path.join(args.output_dir, "generative_manifold.png"), img_size=args.img_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize VAE results")
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory where results are saved')
    parser.add_argument('--latent-dim', type=int, default=32, help='Dimension of the latent space (must match training)')
    parser.add_argument('--img-size', type=int, default=128, help='Image size used during training')

    args = parser.parse_args()
    main(args)