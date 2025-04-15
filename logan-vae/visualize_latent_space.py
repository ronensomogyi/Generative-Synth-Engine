# visualize_latent_space.py

import argparse
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

from train_vae_gan import Encoder, AudioDataset  # reuse your model and dataset

def extract_latents(encoder, dataloader, device):
    encoder.eval()
    latents = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            z, _, _ = encoder(batch)
            latents.append(z.cpu().numpy())
    return np.concatenate(latents, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Visualize the latent space of a trained VAE-GAN encoder.")
    parser.add_argument("model_path", type=str, help="Path to trained model file (with encoder)")
    parser.add_argument("audio_dir", type=str, help="Path to training audio files")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--output_plot", type=str, default="latent_space.png")
    args = parser.parse_args()

    # Load checkpoint
    checkpoint = torch.load(args.model_path)
    input_dim = checkpoint["input_dim"]
    latent_dim = checkpoint["latent_dim"]
    global_mean = checkpoint["global_mean"]
    global_std = checkpoint["global_std"]

    # Initialize encoder
    encoder = Encoder(input_dim, latent_dim)
    encoder.load_state_dict(checkpoint["encoder"])
    device = torch.device("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)

    # Load dataset
    dataset = AudioDataset(args.audio_dir, max_length=1000, global_mean=global_mean, global_std=global_std)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Extract latent vectors
    print("Extracting latent vectors...")
    latents = extract_latents(encoder, dataloader, device)
    latents = latents[:args.max_samples]

    # Apply t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, init="random", random_state=42)
    tsne_result = tsne.fit_transform(latents)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.6, s=10)
    plt.title("Latent Space (t-SNE Projection)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.savefig(args.output_plot)
    print(f"Plot saved to {args.output_plot}")

if __name__ == "__main__":
    main()
