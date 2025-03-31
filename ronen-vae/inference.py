import numpy as np
import matplotlib.pyplot as plt
import torch
from model import VAE

def load_model(filepath, input_channels=1, latent_dim=20):
    """Load the VAE model with pre-trained weights."""
    vae = VAE(input_channels=input_channels, latent_dim=latent_dim)
    vae.load_weights(filepath)
    return vae

def infer(vae, input_tensor):
    """Perform inference using the VAE model."""
    vae.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        reconstructed, mu, sigma = vae(input_tensor)
    return reconstructed, mu, sigma

def select_images(images, labels, num_images=10):
    """Select a random subset of images and their corresponding labels."""
    sample_images_index = np.random.choice(range(len(images)), num_images, replace=False)
    sample_images = images[sample_images_index]
    sample_labels = labels[sample_images_index]
    return sample_images, sample_labels

def plot_reconstructed_images(images, reconstructed_images):
    """Plot original and reconstructed images side by side."""
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
    plt.show()

def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
    """Visualize the latent space with labels."""
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],
                latent_representations[:, 1], 
                cmap="rainbow",
                c=sample_labels, 
                alpha=0.5, 
                s=2)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    # Example usage
    from torchvision import datasets, transforms

    # Load MNIST dataset
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root="dataset/", train=False, transform=transform, download=True)
    images, labels = mnist.data.numpy(), mnist.targets.numpy()

    # Load pre-trained VAE
    model_path = "./weights/vae_weights.pth"
    vae = VAE(input_channels=1, latent_dim=20)
    vae.load_weights(model_path)

    # Select a subset of images
    sample_images, sample_labels = select_images(images, labels, num_images=10)

    # Convert images to PyTorch tensors and normalize
    sample_images_tensor = torch.tensor(sample_images, dtype=torch.float32).unsqueeze(1) / 255.0

    # Perform inference
    vae.eval()
    with torch.no_grad():
        reconstructed_images, latent_mu, _ = vae(sample_images_tensor)

    # Plot original and reconstructed images
    plot_reconstructed_images(sample_images, reconstructed_images.cpu().numpy())

    # Visualize latent space
    with torch.no_grad():
        _, latent_mu, _ = vae(torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0)
    plot_images_encoded_in_latent_space(latent_mu.cpu().numpy(), labels)
