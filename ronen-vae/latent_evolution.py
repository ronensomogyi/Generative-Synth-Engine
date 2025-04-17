from nsynth_dataset import NsynthDataset
import torch
from model import VAE
import numpy as np
import os
from sklearn.decomposition import PCA




DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # Use MPS if available
LATENT_DIM = 2



def load_model(filepath, input_channels=1, input_dim=(128, 126)):
    """Load the VAE model with pre-trained weights."""

    vae = VAE(input_channels=input_channels, latent_dim=LATENT_DIM, input_dim=input_dim).to(DEVICE)
    vae.load_weights(filepath)
    return vae

if __name__ == "__main__":

    nsynth = NsynthDataset(path="/Volumes/ronen_usb/nsynth-train")

    epoch_latent_list = []

    for i in range(30):
        model_path = f"./weights/vae_epoch{i+1}.pth"
        vae = load_model(model_path, input_channels=1, input_dim=(128, 126))
        families_dict = nsynth.specs_per_family
        all_images = []
        all_labels = []

        latent_families_dict = {}

        for family, specs in families_dict.items():
            if len(specs) > 0:
                selected_indices = range(min(500, len(specs)))

                family_images = [specs[i] for i in selected_indices]

                family_images_tensor = torch.stack(family_images).to(torch.float32).to(DEVICE)
                with torch.no_grad():
                    _, latent_mu, _ = vae(family_images_tensor)
                    latent_families_dict[family] = latent_mu
        epoch_latent_list.append(latent_families_dict)

    import matplotlib.pyplot as plt

    # Create the directory to save plots if it doesn't exist
    output_dir = "./latent_evolution"
    os.makedirs(output_dir, exist_ok=True)

    # Function to plot latent space
    def plot_images_encoded_in_latent_space(latent_families_dict, epoch):
        plt.figure(figsize=(10, 10))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(latent_families_dict)))
        
        for (label, latent_points), color in zip(latent_families_dict.items(), colors):
            latent_points = latent_points.cpu().numpy()
            plt.scatter(latent_points[:, 0],
                        latent_points[:, 1],
                        label=label,
                        color=color,
                        alpha=0.5,
                        s=2)
        
        plt.legend()
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, f"latent_space_epoch_{epoch+1}.png"))
        plt.close()

    # Plot latent space for every 3 epochs
    for i, families_dict in enumerate(epoch_latent_list):
        if i % 3 == 0:
            plot_images_encoded_in_latent_space(families_dict, i)
            