import numpy as np
import matplotlib.pyplot as plt
import torch
from model import VAE
from nsynth_dataset import NsynthDataset
import sounddevice as sd
import torchaudio


LATENT_DIM = 2

def load_model(filepath, input_channels=1, input_dim=(128, 126)):
    """Load the VAE model with pre-trained weights."""

    vae = VAE(input_channels=input_channels, latent_dim=LATENT_DIM, input_dim=input_dim)
    vae.load_weights(filepath)
    return vae

def infer(vae, input_tensor):
    """Perform inference using the VAE model."""
    vae.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        reconstructed, mu, sigma = vae(input_tensor)
    return reconstructed, mu, sigma


def select_samples(dataset, num_samples=10):
    """Select a random subset of samples from the dataset."""
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    samples = [dataset[i] for i in indices]
    images = torch.stack([sample[0] for sample in samples])  # Stack spectrograms
    labels = [sample[1] for sample in samples]  # Collect labels
    return images, labels


def plot_reconstructed_images(images, reconstructed_images):
    """Plot original and reconstructed images side by side."""
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image.numpy(), aspect="auto", origin="lower", cmap="viridis")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image.numpy(), aspect="auto", origin="lower", cmap="viridis")
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

def sample_and_play_from_latent_space(vae, num_samples=5, latent_dim=LATENT_DIM, sample_rate=16000):
    """Sample random points from the latent space, decode them, and play the resulting spectrograms."""
    vae.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        # Sample random points from a standard normal distribution
        random_latent_vectors = torch.randn(num_samples, LATENT_DIM).to("cpu")
        # Decode the latent vectors into spectrograms
        decoded_spectrograms = vae.decode(random_latent_vectors)
    
    # Convert decoded spectrograms back to waveforms and play them
    mel_to_stft = torchaudio.transforms.InverseMelScale(
        n_stft=1025, n_mels=128, sample_rate=sample_rate
    )
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=2048, hop_length=512)

    for i, spectrogram in enumerate(decoded_spectrograms):
        # Inverse log-mel spectrogram
        mel_spec = torch.exp(spectrogram) - 1e-9
        # Convert Mel spectrogram to STFT
        stft_spec = mel_to_stft(mel_spec)
        # Use Griffin-Lim to reconstruct the waveform
        waveform = griffin_lim(stft_spec)
        print(f"Playing sample {i + 1}/{num_samples}...")
        sd.play(waveform.squeeze().numpy(), samplerate=sample_rate)
        sd.wait()  # Wait until the sound finishes playing

if __name__ == "__main__":
    # Load NSynth dataset
    nsynth = NsynthDataset(path="/Volumes/ronen_usb/nsynth-train")

    # Load pre-trained VAE
    model_path = "./weights/vae_weights.pth"
    vae = load_model(model_path, input_channels=1, input_dim=(128, 126))

    # Select a subset of samples
    sample_images, sample_labels = select_samples(nsynth, num_samples=10)

    # Perform inference
    sample_images_tensor = sample_images.to(torch.float32).to("cpu")  # Ensure correct dtype
    vae.eval()
    with torch.no_grad():
        reconstructed_images, latent_mu, _ = vae(sample_images_tensor)

    # Plot original and reconstructed images
    plot_reconstructed_images(sample_images, reconstructed_images.cpu())

    # print("visualize latent space")
    # # Visualize latent space
    # all_images = torch.stack([nsynth[i][0] for i in range(len(nsynth))]).to(torch.float32)
    # all_labels = [nsynth[i][1] for i in range(len(nsynth))]
    # print("visualize latent space")
    # with torch.no_grad():
    #     _, latent_mu, _ = vae(all_images)
    # plot_images_encoded_in_latent_space(latent_mu.cpu().numpy(), all_labels)

    # Sample and play from the latent space
    sample_and_play_from_latent_space(vae, num_samples=5, latent_dim=LATENT_DIM, sample_rate=16000)
