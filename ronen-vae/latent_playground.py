import numpy as np
import matplotlib.pyplot as plt
import torch
import sounddevice as sd
import torchaudio
from model import VAE


import os

def load_model(filepath, input_channels=1, latent_dim=2, input_dim=(128, 126)):
    """Load the VAE model with pre-trained weights."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Model weights file not found at: {filepath}")
    vae = VAE(input_channels=input_channels, latent_dim=latent_dim, input_dim=input_dim)
    vae.load_weights(filepath)
    return vae


def decode_and_play(vae, latent_vector, sample_rate=16000):
    """Decode a latent vector, convert to waveform, and play it."""
    vae.eval()
    with torch.no_grad():
        # Decode the latent vector into a spectrogram
        spectrogram = vae.decode(latent_vector.unsqueeze(0)).squeeze(0)
    
    # Convert the spectrogram back to a waveform
    mel_to_stft = torchaudio.transforms.InverseMelScale(
        n_stft=1025, n_mels=128, sample_rate=sample_rate
    )
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=2048, hop_length=512)
    mel_spec = torch.exp(spectrogram) - 1e-9  # Inverse log-mel spectrogram
    stft_spec = mel_to_stft(mel_spec)
    waveform = griffin_lim(stft_spec)

    # Play the waveform
    sd.play(waveform.squeeze().numpy(), samplerate=sample_rate)
    sd.wait()


def visualize_latent_space(vae, latent_dim=20, sample_rate=16000):
    """Visualize a 64x64 color-coded latent space grid.
       Click a square to select a latent point, e.g., (31,40), and decode the corresponding waveform.
    """
    grid_size = 64
    # Create a 64x64 grid with a simple gradient for visualization
    data = np.outer(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))

    fig, ax = plt.subplots(figsize=(8, 8))
    # Display the grid; using extent and origin set so that grid coordinates match click positions
    cax = ax.imshow(data, extent=[0, grid_size, 0, grid_size], origin="lower", cmap="viridis")
    fig.colorbar(cax)
    ax.set_title("Latent Space Grid (64x64)")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    def on_click(event):
        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
            # Map click to a grid cell by converting to int indices
            x_idx = int(event.xdata)
            y_idx = int(event.ydata)
            print(f"Clicked on grid cell: ({x_idx}, {y_idx})")
            # Create a latent vector and encode the selected grid cell in the first two dims
            latent_vector = torch.zeros(latent_dim, dtype=torch.float32)
            latent_vector[0] = x_idx
            latent_vector[1] = y_idx
            decode_and_play(vae, latent_vector, sample_rate=sample_rate)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


if __name__ == "__main__":
    # Load the pre-trained VAE model with latent_dim 2 to match the checkpoint
    model_path = "./weights/vae_weights.pth"
    vae = load_model(model_path, input_channels=1, latent_dim=2, input_dim=(128, 126))

    # Visualize the latent space using both dimensions of the 2D latent space
    visualize_latent_space(vae, latent_dim=2, sample_rate=16000)
