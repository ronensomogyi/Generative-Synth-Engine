import numpy as np
import matplotlib.pyplot as plt
import torch
import sounddevice as sd
import torchaudio
from model import VAE
from nsynth_dataset import NsynthDataset

def load_model(filepath, input_channels=1, latent_dim=20, input_dim=(128, 126)):
    """Load the VAE model with pre-trained weights."""
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
    """Visualize the latent space and allow sampling by clicking."""
    if latent_dim != 2:
        raise ValueError("Latent space visualization requires a 2D latent space.")

    # Generate a grid of points in the latent space
    grid_x, grid_y = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))

    # Create a figure for the latent space
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Latent Space")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")

    # Define the click event handler
    def on_click(event):
        if event.inaxes == ax:
            # Get the clicked coordinates
            latent_vector = torch.tensor([event.xdata, event.ydata], dtype=torch.float32)
            print(f"Clicked at: {latent_vector.numpy()}")
            # Decode and play the corresponding waveform
            decode_and_play(vae, latent_vector, sample_rate=sample_rate)

    # Connect the click event to the handler
    fig.canvas.mpl_connect("button_press_event", on_click)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Load the pre-trained VAE model
    model_path = "./weights/vae_weights.pth"
    vae = load_model(model_path, input_channels=1, latent_dim=2, input_dim=(128, 126))

    # Visualize the latent space
    visualize_latent_space(vae, latent_dim=2, sample_rate=16000)
