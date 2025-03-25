import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# Define the VAE-GAN components
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.mu = nn.Linear(128, latent_dim)
        self.log_var = nn.Linear(128, latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.model(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)

# Load models
def load_models(model_path, input_dim, latent_dim):
    checkpoint = torch.load(model_path)
    encoder = Encoder(input_dim, latent_dim)
    decoder = Decoder(latent_dim, input_dim)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    return encoder, decoder

# Generate audio
def generate_audio(encoder, decoder, latent_dim, n_mels=128, max_length=1000, n_fft=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    z = torch.randn(1, latent_dim).to(device)
    fake_spectrogram_flat = decoder(z).cpu().detach()

    # Reshape the flattened spectrogram to (n_mels, max_length)
    fake_spectrogram = fake_spectrogram_flat.view(n_mels, max_length)

    # Apply log scaling to emphasize higher frequencies
    fake_spectrogram = torch.log1p(fake_spectrogram)

    # Normalize the spectrogram to a fixed range
    fake_spectrogram = (fake_spectrogram - fake_spectrogram.min()) / (fake_spectrogram.max() - fake_spectrogram.min())

    # Upsample the mel spectrogram to the full frequency dimension
    target_freq_dim = n_fft // 2 + 1  # Expected frequency dimension for GriffinLim
    fake_spectrogram = fake_spectrogram.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    fake_spectrogram = F.interpolate(fake_spectrogram, size=(target_freq_dim, max_length), mode="bilinear")
    fake_spectrogram = fake_spectrogram.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

    # Visualize the spectrogram
    plt.imshow(fake_spectrogram, aspect="auto", origin="lower")
    plt.colorbar()
    plt.title("Generated Spectrogram")
    plt.show()

    # Convert spectrogram back to waveform (e.g., using Griffin-Lim)
    griffin_lim = T.GriffinLim(n_fft=n_fft, hop_length=256)
    waveform = griffin_lim(fake_spectrogram.unsqueeze(0))  # Add batch dimension
    waveform = waveform.squeeze(0)  # Remove batch dimension

    # Print waveform statistics
    print(f"Waveform min: {waveform.min()}, max: {waveform.max()}, mean: {waveform.mean()}, std: {waveform.std()}")

    return waveform

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio using a trained VAE-GAN.")
    parser.add_argument("model_path", type=str, help="Path to the trained VAE-GAN model")
    parser.add_argument("--output_file", type=str, default="generated_audio.wav", help="Output WAV file name")
    parser.add_argument("--latent_dim", type=int, default=100, help="Dimension of the latent space")
    parser.add_argument("--input_dim", type=int, default=128000, help="Input dimension (flattened spectrogram size)")
    parser.add_argument("--n_mels", type=int, default=128, help="Number of mel bins in the spectrogram")
    parser.add_argument("--max_length", type=int, default=1000, help="Maximum length of spectrogram (time dimension)")
    parser.add_argument("--n_fft", type=int, default=1024, help="FFT size for Griffin-Lim")
    args = parser.parse_args()

    # Define input and output dimensions (must match training)
    input_dim = args.input_dim  # Flattened spectrogram size (must match training)
    latent_dim = args.latent_dim
    n_mels = args.n_mels
    max_length = args.max_length
    n_fft = args.n_fft

    # Load models
    encoder, decoder = load_models(args.model_path, input_dim, latent_dim)

    # Generate audio
    generated_waveform = generate_audio(encoder, decoder, latent_dim, n_mels, max_length, n_fft)

    # Save generated audio
    sf.write(args.output_file, generated_waveform.numpy().T, 16000)
    print(f"Generated audio saved to {args.output_file}")