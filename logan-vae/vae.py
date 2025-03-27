import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
import numpy as np
import soundfile as sf
import os
from pathlib import Path

# -------------------------
# 1. Extract Mel Spectrogram
# -------------------------


def extract_mel(wav_file, sample_rate=16000, n_fft=400, hop_length=160, n_mels=64):
    """
    Extracts a mel spectrogram from an audio file.

    Args:
        wav_file (str): Path to the WAV file.
        sample_rate (int): Target sample rate.
        n_fft (int): FFT window size.
        hop_length (int): Hop length.
        n_mels (int): Number of mel bins.

    Returns:
        torch.Tensor: Mel spectrogram of shape (n_mels, time).
    """
    waveform, sr = sf.read(wav_file)

    # Convert to mono if stereo
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)

    # Convert to PyTorch tensor
    waveform = torch.tensor(waveform, dtype=torch.float32)

    # Ensure correct sample rate
    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Compute mel spectrogram using torchaudio's transform
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spec = mel_transform(waveform)  # shape: (n_mels, time)
    return mel_spec

# -------------------------
# 2. Define the Variational Autoencoder
# -------------------------


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        """
        Variational Autoencoder (VAE) for mel spectrogram encoding.

        Args:
            input_dim (int): Dimensionality of the input (number of mel bins).
            latent_dim (int): Dimensionality of the latent space.
        """
        super(VAE, self).__init__()

        # Encoder network: compress each mel frame to a lower-dimensional latent vector
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),  # Adjust hidden layer sizes if needed
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Latent space parameters
        self.mu = nn.Linear(64, latent_dim)
        self.log_var = nn.Linear(64, latent_dim)

        # Decoder network: reconstruct mel frame from latent vector
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),  # Output same shape as input
            nn.Tanh()  # Tanh constrains output values between -1 and 1
        )

    def reparameterize(self, mu, log_var):
        """Reparameterization trick: z = mu + sigma * epsilon."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Forward pass: Encode -> Sample -> Decode."""
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return decoded, mu, log_var

# -------------------------
# 3. Training Function
# -------------------------


def train_vae(model, data, epochs=50, batch_size=32, learning_rate=1e-3):
    """Train the VAE on mel spectrogram data."""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()  # Using MSE for reconstruction; KL divergence is added separately

    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch[0]
            optimizer.zero_grad()
            recon, mu, log_var = model(batch)

            # Compute reconstruction loss and KL divergence loss
            recon_loss = loss_fn(recon, batch)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + kl_loss  # Optionally, weight the KL term

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# -------------------------
# 4. Load and Prepare Data
# -------------------------


def load_audio_files(directory, sample_rate=16000, n_mels=64, n_fft=400, hop_length=160):
    """
    Load all audio files from a directory and extract mel spectrograms.

    Args:
        directory (str): Root directory containing audio files.

    Returns:
        List[torch.Tensor]: List of mel spectrogram tensors with shape (time, n_mels).
    """
    mels = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                try:
                    mel_spec = extract_mel(
                        file_path, sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
                    # Permute to (time, n_mels)
                    mels.append(mel_spec.permute(1, 0))
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    return mels


def prepare_data(mels):
    """Combine mel spectrograms into a single tensor and normalize."""
    combined = torch.cat(mels, dim=0)
    combined = (combined - combined.mean()) / combined.std()
    return combined


# -------------------------
# 5. Main: Training the VAE
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a VAE on mel spectrogram representations.")
    parser.add_argument("audio_dir", type=str,
                        help="Path to directory containing audio files")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float,
                        default=1e-3, help="Learning rate for training")
    parser.add_argument("--output_model", type=str,
                        default="vae_mel.pth", help="Path to save the trained model")
    args = parser.parse_args()

    print("Loading audio files and extracting mel spectrograms...")
    mels = load_audio_files(args.audio_dir)

    print("Preparing data for training...")
    data = prepare_data(mels)

    # Flatten each mel frame so that input_dim equals the number of mel bins
    input_dim = data.shape[1]
    data = data.view(data.shape[0], -1)

    print("Training VAE...")
    vae = VAE(input_dim)
    train_vae(vae, data, epochs=args.epochs,
              batch_size=args.batch_size, learning_rate=args.learning_rate)

    torch.save(vae.state_dict(), args.output_model)
    print(f"VAE training complete! Model saved to {args.output_model}.")
