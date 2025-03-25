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

# Load and process audio to MFCC
def extract_mfcc(wav_file, sample_rate=16000, n_mfcc=13, n_fft=400, hop_length=160):
    """Extract MFCC features from an audio file."""
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

    # Compute MFCC
    mfcc_transform = T.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs={
        "n_fft": n_fft, "hop_length": hop_length, "n_mels": 40
    })
    mfcc = mfcc_transform(waveform)
    
    return mfcc

# Define the Variational Autoencoder
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        """Variational Autoencoder (VAE) for MFCC encoding."""
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),  # TODO: Adjust hidden layer sizes
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Latent space
        self.mu = nn.Linear(64, latent_dim)
        self.log_var = nn.Linear(64, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),  # Output same shape as input
            nn.Tanh()  # TODO: Consider activation function (Tanh keeps values between -1 and 1)
        )

    def reparameterize(self, mu, log_var):
        """Reparameterization trick for VAE: z = mu + sigma * epsilon."""
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

# Training function
def train_vae(model, data, epochs=50, batch_size=32, learning_rate=1e-4, beta=0.1):
    """Train the VAE on MFCC data."""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Learning rate scheduler
    loss_fn = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch[0]
            optimizer.zero_grad()
            recon, mu, log_var = model(batch)

            # Compute loss
            recon_loss = loss_fn(recon, batch)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + beta * kl_loss  # Adjust beta for KL loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()  # Update learning rate
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Load all audio files from a directory (including nested folders)
def load_audio_files(directory, sample_rate=16000, n_mfcc=13):
    """Load all audio files from a directory and extract MFCC features."""
    mfccs = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):  # Add other formats if needed
                file_path = os.path.join(root, file)
                try:
                    mfcc = extract_mfcc(file_path, sample_rate=sample_rate, n_mfcc=n_mfcc)
                    mfccs.append(mfcc.permute(1, 0))  # Shape: (time, n_mfcc)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    return mfccs

# Combine MFCCs into a single tensor
def prepare_data(mfccs):
    """Combine MFCCs into a single tensor and normalize."""
    # Concatenate all MFCCs along the time dimension
    combined = torch.cat(mfccs, dim=0)

    # Normalize
    combined = (combined - combined.mean()) / combined.std()

    return combined

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a VAE on MFCC representations.")
    parser.add_argument("audio_dir", type=str, help="Path to directory containing audio files")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--output_model", type=str, default="vae_mfcc.pth", help="Path to save the trained model")
    args = parser.parse_args()

    # Load audio files and extract MFCCs
    print("Loading audio files and extracting MFCCs...")
    mfccs = load_audio_files(args.audio_dir)

    # Prepare data for training
    print("Preparing data for training...")
    data = prepare_data(mfccs)

    # Flatten input for VAE
    input_dim = data.shape[1]
    data = data.view(data.shape[0], -1)

    # Create and train VAE
    print("Training VAE...")
    vae = VAE(input_dim)
    train_vae(vae, data, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)

    # Save the model
    torch.save(vae.state_dict(), args.output_model)
    print(f"VAE training complete! Model saved to {args.output_model}.")