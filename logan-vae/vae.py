import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

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
def train_vae(model, data, epochs=50, batch_size=32, learning_rate=1e-3):
    """Train the VAE on MFCC data."""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()  # TODO: Consider KL divergence + reconstruction loss

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
            loss = recon_loss + kl_loss  # TODO: Adjust weighting of KL loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a VAE on MFCC representations.")
    parser.add_argument("wav_file", type=str, help="Path to input WAV file")
    args = parser.parse_args()

    # Extract MFCCs
    mfcc = extract_mfcc(args.wav_file)
    mfcc = mfcc.permute(1, 0)  # Shape (time, features)

    # Normalize MFCCs
    mfcc = (mfcc - mfcc.mean()) / mfcc.std()

    # Flatten input for VAE
    input_dim = mfcc.shape[1]
    data = mfcc.view(mfcc.shape[0], -1)

    # Create and train VAE
    vae = VAE(input_dim)
    train_vae(vae, data)

    # Save the model
    torch.save(vae.state_dict(), "vae_mfcc.pth")

    print("VAE training complete! Model saved.")
