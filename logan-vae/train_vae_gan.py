import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
import soundfile as sf
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np

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

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

# Define the loss functions
def vae_loss(recon_x, x, mu, log_var):
    recon_loss = nn.MSELoss()(recon_x, x)  # Reconstruction loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # KL divergence
    return recon_loss + kl_loss

def gan_loss(discriminator_output, target):
    # Clamp discriminator output to avoid numerical instability
    discriminator_output = torch.clamp(discriminator_output, 1e-7, 1 - 1e-7)
    return nn.BCELoss()(discriminator_output, target)

# Dataset class for audio files
class AudioDataset(Dataset):
    def __init__(self, audio_dir, sample_rate=16000, n_fft=400, hop_length=160, max_length=1000):
        self.audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")]
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = max_length  # Maximum length of spectrogram

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        waveform, sr = sf.read(self.audio_files[idx])
        if len(waveform.shape) > 1:  # Convert to mono if stereo
            waveform = np.mean(waveform, axis=1)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        spectrogram_transform = T.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length)
        spectrogram = spectrogram_transform(waveform)

        # Pad or truncate spectrogram to fixed length
        if spectrogram.shape[1] < self.max_length:
            padding = torch.zeros((spectrogram.shape[0], self.max_length - spectrogram.shape[1]))
            spectrogram = torch.cat([spectrogram, padding], dim=1)
        else:
            spectrogram = spectrogram[:, :self.max_length]

        # Normalize spectrogram
        spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()

        return spectrogram.reshape(-1)  # Use reshape instead of view

# Training function
def train_vae_gan(encoder, decoder, discriminator, dataloader, epochs=100, latent_dim=100, lr=0.0002):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)
    discriminator.to(device)

    optimizer_E = optim.Adam(encoder.parameters(), lr=lr)
    optimizer_D = optim.Adam(decoder.parameters(), lr=lr)
    optimizer_Dis = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(epochs):
        for real_data in dataloader:
            real_data = real_data.to(device)
            batch_size = real_data.size(0)

            # Train Encoder and Decoder (VAE)
            optimizer_E.zero_grad()
            optimizer_D.zero_grad()
            z, mu, log_var = encoder(real_data)
            recon_data = decoder(z)
            vae_l = vae_loss(recon_data, real_data, mu, log_var)
            vae_l.backward()
            optimizer_E.step()
            optimizer_D.step()

            # Train Discriminator
            optimizer_Dis.zero_grad()
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Real data loss
            real_output = discriminator(real_data)
            loss_real = gan_loss(real_output, real_labels)

            # Fake data loss
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_data = decoder(z)
            fake_output = discriminator(fake_data.detach())
            loss_fake = gan_loss(fake_output, fake_labels)

            # Total discriminator loss
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_Dis.step()

            # Train Generator (Decoder)
            optimizer_D.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_data = decoder(z)
            fake_output = discriminator(fake_data)
            loss_G = gan_loss(fake_output, real_labels)  # Generator tries to fool discriminator
            loss_G.backward()
            optimizer_D.step()

        print(f"Epoch {epoch + 1}, Loss VAE: {vae_l.item()}, Loss D: {loss_D.item()}, Loss G: {loss_G.item()}")

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VAE-GAN on audio data.")
    parser.add_argument("audio_dir", type=str, help="Path to directory containing audio files")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--latent_dim", type=int, default=100, help="Dimension of the latent space")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate for training")
    parser.add_argument("--output_model", type=str, default="vae_gan.pth", help="Path to save the trained model")
    parser.add_argument("--max_length", type=int, default=1000, help="Maximum length of spectrogram (time dimension)")
    args = parser.parse_args()

    # Load dataset
    dataset = AudioDataset(args.audio_dir, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Define model dimensions
    input_dim = dataset[0].shape[0]  # Flattened spectrogram size
    latent_dim = args.latent_dim

    # Initialize models
    encoder = Encoder(input_dim, latent_dim)
    decoder = Decoder(latent_dim, input_dim)
    discriminator = Discriminator(input_dim)

    # Train VAE-GAN
    train_vae_gan(encoder, decoder, discriminator, dataloader, epochs=args.epochs, latent_dim=latent_dim, lr=args.learning_rate)

    # Save models
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "discriminator": discriminator.state_dict(),
    }, args.output_model)
    print(f"Models saved to {args.output_model}")