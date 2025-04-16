import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
import soundfile as sf
import os
from torch.utils.data import Dataset, DataLoader
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
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
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
            nn.ReLU(),
        )

    def forward(self, z):
        output = self.model(z)
        return torch.clamp(output, min=1e-5)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

# GAN loss
def gan_loss(discriminator_output, target):
    discriminator_output = torch.clamp(discriminator_output, 1e-7, 1 - 1e-7)
    return nn.BCELoss()(discriminator_output, target)

# Dataset class for audio files
class AudioDataset(Dataset):
    def __init__(self, audio_dir, sample_rate=16000, n_fft=1024, hop_length=160,
                 max_length=1000, n_mels=128, global_mean=None, global_std=None, normalize=True):
        self.audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")]
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = max_length
        self.n_mels = n_mels
        self.global_mean = global_mean
        self.global_std = global_std
        # self.normalize = normalize
        self.normalize = False

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        waveform, sr = sf.read(self.audio_files[idx])
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # spectrogram = T.MelSpectrogram(
        #     sample_rate=self.sample_rate,
        #     n_fft=self.n_fft,
        #     hop_length=self.hop_length,
        #     n_mels=self.n_mels
        # )(waveform)

        # # spectrogram = torch.log1p(spectrogram)  # log-mel for better scale
        # spectrogram = T.Spectrogram(
        #     n_fft=self.n_fft,
        #     hop_length=self.hop_length,
        #     power=2.0,
        # )(waveform)

        spectrogram = torch.abs(
            T.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length)(waveform)
        )


        if spectrogram.shape[1] < self.max_length:
            padding = torch.zeros((spectrogram.shape[0], self.max_length - spectrogram.shape[1]))
            spectrogram = torch.cat([spectrogram, padding], dim=1)
        else:
            spectrogram = spectrogram[:, :self.max_length]

        if self.normalize and self.global_mean is not None and self.global_std is not None:
            spectrogram = (spectrogram - self.global_mean) / self.global_std

        return spectrogram.reshape(-1)

# Training loop with VAE warmup
def train_vae_gan(encoder, decoder, discriminator, dataloader, epochs=100, latent_dim=64, lr=1e-4, beta=0.001, warmup_epochs=20):
    device = torch.device("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
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

            # VAE step
            optimizer_E.zero_grad()
            optimizer_D.zero_grad()
            z, mu, log_var = encoder(real_data)
            recon_data = decoder(z)
            # recon_loss = nn.MSELoss()(recon_data, real_data)
            recon_loss = nn.L1Loss()(recon_data, real_data)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
            beta_schedule = min(1.0, epoch / 50)
            vae_l = recon_loss + beta_schedule * beta * kl_loss
            vae_l.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer_E.step()
            optimizer_D.step()

            continue  # Skip GAN for first few epochs
            # Skip GAN for first few epochs
            if epoch < warmup_epochs:
                continue

            # Discriminator step
            optimizer_Dis.zero_grad()
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            real_output = discriminator(real_data)
            loss_real = gan_loss(real_output, real_labels)

            z = torch.randn(batch_size, latent_dim).to(device)
            fake_data = decoder(z)
            fake_output = discriminator(fake_data.detach())
            loss_fake = gan_loss(fake_output, fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_Dis.step()

            # Generator step
            optimizer_D.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_data = decoder(z)
            fake_output = discriminator(fake_data)
            loss_G = gan_loss(fake_output, real_labels)
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer_D.step()

        # if epoch < warmup_epochs:
            # print(f"Epoch {epoch + 1}, Loss VAE: {vae_l.item():.4f} (VAE warmup)")
        # else:
            # print(f"Epoch {epoch + 1}, Loss VAE: {vae_l.item():.4f}, Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
        print(f"Epoch {epoch + 1}, Loss VAE: {vae_l.item():.4f} (VAE warmup)")

        # Save reconstructed waveform
        if (epoch + 1) % 10 == 0:
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                real_data = next(iter(dataloader)).to(device)
                z, _, _ = encoder(real_data)
                recon = decoder(z).cpu()[0]
                recon_spec = recon.view(513, -1)

                denorm_recon_spec = recon_spec * global_std + global_mean
                # denorm_recon_spec = torch.clamp(denorm_recon_spec, min=1e-5)
                # denorm_recon_spec = torch.relu(denorm_recon_spec)

                # upsampled = denorm_recon_spec.unsqueeze(0).unsqueeze(0)
                # upsampled = torch.nn.functional.interpolate(upsampled, size=(513, denorm_recon_spec.shape[1]), mode="bilinear")
                # upsampled = upsampled.squeeze(0).squeeze(0)

                # griffin_lim = T.GriffinLim(n_fft=1024, hop_length=160, n_iter=64)
                # waveform = griffin_lim(upsampled.unsqueeze(0)).squeeze(0)
                griffin_lim = T.GriffinLim(n_fft=1024, hop_length=160, n_iter=64)
                waveform = griffin_lim(recon_spec.unsqueeze(0)).squeeze(0)

                debug_dir = f"relu_debug_epoch_{epoch+1}"
                os.makedirs(debug_dir, exist_ok=True)

                plt.plot(waveform.numpy())
                plt.title("Reconstructed waveform")
                plt.savefig(f"{debug_dir}/waveform.png")
                plt.close()

                # Save audio
                sf.write(f"{debug_dir}/reconstructed.wav", waveform.numpy(), 16000)
                print(f"[Saved] {debug_dir}/reconstructed.wav")

                # Save spectrogram plot
                plt.figure(figsize=(10, 4))
                plt.imshow(denorm_recon_spec, aspect="auto", origin="lower")
                plt.title(f"Reconstructed Spectrogram (Epoch {epoch + 1})")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(f"{debug_dir}/spectrogram.png")
                plt.close()

                # Save latent space plot (first two dims)
                z_np = z.cpu().numpy()
                plt.figure(figsize=(6, 6))
                plt.scatter(z_np[:, 0], z_np[:, 1], alpha=0.6)
                plt.title(f"Latent Space (Epoch {epoch + 1})")
                plt.xlabel("z[0]")
                plt.ylabel("z[1]")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{debug_dir}/latent_space.png")
                plt.close()

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VAE-GAN on audio data.")
    parser.add_argument("audio_dir", type=str, help="Path to directory containing audio files")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--output_model", type=str, default="vae_gan.pth")
    parser.add_argument("--max_length", type=int, default=1000)
    args = parser.parse_args()

    # TEMP dataset to calculate mean/std
    print("Computing dataset mean and std for normalization...")
    temp_dataset = AudioDataset(args.audio_dir, max_length=args.max_length, normalize=False)
    all_specs = torch.stack([temp_dataset[i] for i in range(len(temp_dataset))])
    global_mean = all_specs.mean().item()
    global_std = all_specs.std().item()
    print(f"Global mean: {global_mean:.4f}, std: {global_std:.4f}")

    # Final dataset with normalization
    dataset = AudioDataset(args.audio_dir, max_length=args.max_length, global_mean=global_mean, global_std=global_std, normalize=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    input_dim = dataset[0].shape[0]
    latent_dim = args.latent_dim

    encoder = Encoder(input_dim, latent_dim)
    decoder = Decoder(latent_dim, input_dim)
    discriminator = Discriminator(input_dim)

    train_vae_gan(encoder, decoder, discriminator, dataloader,
                  epochs=args.epochs, latent_dim=latent_dim, lr=args.learning_rate)

    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "discriminator": discriminator.state_dict(),
        "latent_dim": latent_dim,
        "input_dim": input_dim,
        "global_mean": global_mean,
        "global_std": global_std,
    }, args.output_model)

    print(f"Models saved to {args.output_model}")