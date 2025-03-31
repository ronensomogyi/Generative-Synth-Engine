import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_channels=1, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # Conv layer 1
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Conv layer 2
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Conv layer 3
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Flatten()
        )
        
        # Compute the shape before bottleneck
        self._shape_before_bottleneck = (128, 4, 4)  # Adjust based on input size
        self.bottleneck_dim = 128 * 4 * 4
        
        # Latent space
        self.fc_mu = nn.Linear(self.bottleneck_dim, latent_dim)
        self.fc_sigma = nn.Linear(self.bottleneck_dim, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.bottleneck_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Deconv layer 1
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Deconv layer 2
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # Deconv layer 3
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        return mu, sigma

    def reparameterize(self, mu, sigma):
        epsilon = torch.randn_like(sigma)
        return mu + epsilon * sigma

    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, *self._shape_before_bottleneck)
        decoded = self.decoder(z)
        return F.interpolate(decoded, size=(28, 28), mode="bilinear", align_corners=False)

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, sigma

    def save_weights(self, filepath='./weights/vae_weights.pth'):
        """Save model weights to the specified file."""
        torch.save(self.state_dict(), filepath)

    def load_weights(self, filepath='./weights/vae_weights.pth'):
        """Load model weights from the specified file."""
        self.load_state_dict(torch.load(filepath))
        self.eval()  # Set the model to evaluation mode after loading weights


if __name__ == '__main__':
    # Example usage
    x = torch.randn(4, 1, 28, 28)  # Batch of 4 grayscale images (28x28)
    vae = VAE(input_channels=1, latent_dim=20)
    x_re, mu, sigma = vae(x)
    print(x_re.shape, mu.shape, sigma.shape)


