import torch
from torch import nn
from torch.utils.data import DataLoader
from model import VAE
from tqdm import tqdm
from nsynth_dataset import NsynthDataset

# Config
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # Use MPS if available
INPUT_CHANNELS = 1  # Grayscale images
LATENT_DIM = 2
LEARNING_RATE = 4e-3  # Reduced learning rate for stability
NUM_EPOCHS = 30
BATCH_SIZE = 32
KL_COEF = 0.1  # Coefficient for KL divergence loss
RECON_COEF = 0.9  # Coefficient for reconstruction loss


def train(train_loader, model, optimizer, loss_fn):
    best_loss = float('inf')  # Initialize best loss to infinity
    best_epoch = -1  # Track the epoch with the lowest loss

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}:")
        #print(f"Before training, weights of fc_mu: {model.fc_mu.weight[0][:5]}")  # Log first 5 weights of fc_mu

        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        alpha = RECON_COEF
        beta = epoch / 10.0  # KL annealing coefficient

        for i, (x, _) in loop:
            x = x.to(DEVICE).view(x.shape[0], 1, 128, 126)  # Reshape for Conv2D input
            x_reconstructed, mu, sigma = model(x)

            # Loss computation
            reconstruction_loss = loss_fn(x_reconstructed, x)
            kl_divergence = -torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2) * 0.5
            loss = alpha * reconstruction_loss + beta * kl_divergence

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())


        # Save model weights if this epoch has the lowest loss
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch + 1
            model.save_weights(filepath='./weights/vae_best.pth')

        print(f"Epoch {epoch + 1} completed. Best loss so far: {best_loss} at epoch {best_epoch}")


def main():
    dataset = NsynthDataset(path="/Volumes/ronen_usb/nsynth-train")
    input_dim = dataset[0][0].shape[1:]  # Get the shape of the first sample
    train_loader = DataLoader(dataset=dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=4)  # Use 4 workers for parallel data loading
    model = VAE(input_channels=INPUT_CHANNELS, latent_dim=LATENT_DIM, input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss(reduction='sum')  # Reconstruction loss
    train(train_loader, model, optimizer, loss_fn)


if __name__ == '__main__':
    main()

