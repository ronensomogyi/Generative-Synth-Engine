import torch




from torch import nn
from torch.utils.data import DataLoader
from model import VAE
from tqdm import tqdm


from nsynth_dataset import NsynthDataset
# Config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
INPUT_CHANNELS = 1  # Grayscale images
LATENT_DIM = 20

NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3  # Reduced learning rate for stability
KL_COEF = 0.1  # Coefficient for KL divergence loss
RECON_COEF = 0.9  # Coefficient for reconstruction loss




def train(train_loader, model, optimizer, loss_fn):

    alpha = RECON_COEF  # Hyperparameter to scale the KL divergence loss
    beta = KL_COEF

    best_loss = float('inf')  # Initialize best loss to infinity
    best_epoch = -1  # Track the epoch with the lowest loss
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(enumerate(train_loader))
 

        for i, (x, _) in loop:
            # forward pass
            x = x.to(DEVICE).view(x.shape[0], 1, 128, 126)  # Reshape for Conv2D input
            x_reconstructed, mu, sigma = model(x)

            # loss
            reconstruction_loss = loss_fn(x_reconstructed, x)
            kl_divergence = -torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2) * 0.5

            # backpropogation
            loss = alpha * reconstruction_loss + beta * kl_divergence
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        # Save model weights if this epoch has the lowest loss
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch + 1
            model.save_weights()

        
        print(f"Epoch {epoch + 1} completed. Best loss so far: {best_loss} at epoch {best_epoch}")



def main():


    dataset = NsynthDataset(path="/Volumes/ronen_usb/nsynth-train")
    input_dim = dataset[0][0].shape[1:]  # Get the shape of the first sample

    train_loader = DataLoader(dataset=dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True)

    model = VAE(input_channels=INPUT_CHANNELS, latent_dim=LATENT_DIM, input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss(reduction='sum') # reconstruction

    train(train_loader, model, optimizer, loss_fn)

if __name__ == '__main__':
    main()