import numpy as np
import matplotlib.pyplot as plt
import torch
from model import VAE
from nsynth_dataset import NsynthDataset
import sounddevice as sd
import torchaudio
import json


LATENT_DIM = 2
# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # Use MPS if available
DEVICE = torch.device("cpu")
def load_model(filepath, input_channels=1, input_dim=(128, 126)):
    """Load the VAE model with pre-trained weights."""

    vae = VAE(input_channels=input_channels, latent_dim=LATENT_DIM, input_dim=input_dim).to(DEVICE)
    vae.load_weights(filepath)
    return vae

def infer(vae, input_tensor):
    """Perform inference using the VAE model."""
    vae.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        reconstructed, mu, sigma = vae(input_tensor)
    return reconstructed, mu, sigma


def select_samples(dataset, num_samples=10):
    """Select a random subset of samples from the dataset."""
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    samples = [dataset[i] for i in indices]
    images = torch.stack([sample[0] for sample in samples])  # Stack spectrograms
    labels = [sample[1] for sample in samples]  # Collect labels
    return images, labels




def plot_images_encoded_in_latent_space(latent_families_dict):
    labels = latent_families_dict.keys()
    for label in labels:
        latent_points = latent_families_dict[label]
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_points[:, 0],
                latent_points[:, 1], 
                cmap="rainbow",
                c=sample_labels, 
                alpha=0.5, 
                s=2)
    plt.colorbar()
    plt.show()



def sample_and_play_from_latent_space(vae, num_samples=5, latent_dim=LATENT_DIM, sample_rate=16000):
    """Sample random points from the latent space, decode them, and play the resulting spectrograms."""
    vae.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        # Sample random points from a standard normal distribution
        random_latent_vectors = torch.randn(num_samples, latent_dim).to(DEVICE)  # Ensure latent vectors are on MPS
        # Decode the latent vectors into spectrograms
        decoded_spectrograms = vae.decode(random_latent_vectors)

    # Convert decoded spectrograms back to waveforms and play them
    mel_to_stft = torchaudio.transforms.InverseMelScale(
        n_stft=1025, n_mels=128, sample_rate=sample_rate
    )
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=2048, hop_length=512)

    for i, spectrogram in enumerate(decoded_spectrograms):
        mel_spec = spectrogram
        stft_spec = mel_to_stft(mel_spec)
        waveform = griffin_lim(stft_spec)
        print(f"Playing sample {i + 1}/{num_samples}...")
        sd.play(waveform.squeeze().numpy(), samplerate=sample_rate)
        sd.wait()

if __name__ == "__main__":
    # Load NSynth dataset
    nsynth = NsynthDataset(path="/Volumes/ronen_usb/nsynth-train")

    # Load pre-trained VAE
    model_path = "./weights/vae_weights.pth"
    vae = load_model(model_path, input_channels=1, input_dim=(128, 126))

    # Select a subset of samples
    sample_images, sample_labels = select_samples(nsynth, num_samples=10)

    # Perform inference
    sample_images_tensor = sample_images.to(torch.float32).to(DEVICE)  # Ensure correct dtype
    vae.eval()
    with torch.no_grad():
        reconstructed_images, latent_mu, _ = vae(sample_images_tensor)

    # Play original and reconstructed spectrograms
    mel_to_stft = torchaudio.transforms.InverseMelScale(
        n_stft=1025, n_mels=128, sample_rate=16000
    )
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=2048, hop_length=512)

    # for i, (original, reconstructed) in enumerate(zip(sample_images_tensor, reconstructed_images)):
    #     print(f"Playing original spectrogram {i + 1}/{len(sample_images_tensor)}...")
    #     original_mel_spec = torch.exp(original) - 1e-9
    #     original_stft_spec = mel_to_stft(original_mel_spec)
    #     original_waveform = griffin_lim(original_stft_spec)
    #     sd.play(original_waveform.squeeze().numpy(), samplerate=16000)
    #     sd.wait()

    #     print(f"Playing reconstructed spectrogram {i + 1}/{len(reconstructed_images)}...")
    #     reconstructed_mel_spec = reconstructed #torch.exp(reconstructed) - 1e-9
    #     reconstructed_stft_spec = mel_to_stft(reconstructed_mel_spec)
    #     reconstructed_waveform = griffin_lim(reconstructed_stft_spec)
    #     sd.play(reconstructed_waveform.squeeze().numpy(), samplerate=16000)
    #     sd.wait()

    #     plot_reconstructed_images(original, reconstructed)




    # Visualize latent space with 100 samples from each instrument family
    # Assuming instrument families are stored as a dictionary in the dataset metadata
    families_dict = nsynth.specs_per_family
    all_images = []
    all_labels = []

    latent_families_dict = {}

    for family, specs in families_dict.items():
        if len(specs) > 0:
            print(len(specs))
            selected_indices = np.random.choice(len(specs), 1000, replace=True)

            family_images = [specs[i] for i in selected_indices]

            family_images_tensor = torch.stack(family_images).to(torch.float32).to(DEVICE)
            with torch.no_grad():
                _, latent_mu, _ = vae(family_images_tensor)
                latent_families_dict[family] = latent_mu

            # Convert latent_mu tensors to lists for JSON serialization
            latent_families_dict_serializable = {family: latent_mu.cpu().numpy().tolist() for family, latent_mu in latent_families_dict.items()}

            # Save the dictionary to a JSON file
            output_path = "./latent_families.json"
            with open(output_path, "w") as json_file:
                json.dump(latent_families_dict_serializable, json_file)

            print(f"Latent representations saved to {output_path}")
            # family_images_tensor = torch.stack().to(torch.float32).to(DEVICE)
            # with torch.no_grad():
            #     _, latent_mu, _ = vae(family_images_tensor)
            #     latent_families_dict[family] = latent_mu

        else:
            print(f"Warning: No spectrograms found for family '{family}'")


    plot_images_encoded_in_latent_space(latent_families_dict)

    # # Sample and play from the latent space
    # sample_and_play_from_latent_space(vae, num_samples=5, latent_dim=LATENT_DIM, sample_rate=16000)
