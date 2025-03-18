import argparse
import torch
import numpy as np
import soundfile as sf
from vae import VAE, extract_mfcc  # Ensure VAE and MFCC extraction functions are available

def encode_audio(wav_file, model_path="vae_mfcc.pth"):
    """
    Encodes an audio file into the latent space using a trained Variational Autoencoder (VAE).

    Args:
        wav_file (str): Path to the input WAV file.
        model_path (str): Path to the trained VAE model.

    Returns:
        torch.Tensor: Latent space representation of the input audio.
    """
    # Load trained VAE
    vae = VAE(input_dim=13)  # TODO: Ensure this matches the training config
    vae.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    vae.eval()

    # Load audio and extract MFCCs
    mfcc = extract_mfcc(wav_file)  # Shape: (n_mfcc, time)
    mfcc = mfcc.permute(1, 0)  # Reshape to (time, n_mfcc)

    # Normalize (same as during training)
    mfcc = (mfcc - mfcc.mean()) / mfcc.std()

    # Encode to latent space
    with torch.no_grad():
        encoded = vae.encoder(mfcc)
        mu = vae.mu(encoded)
        log_var = vae.log_var(encoded)
        latent_vector = vae.reparameterize(mu, log_var)

    return latent_vector.numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an audio file into latent space using a trained VAE.")
    parser.add_argument("wav_file", type=str, help="Path to the WAV file to encode")
    parser.add_argument("--model", type=str, default="vae_mfcc.pth", help="Path to the trained VAE model")

    args = parser.parse_args()
    
    latent_representation = encode_audio(args.wav_file, args.model)
    
    print("Encoded Latent Representation:")
    print(latent_representation)
