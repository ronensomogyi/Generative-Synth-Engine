import argparse
import torch
import numpy as np
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
# Ensure VAE and MFCC extraction functions are available
from vae import VAE, extract_mfcc
import matplotlib.pyplot as plt


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
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    waveform = torch.tensor(waveform, dtype=torch.float32)
    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spec = mel_transform(waveform)  # shape: (n_mels, time)
    return mel_spec


def encode_audio(wav_file, vae):
    """
    Encodes an audio file into the latent space using a trained Variational Autoencoder (VAE).

    Args:
        wav_file (str): Path to the input WAV file.
        model_path (str): Path to the trained VAE model.

    Returns:
        torch.Tensor: Latent space representation of the input audio.
    """
    # Extract mel spectrogram: shape (n_mels, time)
    mel_spec = extract_mel(wav_file)
    # Permute to (time, n_mels) so that each time frame is an input sample
    mel_spec = mel_spec.permute(1, 0)
    # Normalize (as done during training)
    mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

    with torch.no_grad():
        encoded = vae.encoder(mel_spec)
        mu = vae.mu(encoded)
        log_var = vae.log_var(encoded)
        latent_vector = vae.reparameterize(mu, log_var)

    return latent_vector.numpy()


def generate_from_latent(latent_vector, vae):
    """
    Decodes a given latent representation using the VAE's decoder.

    Args:
        vae (VAE): A trained VAE model.
        latent_vector (np.ndarray or torch.Tensor): Latent representation with shape (time, latent_dim).

    Returns:
        torch.Tensor: Generated mel spectrogram with shape (time, input_dim).
    """
    if not isinstance(latent_vector, torch.Tensor):
        latent_tensor = torch.tensor(latent_vector, dtype=torch.float32)
    else:
        latent_tensor = latent_vector

    with torch.no_grad():
        generated_mel = vae.decoder(latent_tensor)
    return generated_mel

# -------------------------
# 4. Invert Mel Spectrogram to Audio
# -------------------------


def invert_mel_to_audio(mel_spec, sample_rate=16000, n_fft=400, hop_length=160):
    """
    Converts a mel spectrogram to a waveform using InverseMelScale and GriffinLim.

    Args:
        mel_spec (torch.Tensor): Mel spectrogram of shape (n_mels, time).
        sample_rate (int): Sample rate.
        n_fft (int): FFT window size.
        hop_length (int): Hop length.

    Returns:
        torch.Tensor: Reconstructed waveform.
    """
    # InverseMelScale expects a mel spectrogram with shape (n_mels, time)
    # and outputs a linear spectrogram of shape (n_stft, time), where n_stft = n_fft//2 + 1.
    n_mels = mel_spec.shape[0]
    inverse_mel = T.InverseMelScale(
        n_stft=(n_fft // 2 + 1), n_mels=n_mels, sample_rate=sample_rate)
    linear_spec = inverse_mel(mel_spec)

    # Use GriffinLim to convert the linear spectrogram to a waveform.
    griffin_lim = T.GriffinLim(n_fft=n_fft, hop_length=hop_length)
    waveform = griffin_lim(linear_spec)
    return waveform


# -------------------------
# 5. Main: Encode, Generate, and Invert
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode and generate from a VAE on mel spectrogram representations."
    )
    parser.add_argument("wav_file", type=str, help="Path to input WAV file")
    parser.add_argument("--model", type=str, default="vae_mfcc.pth",
                        help="Path to a trained VAE model")
    args = parser.parse_args()

    # --- Encoding Phase ---
    mel_spec = extract_mel(args.wav_file)  # shape: (n_mels, time)
    mel_spec = mel_spec.permute(1, 0)        # shape: (time, n_mels)
    mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

    input_dim = mel_spec.shape[1]
    data = mel_spec.view(mel_spec.shape[0], -1)  # (time, n_mels)

    # Load the trained VAE model (make sure it was trained on mel spectrogram frames)
    vae = VAE(input_dim, latent_dim=2)
    vae.load_state_dict(torch.load(
        args.model, map_location=torch.device("cpu")))
    vae.eval()

    latent_representation = encode_audio(args.wav_file, vae)
    print("Encoded Latent Representation:")
    print(latent_representation)

    # --- Generation Phase ---
    generated_mel = generate_from_latent(latent_representation, vae)

    # Visualize the generated mel spectrogram.
    plt.figure(figsize=(8, 4))
    # Transpose to display with mel bins on the vertical axis.
    plt.imshow(generated_mel.T.numpy(), aspect='auto', origin='lower')
    plt.title("Generated Mel Spectrogram")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Frequency Bins")
    plt.colorbar()
    plt.show()

    # --- Inversion Phase: Convert Generated Mel Spectrogram to Audio ---
    # The generated mel spectrogram is of shape (time, n_mels); we need to transpose it to (n_mels, time).
    gen_mel = generated_mel.T  # shape: (n_mels, time)
    reconstructed_waveform = invert_mel_to_audio(
        gen_mel, sample_rate=16000, n_fft=400, hop_length=160)

    # Save the generated audio to a WAV file.
    sf.write("generated_audio.wav", reconstructed_waveform.numpy(), 16000)
    print("Generated audio saved as 'generated_audio.wav'")
