import argparse
import torch
import numpy as np
import soundfile as sf
import torchaudio.transforms as T
from vae import VAE, extract_mfcc  # Ensure these functions are in your project

def interpolate_latent(wav1, wav2, model_path="vae_mfcc.pth", steps=10, output_wav="interpolated.wav"):
    """
    Interpolates between two latent space representations and generates audio.

    Args:
        wav1 (str): Path to the first WAV file.
        wav2 (str): Path to the second WAV file.
        model_path (str): Path to the trained VAE model.
        steps (int): Number of interpolation steps.
        output_wav (str): Name of the output interpolated WAV file.
    """
    # Load trained VAE
    vae = VAE(input_dim=13)  # TODO: Ensure this matches training config
    vae.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    vae.eval()

    # Extract MFCCs and encode both files
    mfcc1 = extract_mfcc(wav1).permute(1, 0)  # Shape: (time, n_mfcc)
    mfcc2 = extract_mfcc(wav2).permute(1, 0)

    # Normalize MFCCs (same as training)
    mfcc1 = (mfcc1 - mfcc1.mean()) / mfcc1.std()
    mfcc2 = (mfcc2 - mfcc2.mean()) / mfcc2.std()

    # Ensure both MFCCs have the same length by truncating the longer one
    min_length = min(mfcc1.size(0), mfcc2.size(0))
    mfcc1 = mfcc1[:min_length, :]
    mfcc2 = mfcc2[:min_length, :]

    # Encode to latent space
    with torch.no_grad():
        latent1 = vae.reparameterize(vae.mu(vae.encoder(mfcc1)), vae.log_var(vae.encoder(mfcc1)))
        latent2 = vae.reparameterize(vae.mu(vae.encoder(mfcc2)), vae.log_var(vae.encoder(mfcc2)))

    # Generate interpolated latent vectors
    interpolated_mfccs = []
    for alpha in np.linspace(0, 1, steps):
        latent_interp = (1 - alpha) * latent1 + alpha * latent2
        with torch.no_grad():
            interpolated_mfcc = vae.decoder(latent_interp)
        interpolated_mfccs.append(interpolated_mfcc)

    # Convert MFCCs back to waveform using Griffin-Lim
    def mfcc_to_audio(mfcc, sample_rate=16000, n_mfcc=13, n_fft=400, n_iter=32):
        # Create a MelSpectrogram transform
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=None,
            hop_length=None,
            n_mels=n_mfcc,
        )

        # Inverse MelSpectrogram using Griffin-Lim
        griffin_lim = T.GriffinLim(
            n_fft=n_fft,
            n_iter=n_iter,
            win_length=None,
            hop_length=None,
        )

        # Reconstruct a pseudo-Mel-spectrogram from MFCCs
        # MFCCs are the result of applying DCT to the log-Mel-spectrogram
        # To reverse this, we need to approximate the log-Mel-spectrogram
        log_mel_spec = torch.exp(mfcc.T)  # Exponentiate to reverse the log operation

        # Pad the log-Mel-spectrogram to match the expected shape for Griffin-Lim
        # Griffin-Lim expects a spectrogram of shape (n_fft // 2 + 1, time)
        target_freq_dim = n_fft // 2 + 1
        if log_mel_spec.shape[0] < target_freq_dim:
            # Pad with zeros to match the target frequency dimension
            padding = torch.zeros((target_freq_dim - log_mel_spec.shape[0], log_mel_spec.shape[1]))
            log_mel_spec = torch.cat([log_mel_spec, padding], dim=0)
        elif log_mel_spec.shape[0] > target_freq_dim:
            # Truncate to match the target frequency dimension
            log_mel_spec = log_mel_spec[:target_freq_dim, :]

        # Convert to linear scale
        mel_spec = log_mel_spec

        # Use Griffin-Lim to reconstruct the waveform
        waveform = griffin_lim(mel_spec)
        return waveform.unsqueeze(0)  # Add batch dimension

    # Generate interpolated audio
    interpolated_audio = torch.cat([mfcc_to_audio(mfcc) for mfcc in interpolated_mfccs], dim=1).squeeze(0).numpy()

    # Save interpolated audio
    sf.write(output_wav, interpolated_audio.T, 16000)
    print(f"Interpolated audio saved as {output_wav}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpolate between two latent space representations and generate audio.")
    parser.add_argument("wav1", type=str, help="Path to the first WAV file")
    parser.add_argument("wav2", type=str, help="Path to the second WAV file")
    parser.add_argument("--model", type=str, default="vae_mfcc.pth", help="Path to the trained VAE model")
    parser.add_argument("--steps", type=int, default=10, help="Number of interpolation steps")
    parser.add_argument("--output", type=str, default="interpolated.wav", help="Output WAV file name")

    args = parser.parse_args()
    interpolate_latent(args.wav1, args.wav2, args.model, args.steps, args.output)