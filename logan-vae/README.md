# Audio Synthesis with Variational Autoencoder (VAE)

This project uses a Variational Autoencoder (VAE) to encode, decode, and interpolate audio files in the latent space. The project consists of three main scripts:

1. **`vae.py`**: Trains a VAE on MFCC representations of audio files.
2. **`encode_audio.py`**: Encodes an audio file into the latent space using a pre-trained VAE.
3. **`interpolate_latent.py`**: Interpolates between two audio files in the latent space and generates a new audio file.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Scripts](#scripts)
    - [vae.py](#vaepy)
    - [encode_audio.py](#encode_audiopy)
    - [interpolate_latent.py](#interpolate_latentpy)
3. [Usage](#usage)
4. [Notes](#notes)

---

## Requirements

To run the scripts, you need the following Python libraries:

-   `torch`
-   `numpy`
-   `soundfile`
-   `torchaudio`
-   `matplotlib` (optional, for visualization)

You can install the required libraries using `pip`:

```bash
pip install torch numpy soundfile torchaudio matplotlib
```

## Scripts

### `vae.py`

This script trains a Variational Autoencoder (VAE) on MFCC representations of audio files.

### Key Functions:

-   `extract_mfcc(wav_file)`: Extracts MFCC features from an audio file.
-   `VAE(input_dim, latent_dim=2)`: Defines the VAE architecture.
-   `train_vae(model, data, epochs=50, batch_size=32, learning_rate=1e-3)`: Trains the VAE on MFCC data.

### Usage:

```bash
python vae.py <wav_file>
```

### Output

-   A trained VAE model saved as `vae_mfcc.pth`.

### `encode_audio.py`

This script encodes an audio file into the latent space using a pre-trained VAE.

### Key Functions:

-   `encode_audio(wav_file, model_path="vae_mfcc.pth")`: Encodes an audio file into the latent space.

### Usage

```bash
python encode_audio.py <wav_file> [--model <model_path>]
```

### Output

-   The latent space representation of the input audio as a NumPy array.

### `interpolate_latent.py`

This script interpolates between two audio files in the latent space and generates a new audio file.

### Key Functions:

-   `interpolate_latent(wav1, wav2, model_path="vae_mfcc.pth", steps=10, output_wav="interpolated.wav")`: Interpolates between two audio files and generates a new audio file.

### Usage

```bash
python interpolate_latent.py <wav1> <wav2> [--model <model_path>] [--steps <steps>] [--output <output_wav>]
```

### Example

```bash
python interpolate_latent.py audio1.wav audio2.wav --model vae_mfcc.pth --steps 10 --output transition.wav
```

### Output

-   An interpolated audio file (e.g., `transition.wav`).

## Usage

### Training the VAE

1. Train the VAE on an audio file using `vae.py`:

```bash
python vae.py example.wav
```

### Encoding an Audio File

2. Encode an audio file into the latent space using `encode_audio.py`:

```bash
python encode_audio.py example.wav --model vae_mfcc.pth
```

### Interpolating Between Two Audio Files

3. Interpolate between two audio files and generate a new audio file using `interpolate_latent.py`:

```bash
python interpolate_latent.py audio1.wav audio2.wav --model vae_mfcc.pth --steps 10 --output transition.wav
```

## Notes

1. **Pre-trained Model**:

    - Ensure that the pre-trained VAE model (`vae_mfcc.pth`) matches the architecture defined in the `VAE` class.
    - The model must be trained using the same MFCC parameters (e.g., `n_mfcc=13`).

2. **MFCC Parameters**:

    - The scripts use default MFCC parameters (`n_mfcc=13`, `n_fft=400`, `hop_length=160`). These must match the parameters used during training.

3. **Griffin-Lim Algorithm**:

    - The Griffin-Lim algorithm is used to reconstruct the waveform from the MFCCs. This is an approximation and may not perfectly reconstruct the original audio.

4. **Interpolation Steps**:
    - The number of interpolation steps (`--steps`) controls the granularity of the transition. A higher number of steps results in a smoother transition.

---

## Next Steps

-   Experiment with different interpolation steps to achieve the desired transition effect.
-   Visualize the latent space to better understand the interpolation process.
-   Extend the scripts to handle more than two audio files for interpolation.
-   Use a pre-trained vocoder (e.g., WaveNet, HiFi-GAN) for higher-quality audio reconstruction.
