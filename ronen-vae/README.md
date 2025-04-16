# Generative Synth Engine - Ronen VAE

This repository contains the implementation of a Variational Autoencoder (VAE) for generative synthesis tasks. The project is designed to explore and create novel audio synthesis techniques using machine learning.

## Folder Structure

- **`/data`**: Contains datasets used for training and testing the VAE model.
- **`/models`**: Includes the VAE model architecture and pre-trained weights.
- **`/scripts`**: Utility scripts for data preprocessing, training, and evaluation.
- **`/notebooks`**: Jupyter notebooks for experimentation and visualization.
- **`/results`**: Stores generated audio samples and evaluation metrics.

## Features

- Variational Autoencoder for audio synthesis.
- Preprocessing pipeline for audio datasets.
- Tools for training, evaluation, and visualization.
- Pre-trained models for quick experimentation.

## Requirements

- Python 3.8+
- TensorFlow or PyTorch (depending on the implementation)
- NumPy, SciPy, and other dependencies listed in `requirements.txt`.

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ronen-vae.git
    cd ronen-vae
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Train the model:
    ```bash
    python scripts/train.py --config configs/train_config.json
    ```

4. Generate audio samples:
    ```bash
    python scripts/generate.py --model models/pretrained_model.pth
    ```

## File Explanations

- **inference.py**  
  Contains routines to run inference using the trained Variational Autoencoder. This script loads a pre-trained model and generates new audio samples based on provided inputs and command-line parameters. Use this file when you want to produce audio from the model without retraining.

- **model.py**  
  Defines the architecture of the Variational Autoencoder, including both the encoder and decoder networks. It also includes loss functions and helper methods needed for training and inference. This is the core component that constructs the model used in both training and generating new samples.

- **nsynth_dataset.py**  
  Implements a custom dataset class tailored for handling the NSynth audio dataset. This file manages loading, preprocessing, and batching audio samples to streamline the data pipeline for training and evaluation. It ensures that audio data is correctly formatted and normalized before being fed into the model.

- **train.py**  
  Orchestrates the training process for the VAE model. It handles parsing of command-line arguments (such as configuration files), initializes the model and dataset, and contains the main training loop that optimizes the model parameters. Run this script to begin training the model on your audio dataset.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments

Inspired by advancements in generative audio synthesis and the work of the research community.

